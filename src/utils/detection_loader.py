"""
检测结果加载器

用于加载和解析交通检测结果数据
支持两种数据格式：
1. json_results 格式（新格式）
2. result_all_V1 格式（原始格式）
包含车辆跟踪功能
"""

import json
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class DataFormat(str, Enum):
    """数据格式类型"""
    JSON_RESULTS = "json_results"      # 新格式: data/json_results
    RESULT_ALL_V1 = "result_all_v1"    # 原格式: data/00/annotations/result_all_V1


@dataclass
class DetectedObject:
    """检测到的单个对象"""
    id: int                                    # 检测ID（每帧唯一）
    type: str                                  # 车辆类型
    location: Tuple[float, float, float]       # 位置
    size: Tuple[float, float, float]           # 尺寸
    rotation: Tuple[float, float, float]       # 旋转角度
    velocity: Tuple[float, float, float]       # 速度
    heading: float = 0.0                       # 航向角
    score: float = 1.0                         # 检测置信度
    tracking_id: int = -1                      # 跟踪ID（跨帧唯一，-1表示未分配）
    attribute: Dict[str, Any] = field(default_factory=dict)
    num_points: Dict[str, int] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        """获取速度大小"""
        vx, vy, vz = self.velocity
        return (vx**2 + vy**2 + vz**2) ** 0.5

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "location": list(self.location),
            "size": list(self.size),
            "rotation": list(self.rotation),
            "velocity": list(self.velocity),
            "heading": self.heading,
            "speed": self.speed,
            "score": self.score,
            "tracking_id": self.tracking_id,
            "attribute": self.attribute
        }


@dataclass
class TrackedObject:
    """跟踪对象"""
    track_id: int
    object_type: str
    positions: List[Tuple[float, float, float]]  # 历史位置列表
    velocities: List[Tuple[float, float, float]]  # 历史速度列表
    headings: List[float]                         # 历史航向角列表
    last_frame_id: int                            # 最后出现的帧ID
    total_frames: int = 1                         # 总出现帧数

    def predict_position(self) -> Tuple[float, float, float]:
        """预测下一帧位置"""
        if len(self.positions) < 2:
            return self.positions[-1]

        # 使用速度预测
        if self.velocities:
            vx, vy, vz = self.velocities[-1]
            last_pos = self.positions[-1]
            # 假设帧间隔0.1秒
            dt = 0.1
            return (last_pos[0] + vx * dt, last_pos[1] + vy * dt, last_pos[2])

        return self.positions[-1]

    def add_observation(self, position: Tuple[float, float, float],
                        velocity: Tuple[float, float, float],
                        heading: float, frame_id: int):
        """添加观测"""
        self.positions.append(position)
        self.velocities.append(velocity)
        self.headings.append(heading)
        self.last_frame_id = frame_id
        self.total_frames += 1


class VehicleTracker:
    """
    车辆跟踪器

    使用贪心最近邻匹配进行跨帧跟踪
    """

    def __init__(self, max_distance: float = 5.0, max_frames_lost: int = 3):
        """
        初始化

        Args:
            max_distance: 最大匹配距离（米）
            max_frames_lost: 最大丢失帧数
        """
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.current_frame_id = -1

    def update(self, frame_id: int, detections: List[DetectedObject]) -> List[DetectedObject]:
        """
        更新跟踪状态

        Args:
            frame_id: 当前帧ID
            detections: 当前帧的检测对象列表

        Returns:
            更新了tracking_id的检测对象列表
        """
        self.current_frame_id = frame_id

        # 如果没有现有轨迹，初始化所有检测
        if not self.tracks:
            for det in detections:
                det.tracking_id = self._create_track(det, frame_id)
            return detections

        # 过滤掉太老的轨迹
        active_tracks = {
            tid: track for tid, track in self.tracks.items()
            if frame_id - track.last_frame_id <= self.max_frames_lost
        }

        if not active_tracks:
            # 所有轨迹都过期，创建新轨迹
            for det in detections:
                det.tracking_id = self._create_track(det, frame_id)
            return detections

        # 匹配检测到轨迹
        matched, unmatched_dets, unmatched_tracks = self._greedy_match(
            detections, active_tracks
        )

        # 更新匹配的轨迹
        for det, track_id in matched:
            track = self.tracks[track_id]
            track.add_observation(
                det.location, det.velocity, det.heading, frame_id
            )
            det.tracking_id = track_id

        # 为未匹配的检测创建新轨迹
        for det in unmatched_dets:
            det.tracking_id = self._create_track(det, frame_id)

        return detections

    def _greedy_match(self, detections: List[DetectedObject],
                      tracks: Dict[int, TrackedObject]) -> Tuple[List, List, List]:
        """
        使用贪心最近邻匹配

        Returns:
            (matched_pairs, unmatched_detections, unmatched_track_ids)
        """
        matched = []
        unmatched_dets = list(detections)
        unmatched_track_ids = set(tracks.keys())

        # 计算所有距离
        distances = []
        for i, det in enumerate(detections):
            for track_id, track in tracks.items():
                dist = self._calculate_distance(det, track)
                if dist < self.max_distance:
                    distances.append((dist, i, det, track_id))

        # 按距离排序，贪心匹配
        distances.sort(key=lambda x: x[0])

        used_dets = set()
        used_tracks = set()

        for dist, i, det, track_id in distances:
            if i not in used_dets and track_id not in used_tracks:
                matched.append((det, track_id))
                used_dets.add(i)
                used_tracks.add(track_id)

        # 未匹配的检测
        unmatched_dets = [d for i, d in enumerate(detections) if i not in used_dets]
        # 未匹配的轨迹ID
        unmatched_track_ids = unmatched_track_ids - used_tracks

        return matched, unmatched_dets, list(unmatched_track_ids)

    def _calculate_distance(self, det: DetectedObject, track: TrackedObject) -> float:
        """
        计算检测对象与轨迹之间的距离

        考虑：
        1. 位置距离
        2. 类型匹配
        3. 预测位置
        """
        # 类型不匹配直接返回大距离
        if det.type != track.object_type:
            return float('inf')

        # 使用预测位置
        predicted_pos = track.predict_position()

        # 计算欧式距离
        dx = det.location[0] - predicted_pos[0]
        dy = det.location[1] - predicted_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        return dist

    def _create_track(self, det: DetectedObject, frame_id: int) -> int:
        """创建新轨迹"""
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = TrackedObject(
            track_id=track_id,
            object_type=det.type,
            positions=[det.location],
            velocities=[det.velocity],
            headings=[det.heading],
            last_frame_id=frame_id,
            total_frames=1
        )

        return track_id

    def get_active_tracks(self, frame_id: int) -> Dict[int, TrackedObject]:
        """获取当前活跃的轨迹"""
        return {
            tid: track for tid, track in self.tracks.items()
            if frame_id - track.last_frame_id <= self.max_frames_lost
        }

    def cleanup_old_tracks(self, frame_id: int):
        """清理过期轨迹"""
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if frame_id - track.last_frame_id <= self.max_frames_lost * 2
        }

    def get_track_statistics(self) -> Dict:
        """获取跟踪统计信息"""
        return {
            "total_tracks": len(self.tracks),
            "track_lengths": {
                tid: track.total_frames
                for tid, track in self.tracks.items()
            }
        }


@dataclass
class FrameDetection:
    """单帧检测结果"""
    frame_id: int
    timestamp: Optional[float] = None
    token: Optional[str] = None
    sequence: Optional[str] = None
    objects: List[DetectedObject] = field(default_factory=list)
    ego_position: Optional[Tuple[float, float, float]] = None
    ego_velocity: Optional[Tuple[float, float, float]] = None
    ego_transform: Optional[List[List[float]]] = None

    @property
    def vehicle_count(self) -> int:
        """车辆数量"""
        vehicle_types = ['Car', 'Truck', 'Bus', 'Suv', 'Non_motor_rider', 'Motorcycle']
        return len([o for o in self.objects if o.type in vehicle_types])

    def get_objects_by_type(self, type_name: str) -> List[DetectedObject]:
        """按类型获取对象"""
        return [o for o in self.objects if o.type == type_name]

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "token": self.token,
            "objects": [o.to_dict() for o in self.objects],
            "vehicle_count": self.vehicle_count,
            "ego_position": list(self.ego_position) if self.ego_position else None,
            "ego_velocity": list(self.ego_velocity) if self.ego_velocity else None
        }


class DetectionLoader:
    """
    检测结果加载器

    支持加载指定目录下的检测结果文件
    自动检测数据格式
    支持车辆跟踪
    """

    def __init__(self, detection_path: str, enable_tracking: bool = True,
                 ego_transform_path: Optional[str] = None):
        """
        初始化

        Args:
            detection_path: 检测结果目录路径
            enable_tracking: 是否启用跟踪（默认True）
            ego_transform_path: ego 变换矩阵目录路径（用于 json_results 格式的坐标变换）
        """
        self.detection_path = Path(detection_path)
        self._frame_files: Dict[int, Path] = {}
        self._loaded_frames: Dict[int, FrameDetection] = {}
        self._data_format: Optional[DataFormat] = None
        self._enable_tracking = enable_tracking
        self._tracker = VehicleTracker() if enable_tracking else None
        self._tracking_done = False

        # ego 变换矩阵目录（用于 json_results 格式）
        self._ego_transform_path = Path(ego_transform_path) if ego_transform_path else None
        self._ego_transforms: Dict[int, List[List[float]]] = {}

        self._scan_directory()

    def _scan_directory(self):
        """扫描目录，建立帧ID到文件的映射"""
        if not self.detection_path.exists():
            raise FileNotFoundError(f"检测结果目录不存在: {self.detection_path}")

        # 扫描所有JSON文件
        for file in self.detection_path.glob("*.json"):
            # 尝试匹配新格式: 00_000000.json
            match_new = re.match(r"(\d+)_(\d+)\.json", file.name)
            # 尝试匹配原格式: 000002.json
            match_old = re.match(r"(\d+)\.json", file.name)

            if match_new:
                # 新格式: 使用后半部分作为帧ID
                frame_id = int(match_new.group(2))
                self._data_format = DataFormat.JSON_RESULTS
                self._frame_files[frame_id] = file
            elif match_old:
                # 原格式
                frame_id = int(match_old.group(1))
                self._frame_files[frame_id] = file

        # 按帧ID排序
        self._sorted_frame_ids = sorted(self._frame_files.keys())

        # 如果还没确定格式，加载第一个文件检测
        if self._data_format is None and self._frame_files:
            first_file = self._frame_files[self._sorted_frame_ids[0]]
            try:
                with open(first_file, 'r') as f:
                    data = json.load(f)
                if 'detections' in data:
                    self._data_format = DataFormat.JSON_RESULTS
                else:
                    self._data_format = DataFormat.RESULT_ALL_V1
            except Exception:
                self._data_format = DataFormat.RESULT_ALL_V1

        # 对于 JSON_RESULTS 格式，自动查找 ego 变换文件
        if self._data_format == DataFormat.JSON_RESULTS and self._ego_transform_path is None:
            self._auto_find_ego_transform_path()

        # 加载 ego 变换矩阵
        if self._data_format == DataFormat.JSON_RESULTS and self._ego_transform_path:
            self._load_ego_transforms()

    def _auto_find_ego_transform_path(self):
        """自动查找 ego 变换矩阵目录"""
        # 尝试常见的路径
        possible_paths = [
            self.detection_path.parent / "00" / "annotations" / "result_all_V1",
            self.detection_path.parent / "annotations" / "result_all_V1",
            self.detection_path.parent.parent / "00" / "annotations" / "result_all_V1",
            Path("data/00/annotations/result_all_V1"),
        ]

        for path in possible_paths:
            if path.exists() and list(path.glob("*.json")):
                self._ego_transform_path = path
                print(f"自动找到 ego 变换目录: {path}")
                return

    def _load_ego_transforms(self):
        """加载所有帧的 ego 变换矩阵"""
        if not self._ego_transform_path or not self._ego_transform_path.exists():
            return

        for frame_id in self._sorted_frame_ids:
            # 构建对应的原始文件路径
            # 00_000000.json -> 000000.json
            transform_file = self._ego_transform_path / f"{frame_id:06d}.json"

            if not transform_file.exists():
                continue

            try:
                with open(transform_file, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)

                transform = data.get('ego2global_transformation_matrix')
                if transform:
                    self._ego_transforms[frame_id] = transform
            except Exception:
                continue

        if self._ego_transforms:
            print(f"加载了 {len(self._ego_transforms)} 个 ego 变换矩阵")

    def _transform_point(self, point: List[float], transform: List[List[float]]) -> List[float]:
        """使用变换矩阵变换点坐标"""
        if not transform or len(transform) < 4:
            return point

        x, y, z = point[0], point[1], point[2]
        global_x = transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3]
        global_y = transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3]
        global_z = transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3]

        return [global_x, global_y, global_z]

    def get_data_format(self) -> DataFormat:
        """获取数据格式"""
        return self._data_format

    def get_frame_count(self) -> int:
        """获取帧数量"""
        return len(self._frame_files)

    def get_frame_ids(self) -> List[int]:
        """获取所有帧ID列表"""
        return self._sorted_frame_ids

    def get_timestamps(self) -> List[float]:
        """获取所有时间戳列表"""
        timestamps = []
        for frame_id in self._sorted_frame_ids:
            frame = self.load_frame(frame_id)
            if frame and frame.timestamp:
                timestamps.append(frame.timestamp)
        return timestamps

    def load_frame(self, frame_id: int, use_tracking: bool = True) -> Optional[FrameDetection]:
        """
        加载单帧检测结果

        Args:
            frame_id: 帧ID
            use_tracking: 是否应用跟踪ID

        Returns:
            FrameDetection 对象
        """
        # 如果已完成跟踪，直接返回缓存
        if self._tracking_done and frame_id in self._loaded_frames:
            return self._loaded_frames[frame_id]

        # 查找文件
        if frame_id not in self._frame_files:
            return None

        file_path = self._frame_files[frame_id]

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 根据数据格式解析
            if self._data_format == DataFormat.JSON_RESULTS:
                frame = self._parse_json_results_format(frame_id, data)
            else:
                frame = self._parse_result_all_format(frame_id, data)

            # 应用跟踪
            if use_tracking and self._tracker:
                frame.objects = self._tracker.update(frame_id, frame.objects)

            self._loaded_frames[frame_id] = frame
            return frame

        except Exception as e:
            print(f"加载帧 {frame_id} 失败: {e}")
            return None

    def run_tracking(self, start_frame: int = None, end_frame: int = None):
        """
        运行跟踪算法

        必须在加载帧之前调用，对指定范围的帧进行跟踪

        Args:
            start_frame: 起始帧ID
            end_frame: 结束帧ID
        """
        if not self._tracker:
            print("跟踪未启用")
            return

        print("运行车辆跟踪...")

        # 重置跟踪器
        self._tracker = VehicleTracker()
        self._loaded_frames.clear()
        self._tracking_done = False

        # 获取帧范围
        start = start_frame if start_frame is not None else (self._sorted_frame_ids[0] if self._sorted_frame_ids else 0)
        end = end_frame if end_frame is not None else (self._sorted_frame_ids[-1] if self._sorted_frame_ids else 0)

        # 按顺序处理每一帧
        for frame_id in self._sorted_frame_ids:
            if frame_id < start or frame_id > end:
                continue

            # 解析原始数据（不应用跟踪）
            file_path = self._frame_files[frame_id]
            with open(file_path, 'r') as f:
                data = json.load(f)

            if self._data_format == DataFormat.JSON_RESULTS:
                frame = self._parse_json_results_format(frame_id, data)
            else:
                frame = self._parse_result_all_format(frame_id, data)

            # 应用跟踪
            frame.objects = self._tracker.update(frame_id, frame.objects)
            self._loaded_frames[frame_id] = frame

        self._tracking_done = True

        # 输出统计
        stats = self._tracker.get_track_statistics()
        print(f"跟踪完成: 共 {stats['total_tracks']} 个轨迹")

    def _parse_json_results_format(self, frame_id: int, data: Dict) -> FrameDetection:
        """解析新格式数据 (json_results)

        注意: json_results 格式的坐标已经是全局坐标系，不需要应用 ego 变换。
        检测坐标是 ego 全局位置 + 局部偏移，直接使用即可。
        """
        token = data.get('token', '')
        sequence = data.get('sequence', '')

        # json_results 格式的坐标已经是全局坐标系，不需要 ego 变换

        # 解析检测对象
        objects = []
        detections = data.get('detections', [])

        for det in detections:
            try:
                pos = det.get('position', {})
                size = det.get('size', {})
                vel = det.get('velocity', {})

                # json_results 坐标已经是全局坐标，直接使用
                location = [
                    pos.get('x', 0.0),
                    pos.get('y', 0.0),
                    pos.get('z', 0.0)
                ]

                # 速度
                velocity = [
                    vel.get('vx', 0.0),
                    vel.get('vy', 0.0),
                    0.0
                ]

                obj = DetectedObject(
                    id=det.get('id', 0),
                    type=det.get('class', 'Unknown'),
                    location=tuple(location),
                    size=(
                        size.get('length', 4.0),
                        size.get('width', 2.0),
                        size.get('height', 1.5)
                    ),
                    rotation=(0.0, 0.0, det.get('heading', 0.0)),
                    velocity=tuple(velocity),
                    heading=det.get('heading', 0.0),
                    score=det.get('score', 1.0),
                    tracking_id=-1  # 初始化为未分配
                )
                objects.append(obj)
            except Exception:
                continue

        return FrameDetection(
            frame_id=frame_id,
            timestamp=None,
            token=token,
            sequence=sequence,
            objects=objects
        )

    def _parse_result_all_format(self, frame_id: int, data: Dict) -> FrameDetection:
        """解析原格式数据 (result_all_V1)"""
        # 提取时间戳
        timestamp = data.get('timestamp')
        if timestamp:
            # 转换为秒（如果是纳秒）
            if timestamp > 1e12:
                timestamp = timestamp / 1e9

        # 提取自车信息
        ego_position = None
        ego_velocity = None
        ego_transform = data.get('ego2global_transformation_matrix')

        if ego_transform:
            ego_position = (
                ego_transform[0][3],
                ego_transform[1][3],
                ego_transform[2][3]
            )

        ego_vel = data.get('ego_velocity')
        if ego_vel:
            ego_velocity = tuple(ego_vel)

        # 解析检测对象
        objects = []
        obj_data = data.get('objects', [])

        if isinstance(obj_data, list):
            for obj in obj_data:
                detected_obj = self._parse_object_old_format(obj, ego_transform)
                if detected_obj:
                    objects.append(detected_obj)

        return FrameDetection(
            frame_id=frame_id,
            timestamp=timestamp,
            objects=objects,
            ego_position=ego_position,
            ego_velocity=ego_velocity,
            ego_transform=ego_transform
        )

    def _parse_object_old_format(self, obj: Dict,
                                   ego_transform: Optional[List[List[float]]] = None) -> Optional[DetectedObject]:
        """解析原格式的单个检测对象"""
        try:
            location = obj.get('location')
            size = obj.get('size')
            rotation = obj.get('rotation')
            velocity = obj.get('velocity')

            if location is None:
                return None

            # 应用 ego2global 坐标变换
            if ego_transform:
                location = self._transform_point(list(location), ego_transform)

            return DetectedObject(
                id=obj.get('id', 0),
                type=obj.get('type', 'Unknown'),
                location=tuple(location) if location else (0.0, 0.0, 0.0),
                size=tuple(size) if size else (4.0, 2.0, 1.5),
                rotation=tuple(rotation) if rotation else (0.0, 0.0, 0.0),
                velocity=tuple(velocity) if velocity else (0.0, 0.0, 0.0),
                heading=rotation[2] if rotation else 0.0,
                tracking_id=-1,
                attribute=obj.get('attribute', {}),
                num_points=obj.get('num_points', {})
            )
        except Exception:
            return None

    def load_frames(self, start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None) -> List[FrameDetection]:
        """加载指定范围的帧"""
        if not self._sorted_frame_ids:
            return []

        start = start_frame or self._sorted_frame_ids[0]
        end = end_frame or self._sorted_frame_ids[-1]

        # 如果启用跟踪且未完成，先运行跟踪
        if self._enable_tracking and not self._tracking_done:
            self.run_tracking(start, end)

        frames = []
        for frame_id in self._sorted_frame_ids:
            if frame_id >= start and frame_id <= end:
                frame = self._loaded_frames.get(frame_id) or self.load_frame(frame_id)
                if frame:
                    frames.append(frame)

        return frames

    def load_all_frames(self) -> List[FrameDetection]:
        """加载所有帧"""
        return self.load_frames()

    def get_vehicle_ids(self) -> List[int]:
        """获取所有出现过的车辆跟踪ID"""
        vehicle_types = ['Car', 'Truck', 'Bus', 'Suv', 'Non_motor_rider', 'Motorcycle']
        vehicle_ids = set()

        for frame_id in self._sorted_frame_ids:
            frame = self._loaded_frames.get(frame_id) or self.load_frame(frame_id)
            if frame:
                for obj in frame.objects:
                    if obj.type in vehicle_types and obj.tracking_id > 0:
                        vehicle_ids.add(obj.tracking_id)

        return sorted(vehicle_ids)

    def get_object_types(self) -> Dict[str, int]:
        """获取对象类型统计"""
        type_counts = {}
        for frame_id in self._sorted_frame_ids:
            frame = self._loaded_frames.get(frame_id) or self.load_frame(frame_id)
            if frame:
                for obj in frame.objects:
                    type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
        return type_counts

    def get_summary(self) -> Dict:
        """获取数据摘要"""
        return {
            "path": str(self.detection_path),
            "format": self._data_format.value if self._data_format else None,
            "total_frames": self.get_frame_count(),
            "frame_ids": self.get_frame_ids(),
            "total_tracks": len(self.get_vehicle_ids()),
            "object_types": self.get_object_types(),
            "tracking_enabled": self._enable_tracking
        }

    def get_tracker_statistics(self) -> Optional[Dict]:
        """获取跟踪器统计信息"""
        if self._tracker:
            return self._tracker.get_track_statistics()
        return None

    def _transform_point(self, point: List[float], transform: List[List[float]]) -> List[float]:
        """
        使用 4x4 变换矩阵将点从局部坐标系转换到全局坐标系

        Args:
            point: 局部坐标系中的点 [x, y, z]
            transform: 4x4 齐次变换矩阵

        Returns:
            全局坐标系中的点 [x, y, z]
        """
        if not transform or len(transform) < 4:
            return point

        x, y, z = point[0], point[1], point[2]

        # 应用 4x4 变换矩阵
        global_x = transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3]
        global_y = transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3]
        global_z = transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3]

        return [global_x, global_y, global_z]