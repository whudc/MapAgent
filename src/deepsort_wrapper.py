"""
DeepSORT 跟踪器包装器

基于官方 deep_sort 实现，提供简化的接口
"""
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from deep_sort import Tracker, Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric


class TrackState(str, Enum):
    """轨迹状态"""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class TrackedObject:
    """跟踪目标包装类"""
    track_id: int
    state: TrackState = TrackState.TENTATIVE
    positions: List[List[float]] = field(default_factory=list)
    velocities: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    obj_type: str = "Unknown"
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @property
    def last_position(self) -> Optional[List[float]]:
        return self.positions[-1] if self.positions else None

    @property
    def last_frame(self) -> Optional[int]:
        return self.frame_ids[-1] if self.frame_ids else None

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


class DeepSORTTracker:
    """
    DeepSORT 多目标跟踪器（基于官方实现）

    参数：
    - max_distance: 最大匹配距离
    - max_velocity: 最大速度
    - min_hits: 确认轨迹需要的最小命中次数
    - max_misses: 轨迹最大丢失帧数
    """

    def __init__(self,
                 map_api: Optional[Any] = None,
                 max_distance: float = 5.0,
                 max_velocity: float = 30.0,
                 frame_interval: float = 0.1,
                 min_hits: int = 2,
                 max_misses: int = 30,
                 use_map: bool = False,
                 lane_weight: float = 0.0,
                 max_lane_distance: float = 0.0,
                 max_iou_distance: float = 0.7,
                 budget: Optional[int] = 100,
                 ):
        # 创建距离度量（使用欧式距离，因为不使用外观特征）
        self.metric = NearestNeighborDistanceMetric(
            metric="euclidean",
            matching_threshold=max_distance * 10,  # 放大阈值，因为欧式距离较大
            budget=budget
        )

        # 创建官方 deep_sort Tracker
        self.tracker = Tracker(
            metric=self.metric,
            max_iou_distance=max_iou_distance,
            max_age=max_misses,
            n_init=min_hits
        )

        # 参数存储
        self.max_distance = max_distance
        self.max_velocity = max_velocity
        self.dt = frame_interval

        # 统计
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'matches': 0,
            'misses': 0,
            'new_tracks': 0,
            'deleted_tracks': 0,
        }

        self._prev_track_count = 0

    def update(self, detections: List[Dict], frame_id: int) -> Dict[int, TrackedObject]:
        """
        更新跟踪器

        参数
        ----------
        detections : List[Dict]
            检测列表，每个元素包含 'location', 'size', 'velocity', 'type', 'heading' 等
        frame_id : int
            当前帧 ID

        返回
        -------
        Dict[int, TrackedObject]
            活跃轨迹字典
        """
        # 解析检测数据并转换为 deep_sort Detection 格式
        parsed_dets = []
        for det in detections:
            pos = det.get('location') or det.get('position')
            size = det.get('size')
            heading = det.get('heading', 0.0)

            if pos is None:
                continue

            # 从 3D 边界框创建 Detection
            d = Detection.from_3d_object(
                location=pos,
                size=size if size is not None else [4.0, 2.0, 1.5],  # 默认车辆尺寸
                heading=heading,
                velocity=det.get('velocity', [0, 0, 0]),
                confidence=det.get('confidence', 1.0),
                obj_type=det.get('type', 'Unknown'),
                feature=None  # 不使用外观特征
            )
            # 将 frame_id 存储在 heading 属性中（用于后续提取）
            d.heading = float(frame_id)
            parsed_dets.append(d)

        # 记录之前的轨迹数
        self._prev_track_count = len(self.tracker.tracks)

        # 预测步骤
        self.tracker.predict()

        # 更新步骤
        self.tracker.update(parsed_dets)

        # 更新统计
        new_tracks = len(self.tracker.tracks) - self._prev_track_count
        if new_tracks > 0:
            self.stats['new_tracks'] += new_tracks

        # 转换为 TrackedObject 格式
        result = {}
        for track in self.tracker.tracks:
            if track.is_deleted():
                continue

            state = TrackState.CONFIRMED if track.is_confirmed() else TrackState.TENTATIVE

            # 从 track 中提取数据
            positions = track.positions if track.positions else [track.mean[:3].tolist()]
            velocities = track.velocities if track.velocities else [track.mean[4:6].tolist() + [0.0]]
            frame_ids = track.frame_ids if track.frame_ids else [frame_id]

            obj = TrackedObject(
                track_id=track.track_id,
                state=state,
                positions=positions,
                velocities=velocities,
                frame_ids=frame_ids,
                obj_type=track.obj_type,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            result[track.track_id] = obj

        self.stats['active_tracks'] = len(result)
        self.stats['total_tracks'] = self.tracker._next_id - 1

        return result

    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """获取活跃轨迹"""
        result = {}
        for track in self.tracker.tracks:
            if track.is_deleted():
                continue

            state = TrackState.CONFIRMED if track.is_confirmed() else TrackState.TENTATIVE
            positions = track.positions if track.positions else [track.mean[:3].tolist()]

            obj = TrackedObject(
                track_id=track.track_id,
                state=state,
                positions=positions,
                velocities=track.velocities if track.velocities else [track.mean[4:6].tolist() + [0.0]],
                frame_ids=track.frame_ids if track.frame_ids else [],
                obj_type=track.obj_type,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            result[track.track_id] = obj
        return result

    def get_confirmed_tracks(self) -> Dict[int, TrackedObject]:
        """获取确认状态的轨迹"""
        result = {}
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue

            positions = track.positions if track.positions else [track.mean[:3].tolist()]
            obj = TrackedObject(
                track_id=track.track_id,
                state=TrackState.CONFIRMED,
                positions=positions,
                velocities=track.velocities if track.velocities else [track.mean[4:6].tolist() + [0.0]],
                frame_ids=track.frame_ids if track.frame_ids else [],
                obj_type=track.obj_type,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            result[track.track_id] = obj
        return result

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        active = self.get_active_tracks()
        lengths = [t.age for t in active.values()]

        return {
            **self.stats,
            'current_active': len(active),
            'avg_track_length': np.mean(lengths) if lengths else 0,
            'max_track_length': max(lengths) if lengths else 0,
        }
