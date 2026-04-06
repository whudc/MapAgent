"""
交通流重建 Agent

基于位置跟踪的轨迹重建，结合地图信息和车道拓扑约束
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import json
import time
from pathlib import Path
import numpy as np

from agents.base import BaseAgent, AgentContext
from agents.position_tracker import PositionTracker, TrackedObject, TrackState, LaneAssignment
from models.agent_io import (
    TrafficFlowQuery, TrafficFlowResult,
    VehicleState, VehicleTrajectory, FrameData
)
from models.map_data import MapLoader, VectorMap
from apis.map_api import MapAPI
from utils.detection_loader import DetectionLoader, FrameDetection, DetectedObject


@dataclass
class Trajectory:
    """轨迹数据结构"""
    track_id: int
    positions: List[List[float]] = field(default_factory=list)
    velocities: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    obj_types: List[str] = field(default_factory=list)
    raw_data: List[Dict] = field(default_factory=list)
    lane_assignments: List[Dict] = field(default_factory=list)
    dominant_lane: str = ""

    @property
    def length(self) -> int:
        return len(self.frame_ids)

    @property
    def start_frame(self) -> int:
        return self.frame_ids[0] if self.frame_ids else -1

    @property
    def end_frame(self) -> int:
        return self.frame_ids[-1] if self.frame_ids else -1

    @property
    def dominant_type(self) -> str:
        if not self.obj_types:
            return "Unknown"
        from collections import Counter
        return Counter(self.obj_types).most_common(1)[0][0]

    def get_position_at_frame(self, frame_id: int) -> Optional[List[float]]:
        """获取指定帧的位置"""
        if frame_id in self.frame_ids:
            idx = self.frame_ids.index(frame_id)
            return self.positions[idx]
        return None

    def get_lane_at_frame(self, frame_id: int) -> Optional[str]:
        """获取指定帧的车道"""
        if frame_id in self.frame_ids:
            idx = self.frame_ids.index(frame_id)
            if idx < len(self.lane_assignments):
                return self.lane_assignments[idx].get('lane_id', '')
        return None

    def to_vehicle_trajectory(self) -> VehicleTrajectory:
        """转换为 VehicleTrajectory 格式"""
        states = []
        for i, frame_id in enumerate(self.frame_ids):
            state = VehicleState(
                frame_id=frame_id,
                position=self.positions[i],
                velocity=self.velocities[i] if i < len(self.velocities) else [0, 0, 0],
                heading=0.0,
                speed=0.0,
            )
            states.append(state)

        return VehicleTrajectory(
            vehicle_id=self.track_id,
            vehicle_type=self.dominant_type,
            states=states,
        )


class TrafficFlowAgent(BaseAgent):
    """
    交通流重建 Agent

    功能：
    - 加载检测结果数据和地图数据
    - 基于位置跟踪重建车辆轨迹（结合车道拓扑约束）
    - 使用地图信息辅助轨迹预测和关联
    - 分析整体交通流
    - 保存重建结果
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "traffic_flow_agent"
        self._loader: Optional[DetectionLoader] = None
        self._tracker: Optional[PositionTracker] = None
        self._trajectories: Dict[int, Trajectory] = {}
        self._frames: List[FrameData] = []
        self._map_api: Optional[MapAPI] = None
        self._map_file: Optional[str] = None

    def get_tools(self) -> List[Dict]:
        """返回交通流重建相关工具"""
        return [
            {
                "name": "load_detection_results",
                "description": "加载检测结果数据",
                "parameters": {
                    "path": {"type": "string", "description": "检测结果目录路径"}
                },
                "handler": self._load_detection_results
            },
            {
                "name": "load_map_data",
                "description": "加载矢量地图数据",
                "parameters": {
                    "path": {"type": "string", "description": "地图文件路径，默认使用 data/vector_map.json"}
                },
                "handler": self._load_map_data
            },
            {
                "name": "reconstruct_traffic_flow",
                "description": "重建交通流轨迹（结合地图拓扑约束）",
                "parameters": {
                    "start_frame": {"type": "integer", "description": "起始帧ID", "default": None},
                    "end_frame": {"type": "integer", "description": "结束帧ID", "default": None},
                    "max_distance": {"type": "number", "description": "最大匹配距离(米)", "default": 5.0},
                    "max_velocity": {"type": "number", "description": "最大速度(m/s)", "default": 30.0},
                    "use_map": {"type": "boolean", "description": "是否使用地图约束", "default": True},
                },
                "handler": self._reconstruct_traffic_flow
            },
            {
                "name": "get_trajectory_by_id",
                "description": "获取指定车辆的轨迹",
                "parameters": {
                    "vehicle_id": {"type": "integer", "description": "车辆ID"}
                },
                "handler": self._get_trajectory_by_id
            },
            {
                "name": "analyze_vehicle_behavior",
                "description": "分析车辆行为",
                "parameters": {
                    "vehicle_id": {"type": "integer", "description": "车辆ID"},
                    "frame_id": {"type": "integer", "description": "帧ID"}
                },
                "handler": self._analyze_vehicle_behavior
            },
            {
                "name": "save_reconstruction_result",
                "description": "保存重建结果",
                "parameters": {
                    "output_path": {"type": "string", "description": "输出文件路径", "default": "reconstruction_result.json"}
                },
                "handler": self._save_reconstruction_result
            },
            {
                "name": "get_traffic_flow_summary",
                "description": "获取交通流摘要",
                "parameters": {},
                "handler": self._get_traffic_flow_summary
            },
            {
                "name": "get_lane_statistics",
                "description": "获取车道统计信息",
                "parameters": {},
                "handler": self._get_lane_statistics
            }
        ]

    def get_system_prompt(self) -> str:
        return """你是一个交通流分析专家，专门重建和分析车辆轨迹。

核心能力：
1. 加载检测结果数据（支持多种格式）
2. 加载矢量地图数据（车道线、中心线、拓扑关系）
3. 基于位置跟踪重建连续轨迹（结合车道拓扑约束）
4. 分析车辆行为和交通流模式

使用方法：
1. 首先调用 load_detection_results 加载检测数据
2. 调用 load_map_data 加载地图数据（可选，用于车道约束）
3. 然后调用 reconstruct_traffic_flow 重建轨迹
4. 使用 get_trajectory_by_id 查询特定轨迹
5. 使用 save_reconstruction_result 保存结果"""

    def _load_detection_results(self, path: str) -> Dict[str, Any]:
        """加载检测结果"""
        self._loader = DetectionLoader(path, enable_tracking=False)

        return {
            "success": True,
            "message": f"已加载检测结果",
            "data_format": self._loader.get_data_format(),
            "frame_count": self._loader.get_frame_count(),
        }

    def _load_map_data(self, path: str = None) -> Dict[str, Any]:
        """加载地图数据"""
        if path is None:
            # 使用默认地图
            default_path = Path(__file__).parent.parent.parent / "data" / "vector_map.json"
            if default_path.exists():
                path = str(default_path)
            else:
                return {
                    "success": False,
                    "error": f"默认地图文件不存在: {default_path}",
                }

        try:
            self._map_api = MapAPI(map_file=path)
            self._map_file = path

            summary = self._map_api.get_map_summary()

            return {
                "success": True,
                "message": f"已加载地图数据",
                "map_file": path,
                "map_summary": summary,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"加载地图失败: {str(e)}",
            }

    def _reconstruct_traffic_flow(self,
                                   start_frame: Optional[int] = None,
                                   end_frame: Optional[int] = None,
                                   max_distance: float = 5.0,
                                   max_velocity: float = 30.0,
                                   use_map: bool = True) -> Dict[str, Any]:
        """重建交通流"""
        if not self._loader:
            return {"success": False, "error": "请先加载检测结果"}

        # 加载帧数据
        frames = self._loader.load_frames(start_frame, end_frame)
        if not frames:
            return {"success": False, "error": "未加载到帧数据"}

        # 创建位置跟踪器
        self._tracker = PositionTracker(
            map_api=self._map_api,
            max_distance=max_distance,
            max_velocity=max_velocity,
            frame_interval=0.1,
            min_hits=2,
            max_misses=5,
            use_map=use_map and self._map_api is not None,
        )

        # 处理每一帧
        for frame in frames:
            detections = []
            for obj in frame.objects:
                d = obj.to_dict()
                pos = d.get('location') or d.get('position')
                if pos is not None:
                    detections.append({
                        'location': pos,
                        'type': d.get('type', 'Unknown'),
                        'heading': d.get('heading'),
                        'speed': d.get('speed'),
                    })

            self._tracker.update(detections, frame.frame_id)

        # 构建轨迹
        self._build_trajectories()

        stats = self._tracker.get_statistics()

        return {
            "success": True,
            "message": f"重建完成",
            "num_trajectories": len(self._trajectories),
            "use_map": stats.get('use_map', False),
            "statistics": stats,
        }

    def _build_trajectories(self):
        """从跟踪器构建轨迹"""
        tracks = self._tracker.get_active_tracks()

        self._trajectories = {}
        for track_id, tracked_obj in tracks.items():
            # 提取车道分配信息
            lane_assignments = []
            for la in tracked_obj.lane_assignments:
                lane_assignments.append({
                    'lane_id': la.lane_id,
                    'centerline_id': la.centerline_id,
                    'distance': la.distance,
                    'heading': la.heading,
                })

            trajectory = Trajectory(
                track_id=track_id,
                positions=tracked_obj.positions.copy(),
                velocities=tracked_obj.velocities.copy(),
                frame_ids=tracked_obj.frame_ids.copy(),
                obj_types=[tracked_obj.obj_type] * len(tracked_obj.frame_ids),
                lane_assignments=lane_assignments,
                dominant_lane=tracked_obj.dominant_lane,
            )
            self._trajectories[track_id] = trajectory

    def _get_trajectory_by_id(self, vehicle_id: int) -> Dict[str, Any]:
        """获取指定车辆的轨迹"""
        traj = self._trajectories.get(vehicle_id)
        if not traj:
            return {"success": False, "error": f"未找到轨迹 {vehicle_id}"}

        return {
            "success": True,
            "trajectory": {
                "track_id": traj.track_id,
                "length": traj.length,
                "start_frame": traj.start_frame,
                "end_frame": traj.end_frame,
                "type": traj.dominant_type,
                "positions": traj.positions,
                "frame_ids": traj.frame_ids,
            }
        }

    def _analyze_vehicle_behavior(self, vehicle_id: int, frame_id: int) -> Dict[str, Any]:
        """分析车辆行为"""
        traj = self._trajectories.get(vehicle_id)
        if not traj:
            return {"success": False, "error": f"未找到轨迹 {vehicle_id}"}

        pos = traj.get_position_at_frame(frame_id)
        if pos is None:
            return {"success": False, "error": f"轨迹 {vehicle_id} 在帧 {frame_id} 无数据"}

        # 计算速度和方向
        idx = traj.frame_ids.index(frame_id)
        velocity = traj.velocities[idx] if idx < len(traj.velocities) else [0, 0, 0]
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

        return {
            "success": True,
            "analysis": {
                "vehicle_id": vehicle_id,
                "frame_id": frame_id,
                "position": pos,
                "velocity": velocity,
                "speed": speed,
                "type": traj.dominant_type,
            }
        }

    def _save_reconstruction_result(self, output_path: str = "reconstruction_result.json") -> Dict[str, Any]:
        """保存重建结果"""
        result = self.get_reconstruction_result()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"结果已保存到 {output_path}",
            "path": output_path,
        }

    def _get_traffic_flow_summary(self) -> Dict[str, Any]:
        """获取交通流摘要"""
        if not self._trajectories:
            return {"success": False, "error": "请先重建交通流"}

        lengths = [t.length for t in self._trajectories.values()]
        types = [t.dominant_type for t in self._trajectories.values()]
        lanes = [t.dominant_lane for t in self._trajectories.values()]

        from collections import Counter
        type_counts = Counter(types)
        lane_counts = Counter([l for l in lanes if l])

        return {
            "success": True,
            "summary": {
                "total_trajectories": len(self._trajectories),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "vehicle_types": dict(type_counts),
                "lane_distribution": dict(lane_counts),
            }
        }

    def _get_lane_statistics(self) -> Dict[str, Any]:
        """获取车道统计信息"""
        if not self._trajectories:
            return {"success": False, "error": "请先重建交通流"}

        # 统计每个车道的轨迹数量和长度
        lane_stats: Dict[str, Dict] = {}
        for traj in self._trajectories.values():
            lane = traj.dominant_lane
            if lane:
                if lane not in lane_stats:
                    lane_stats[lane] = {
                        'trajectory_count': 0,
                        'total_length': 0,
                        'avg_length': 0,
                    }
                lane_stats[lane]['trajectory_count'] += 1
                lane_stats[lane]['total_length'] += traj.length

        # 计算平均值
        for lane, stats in lane_stats.items():
            if stats['trajectory_count'] > 0:
                stats['avg_length'] = stats['total_length'] / stats['trajectory_count']

        return {
            "success": True,
            "lane_statistics": lane_stats,
            "total_lanes_used": len(lane_stats),
        }

    def get_reconstruction_result(self) -> Dict[str, Any]:
        """获取重建结果"""
        trajectories_data = {}
        for tid, traj in self._trajectories.items():
            trajectories_data[tid] = {
                'track_id': tid,
                'length': traj.length,
                'start_frame': traj.start_frame,
                'end_frame': traj.end_frame,
                'type': traj.dominant_type,
                'positions': traj.positions,
                'frame_ids': traj.frame_ids,
                'dominant_lane': traj.dominant_lane,
                'lane_assignments': traj.lane_assignments,
            }

        stats = self._tracker.get_statistics() if self._tracker else {}

        return {
            'trajectories': trajectories_data,
            'statistics': {
                'total_trajectories': len(self._trajectories),
                'tracker_stats': stats,
                'use_map': stats.get('use_map', False),
            }
        }

    def get_trajectory_positions(self, track_id: int) -> Optional[np.ndarray]:
        """获取轨迹的位置序列"""
        traj = self._trajectories.get(track_id)
        if traj:
            return np.array(traj.positions)
        return None

    def get_trajectory_at_frame(self, frame_id: int) -> Dict[int, List[float]]:
        """获取指定帧的所有目标位置"""
        result = {}
        for tid, traj in self._trajectories.items():
            pos = traj.get_position_at_frame(frame_id)
            if pos is not None:
                result[tid] = pos
        return result


def reconstruct_traffic_flow(frames: List[Dict],
                             map_api: Optional[MapAPI] = None,
                             max_distance: float = 5.0,
                             max_velocity: float = 30.0,
                             use_map: bool = True) -> Dict:
    """
    便捷函数：重建交通流

    Args:
        frames: 帧数据列表，每帧包含 'frame_id' 和 'objects'
        map_api: 地图API实例（可选）
        max_distance: 最大匹配距离（米）
        max_velocity: 最大速度（米/秒）
        use_map: 是否使用地图信息

    Returns:
        重建结果
    """
    tracker = PositionTracker(
        map_api=map_api,
        max_distance=max_distance,
        max_velocity=max_velocity,
        frame_interval=0.1,
        min_hits=2,
        max_misses=5,
        use_map=use_map and map_api is not None,
    )

    for frame_data in frames:
        frame_id = frame_data.get('frame_id', 0)
        objects = frame_data.get('objects', [])

        detections = []
        for obj in objects:
            pos = obj.get('location') or obj.get('position')
            if pos is not None:
                detections.append({
                    'location': pos,
                    'type': obj.get('type', 'Unknown'),
                    'heading': obj.get('heading'),
                })

        tracker.update(detections, frame_id)

    tracks = tracker.get_active_tracks()

    trajectories = {}
    for track_id, tracked_obj in tracks.items():
        lane_assignments = []
        for la in tracked_obj.lane_assignments:
            if la.is_valid():
                lane_assignments.append({
                    'lane_id': la.lane_id,
                    'centerline_id': la.centerline_id,
                    'distance': la.distance,
                })

        trajectories[track_id] = {
            'track_id': track_id,
            'positions': tracked_obj.positions,
            'frame_ids': tracked_obj.frame_ids,
            'type': tracked_obj.obj_type,
            'length': len(tracked_obj.frame_ids),
            'dominant_lane': tracked_obj.dominant_lane,
            'lane_assignments': lane_assignments,
        }

    stats = tracker.get_statistics()

    return {
        'trajectories': trajectories,
        'statistics': stats,
        'use_map': stats.get('use_map', False),
    }