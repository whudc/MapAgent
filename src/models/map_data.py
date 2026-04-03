"""
矢量地图数据模型

定义加载 vector_map.json 的数据结构
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path


class LaneType(str, Enum):
    """车道线类型"""
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    DOUBLE_DASHED = "double_dashed"
    LEFT_DASHED_RIGHT_SOLID = "left_dashed_right_solid"
    BILATERAL = "bilateral"
    CURB = "curb"
    FENCE = "fence"
    NO_LANE = "no_lane"
    DIVERSION_BOUNDARY = "diversion_boundary"
    UNKNOWN = "unknown"


class LaneColor(str, Enum):
    """车道线颜色"""
    YELLOW = "yellow"
    WHITE = "white"
    UNKNOWN = "unknown"


class LaneLine(BaseModel):
    """车道线"""
    id: str = Field(..., description="车道线ID")
    type: str = Field(..., description="车道线类型")
    color: str = Field(..., description="车道线颜色")
    coordinates: List[List[float]] = Field(default_factory=list, description="3D坐标点列表")
    length: float = Field(default=0.0, description="长度(米)")

    def get_start_point(self) -> Tuple[float, float, float]:
        """获取起点坐标"""
        if self.coordinates:
            return tuple(self.coordinates[0])
        return (0.0, 0.0, 0.0)

    def get_end_point(self) -> Tuple[float, float, float]:
        """获取终点坐标"""
        if self.coordinates:
            return tuple(self.coordinates[-1])
        return (0.0, 0.0, 0.0)

    def get_center_point(self) -> Tuple[float, float, float]:
        """获取中心点坐标"""
        if self.coordinates:
            mid = len(self.coordinates) // 2
            return tuple(self.coordinates[mid])
        return (0.0, 0.0, 0.0)


class Centerline(BaseModel):
    """车道中心线（含拓扑关系）"""
    id: str = Field(..., description="中心线ID")
    coordinates: List[List[float]] = Field(default_factory=list, description="3D坐标点列表")
    left_boundary_id: Optional[str] = Field(None, description="左边界车道线ID")
    right_boundary_id: Optional[str] = Field(None, description="右边界车道线ID")
    predecessor_ids: List[str] = Field(default_factory=list, description="前驱中心线ID列表")
    successor_ids: List[str] = Field(default_factory=list, description="后继中心线ID列表")

    def has_predecessors(self) -> bool:
        """是否有前驱"""
        return len(self.predecessor_ids) > 0

    def has_successors(self) -> bool:
        """是否有后继"""
        return len(self.successor_ids) > 0

    def is_junction_entry(self) -> bool:
        """是否是路口入口（多个前驱汇聚）"""
        return len(self.predecessor_ids) > 1

    def is_junction_exit(self) -> bool:
        """是否是路口出口（多个后继分流）"""
        return len(self.successor_ids) > 1


class RoadMark(BaseModel):
    """道路标记"""
    id: str = Field(..., description="标记ID")
    type: str = Field(..., description="标记类型")
    coordinates: List[List[float]] = Field(default_factory=list, description="坐标点列表")
    semantic: Dict[str, Any] = Field(default_factory=dict, description="语义信息")


class TrafficSign(BaseModel):
    """交通标志"""
    id: str = Field(..., description="标志ID")
    category: str = Field(..., description="标志类别")
    function: Optional[Dict[str, Any]] = Field(default=None, description="功能描述")
    camera: str = Field(default="", description="拍摄相机")
    bbox: List[float] = Field(default_factory=list, description="2D边界框")
    position_3d: Optional[List[float]] = Field(None, description="3D位置")

    def is_direction_sign(self) -> bool:
        """是否是方向指示标志"""
        return self.category == "lane_direction"

    def get_direction(self) -> Optional[str]:
        """获取指示方向"""
        if self.function and "lane_direction_sign" in self.function:
            return self.function["lane_direction_sign"].get("direction_arrow")
        return None


class Intersection(BaseModel):
    """路口"""
    id: str = Field(..., description="路口ID")
    center: List[float] = Field(..., description="中心坐标")
    lanes: List[str] = Field(default_factory=list, description="关联车道ID列表")

    def get_center_tuple(self) -> Tuple[float, float, float]:
        """获取中心坐标元组"""
        return tuple(self.center)


class VectorMap(BaseModel):
    """完整矢量地图"""
    version: str = Field(default="1.0", description="地图版本")
    type: str = Field(default="static_map", description="地图类型")
    frames_processed: int = Field(default=0, description="处理的帧数")
    lane_lines: Dict[str, LaneLine] = Field(default_factory=dict, description="车道线字典")
    centerlines: Dict[str, Centerline] = Field(default_factory=dict, description="中心线字典")
    road_marks: Dict[str, RoadMark] = Field(default_factory=dict, description="道路标记字典")
    traffic_signs: Dict[str, TrafficSign] = Field(default_factory=dict, description="交通标志字典")
    traffic_lights: Dict[str, Any] = Field(default_factory=dict, description="交通灯字典")
    intersections: Dict[str, Intersection] = Field(default_factory=dict, description="路口字典")
    statistics: Dict[str, int] = Field(default_factory=dict, description="统计信息")

    # ========== 查询方法 ==========

    def get_lane_line(self, lane_id: str) -> Optional[LaneLine]:
        """获取指定车道线"""
        return self.lane_lines.get(lane_id)

    def get_centerline(self, centerline_id: str) -> Optional[Centerline]:
        """获取指定中心线"""
        return self.centerlines.get(centerline_id)

    def get_intersection(self, intersection_id: str) -> Optional[Intersection]:
        return self.intersections.get(intersection_id)

    def get_traffic_sign(self, sign_id: str) -> Optional[TrafficSign]:
        return self.traffic_signs.get(sign_id)

    def get_all_lane_ids(self) -> List[str]:
        """获取所有车道线ID"""
        return list(self.lane_lines.keys())

    def get_all_centerline_ids(self) -> List[str]:
        """获取所有中心线ID"""
        return list(self.centerlines.keys())

    def get_lane_count(self) -> int:
        """获取车道线总数"""
        return len(self.lane_lines)

    def get_centerline_count(self) -> int:
        """获取中心线总数"""
        return len(self.centerlines)

    def get_intersection_count(self) -> int:
        """获取路口总数"""
        return len(self.intersections)

    # ========== 拓扑查询 ==========

    def get_predecessor_centerlines(self, centerline_id: str) -> List[Centerline]:
        """获取前驱中心线对象列表"""
        cl = self.get_centerline(centerline_id)
        if not cl:
            return []
        return [self.get_centerline(pid) for pid in cl.predecessor_ids if self.get_centerline(pid)]

    def get_successor_centerlines(self, centerline_id: str) -> List[Centerline]:
        """获取后继中心线对象列表"""
        cl = self.get_centerline(centerline_id)
        if not cl:
            return []
        return [self.get_centerline(sid) for sid in cl.successor_ids if self.get_centerline(sid)]

    def get_left_boundary(self, centerline_id: str) -> Optional[LaneLine]:
        """获取左边界车道线"""
        cl = self.get_centerline(centerline_id)
        if cl and cl.left_boundary_id:
            return self.get_lane_line(cl.left_boundary_id)
        return None

    def get_right_boundary(self, centerline_id: str) -> Optional[LaneLine]:
        """获取右边界车道线"""
        cl = self.get_centerline(centerline_id)
        if cl and cl.right_boundary_id:
            return self.get_lane_line(cl.right_boundary_id)
        return None

    # ========== 统计查询 ==========

    def get_lane_type_statistics(self) -> Dict[str, int]:
        """获取车道线类型统计"""
        stats = {}
        for lane in self.lane_lines.values():
            stats[lane.type] = stats.get(lane.type, 0) + 1
        return stats

    def get_lane_color_statistics(self) -> Dict[str, int]:
        """获取车道线颜色统计"""
        stats = {}
        for lane in self.lane_lines.values():
            stats[lane.color] = stats.get(lane.color, 0) + 1
        return stats

    def get_sign_category_statistics(self) -> Dict[str, int]:
        """获取交通标志类别统计"""
        stats = {}
        for sign in self.traffic_signs.values():
            stats[sign.category] = stats.get(sign.category, 0) + 1
        return stats


class MapLoader:
    """地图加载器"""

    @staticmethod
    def load_from_json(filepath: str) -> VectorMap:
        """从JSON文件加载地图"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return MapLoader.parse_dict(data)

    @staticmethod
    def parse_dict(data: Dict) -> VectorMap:
        """解析字典数据为VectorMap"""
        # 解析车道线
        lane_lines = {}
        for k, v in data.get("lane_lines", {}).items():
            lane_lines[k] = LaneLine(**v)

        # 解析中心线
        centerlines = {}
        for k, v in data.get("centerlines", {}).items():
            centerlines[k] = Centerline(**v)

        # 解析道路标记
        road_marks = {}
        for k, v in data.get("road_marks", {}).items():
            road_marks[k] = RoadMark(**v)

        # 解析交通标志
        traffic_signs = {}
        for k, v in data.get("traffic_signs", {}).items():
            traffic_signs[k] = TrafficSign(**v)

        # 解析路口
        intersections = {}
        for k, v in data.get("intersections", {}).items():
            intersections[k] = Intersection(**v)

        return VectorMap(
            version=data.get("version", "1.0"),
            type=data.get("type", "static_map"),
            frames_processed=data.get("frames_processed", 0),
            lane_lines=lane_lines,
            centerlines=centerlines,
            road_marks=road_marks,
            traffic_signs=traffic_signs,
            traffic_lights=data.get("traffic_lights", {}),
            intersections=intersections,
            statistics=data.get("statistics", {}),
        )

    @staticmethod
    def load_default() -> VectorMap:
        """加载默认地图文件"""
        default_path = Path(__file__).parent.parent.parent / "data" / "vector_map.json"
        if default_path.exists():
            return MapLoader.load_from_json(str(default_path))
        raise FileNotFoundError(f"Default map file not found: {default_path}")