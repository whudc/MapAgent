"""
地图数据访问 API

提供对矢量地图的查询接口
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import sys
from pathlib import Path

# 确保能找到模块
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.map_data import VectorMap, LaneLine, Centerline, Intersection, MapLoader
from utils.geo import (
    calculate_distance,
    point_to_polyline_distance,
    line_segments_intersect,
    polyline_bounding_box,
    point_in_bbox,
    project_point_on_polyline,
    get_polyline_heading,
)


@dataclass
class LaneInfo:
    """车道信息"""
    id: str
    type: str
    color: str
    length: float
    coordinates: List[Tuple[float, float, float]]
    left_boundary_id: Optional[str] = None
    right_boundary_id: Optional[str] = None
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "color": self.color,
            "length": self.length,
            "coordinates": [[c for c in p] for p in self.coordinates],
            "left_boundary_id": self.left_boundary_id,
            "right_boundary_id": self.right_boundary_id,
            "predecessor_ids": self.predecessor_ids,
            "successor_ids": self.successor_ids,
        }


@dataclass
class IntersectionInfo:
    """路口信息"""
    id: str
    center: Tuple[float, float, float]
    lanes: List[str]
    lane_count: int

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "center": list(self.center),
            "lanes": self.lanes,
            "lane_count": self.lane_count,
        }


@dataclass
class AreaStatistics:
    """区域统计"""
    center: Tuple[float, float, float]
    radius: float
    lane_count: int
    centerline_count: int
    intersection_count: int
    lane_types: Dict[str, int]
    lane_colors: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "lane_count": self.lane_count,
            "centerline_count": self.centerline_count,
            "intersection_count": self.intersection_count,
            "lane_types": self.lane_types,
            "lane_colors": self.lane_colors,
        }


@dataclass
class ConnectedLaneInfo:
    """连接车道信息"""
    lane_id: str
    centerline_id: str
    predecessors: List[Dict]
    successors: List[Dict]
    left_neighbor: Optional[str]
    right_neighbor: Optional[str]

    def to_dict(self) -> Dict:
        return {
            "lane_id": self.lane_id,
            "centerline_id": self.centerline_id,
            "predecessors": self.predecessors,
            "successors": self.successors,
            "left_neighbor": self.left_neighbor,
            "right_neighbor": self.right_neighbor,
        }


@dataclass
class LaneMatchResult:
    """车道匹配结果"""
    lane_id: str
    centerline_id: str
    distance: float
    projected_point: Tuple[float, float, float]
    heading: float
    is_forward: bool

    def to_dict(self) -> Dict:
        return {
            "lane_id": self.lane_id,
            "centerline_id": self.centerline_id,
            "distance": round(self.distance, 2),
            "projected_point": list(self.projected_point),
            "heading": round(self.heading, 1),
            "is_forward": self.is_forward,
        }


class MapAPI:
    """
    地图数据访问 API

    提供对矢量地图的各种查询功能
    """

    def __init__(self, map_data: Optional[VectorMap] = None,
                 map_file: Optional[str] = None):
        """
        初始化

        Args:
            map_data: 矢量地图数据
            map_file: 地图文件路径
        """
        if map_data:
            self.map = map_data
        elif map_file:
            self.map = MapLoader.load_from_json(map_file)
        else:
            # 尝试加载默认地图
            self.map = MapLoader.load_default()

        # 构建空间索引（简化版）
        self._build_spatial_index()

    def _build_spatial_index(self):
        """构建简化的空间索引"""
        # 将地图划分为网格，加速空间查询
        self._grid_size = 50.0  # 50米网格
        self._lane_grid: Dict[Tuple[int, int, int], List[str]] = {}
        self._centerline_grid: Dict[Tuple[int, int, int], List[str]] = {}

        # 索引车道线
        for lane_id, lane in self.map.lane_lines.items():
            for coord in lane.coordinates:
                grid_key = self._get_grid_key(tuple(coord))
                if grid_key not in self._lane_grid:
                    self._lane_grid[grid_key] = []
                if lane_id not in self._lane_grid[grid_key]:
                    self._lane_grid[grid_key].append(lane_id)

        # 索引中心线
        for cl_id, cl in self.map.centerlines.items():
            for coord in cl.coordinates:
                grid_key = self._get_grid_key(tuple(coord))
                if grid_key not in self._centerline_grid:
                    self._centerline_grid[grid_key] = []
                if cl_id not in self._centerline_grid[grid_key]:
                    self._centerline_grid[grid_key].append(cl_id)

    def _get_grid_key(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """获取网格键"""
        return (
            int(point[0] // self._grid_size),
            int(point[1] // self._grid_size),
            int(point[2] // self._grid_size) if len(point) > 2 else 0
        )

    def _get_nearby_grids(self, point: Tuple[float, float, float],
                          radius: float) -> List[Tuple[int, int, int]]:
        """获取附近的网格"""
        center_grid = self._get_grid_key(point)
        grid_radius = int(radius // self._grid_size) + 1

        grids = []
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-1, 2):  # z方向只检查附近
                    grids.append((
                        center_grid[0] + dx,
                        center_grid[1] + dy,
                        center_grid[2] + dz
                    ))
        return grids

    # ========== 基础查询 ==========

    def get_lane_info(self, lane_id: str) -> Optional[Dict]:
        """
        获取车道信息

        Args:
            lane_id: 车道ID

        Returns:
            车道信息字典
        """
        lane = self.map.get_lane_line(lane_id)
        if not lane:
            return None

        # 尝试获取拓扑信息
        centerline = self._find_centerline_by_boundary(lane_id)

        return LaneInfo(
            id=lane.id,
            type=lane.type,
            color=lane.color,
            length=lane.length,
            coordinates=[tuple(c) for c in lane.coordinates],
            left_boundary_id=centerline.left_boundary_id if centerline else None,
            right_boundary_id=centerline.right_boundary_id if centerline else None,
            predecessor_ids=centerline.predecessor_ids if centerline else [],
            successor_ids=centerline.successor_ids if centerline else [],
        ).to_dict()

    def get_intersection_info(self, intersection_id: str) -> Optional[Dict]:
        """
        获取路口信息

        Args:
            intersection_id: 路口ID

        Returns:
            路口信息字典
        """
        intersection = self.map.get_intersection(intersection_id)
        if not intersection:
            return None

        return IntersectionInfo(
            id=intersection.id,
            center=intersection.get_center_tuple(),
            lanes=intersection.lanes,
            lane_count=len(intersection.lanes),
        ).to_dict()

    def get_lane_topology(self, lane_id: str) -> Dict:
        """
        获取车道拓扑关系

        Args:
            lane_id: 车道ID

        Returns:
            拓扑信息
        """
        centerline = self._find_centerline_by_boundary(lane_id)
        if not centerline:
            return {"error": f"Lane {lane_id} not found in any centerline"}

        return {
            "lane_id": lane_id,
            "centerline_id": centerline.id,
            "predecessor_ids": centerline.predecessor_ids,
            "successor_ids": centerline.successor_ids,
            "left_boundary_id": centerline.left_boundary_id,
            "right_boundary_id": centerline.right_boundary_id,
        }

    def get_connected_lanes(self, lane_id: str) -> Optional[Dict]:
        """
        获取连接车道信息

        Args:
            lane_id: 车道ID

        Returns:
            连接车道信息
        """
        centerline = self._find_centerline_by_boundary(lane_id)
        if not centerline:
            # 尝试作为中心线ID查找
            centerline = self.map.get_centerline(lane_id)

        if not centerline:
            return None

        # 收集前驱信息
        predecessors = []
        for pred_id in centerline.predecessor_ids:
            pred_cl = self.map.get_centerline(pred_id)
            if pred_cl:
                predecessors.append({
                    "centerline_id": pred_id,
                    "left_boundary": pred_cl.left_boundary_id,
                    "right_boundary": pred_cl.right_boundary_id,
                })

        # 收集后继信息
        successors = []
        for succ_id in centerline.successor_ids:
            succ_cl = self.map.get_centerline(succ_id)
            if succ_cl:
                successors.append({
                    "centerline_id": succ_id,
                    "left_boundary": succ_cl.left_boundary_id,
                    "right_boundary": succ_cl.right_boundary_id,
                })

        return ConnectedLaneInfo(
            lane_id=lane_id,
            centerline_id=centerline.id,
            predecessors=predecessors,
            successors=successors,
            left_neighbor=centerline.left_boundary_id,
            right_neighbor=centerline.right_boundary_id,
        ).to_dict()

    def get_centerline_info(self, centerline_id: str) -> Optional[Dict]:
        """
        获取中心线信息

        Args:
            centerline_id: 中心线ID

        Returns:
            中心线信息
        """
        cl = self.map.get_centerline(centerline_id)
        if not cl:
            return None

        return {
            "id": cl.id,
            "coordinates": [list(c) for c in cl.coordinates],
            "left_boundary_id": cl.left_boundary_id,
            "right_boundary_id": cl.right_boundary_id,
            "predecessor_ids": cl.predecessor_ids,
            "successor_ids": cl.successor_ids,
            "has_junction_entry": cl.is_junction_entry(),
            "has_junction_exit": cl.is_junction_exit(),
        }

    # ========== 空间查询 ==========

    def find_nearest_lane(self, position: Tuple[float, float, float],
                          max_distance: float = 50.0) -> Optional[Dict]:
        """
        查找最近的车道

        Args:
            position: 查询位置
            max_distance: 最大搜索距离

        Returns:
            最近车道信息
        """
        min_dist = float('inf')
        nearest_lane = None
        nearest_idx = 0

        # 使用空间索引加速
        candidate_lanes = set()
        for grid_key in self._get_nearby_grids(position, max_distance):
            candidate_lanes.update(self._lane_grid.get(grid_key, []))

        # 如果索引没有结果，遍历所有
        if not candidate_lanes:
            candidate_lanes = set(self.map.lane_lines.keys())

        for lane_id in candidate_lanes:
            lane = self.map.lane_lines.get(lane_id)
            if not lane or not lane.coordinates:
                continue

            coords = [tuple(c) for c in lane.coordinates]
            dist, idx = point_to_polyline_distance(position, coords)

            if dist < min_dist:
                min_dist = dist
                nearest_lane = lane
                nearest_idx = idx

        if not nearest_lane or min_dist > max_distance:
            return None

        return {
            "lane_id": nearest_lane.id,
            "type": nearest_lane.type,
            "color": nearest_lane.color,
            "distance": round(min_dist, 2),
            "nearest_point_index": nearest_idx,
        }

    def find_nearest_centerline(self, position: Tuple[float, float, float],
                                 max_distance: float = 50.0) -> Optional[Dict]:
        """
        查找最近的中心线

        Args:
            position: 查询位置
            max_distance: 最大搜索距离

        Returns:
            最近中心线信息
        """
        min_dist = float('inf')
        nearest_cl = None
        nearest_idx = 0

        # 使用空间索引
        candidate_cls = set()
        for grid_key in self._get_nearby_grids(position, max_distance):
            candidate_cls.update(self._centerline_grid.get(grid_key, []))

        if not candidate_cls:
            candidate_cls = set(self.map.centerlines.keys())

        for cl_id in candidate_cls:
            cl = self.map.centerlines.get(cl_id)
            if not cl or not cl.coordinates:
                continue

            coords = [tuple(c) for c in cl.coordinates]
            dist, idx = point_to_polyline_distance(position, coords)

            if dist < min_dist:
                min_dist = dist
                nearest_cl = cl
                nearest_idx = idx

        if not nearest_cl or min_dist > max_distance:
            return None

        return {
            "centerline_id": nearest_cl.id,
            "distance": round(min_dist, 2),
            "nearest_point_index": nearest_idx,
            "left_boundary_id": nearest_cl.left_boundary_id,
            "right_boundary_id": nearest_cl.right_boundary_id,
        }

    def match_vehicle_to_lane(self, position: Tuple[float, float, float],
                               heading: Optional[float] = None,
                               max_distance: float = 10.0) -> Optional[Dict]:
        """
        将车辆匹配到车道

        Args:
            position: 车辆位置
            heading: 车辆航向角（度），可选
            max_distance: 最大匹配距离

        Returns:
            匹配结果
        """
        # 找最近的中心线
        nearest_cl_info = self.find_nearest_centerline(position, max_distance)
        if not nearest_cl_info:
            return None

        cl = self.map.get_centerline(nearest_cl_info["centerline_id"])
        if not cl or not cl.coordinates:
            return None

        coords = [tuple(c) for c in cl.coordinates]

        # 计算投影点和航向
        proj_point, seg_idx, seg_progress = project_point_on_polyline(position, coords)

        # 计算车道航向
        lane_heading = get_polyline_heading(coords, seg_idx)

        # 判断行驶方向
        is_forward = True
        if heading is not None:
            heading_diff = abs(heading - lane_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            is_forward = heading_diff < 90

        # 查找边界车道
        left_lane = None
        right_lane = None
        if cl.left_boundary_id:
            left_lane = cl.left_boundary_id
        if cl.right_boundary_id:
            right_lane = cl.right_boundary_id

        return LaneMatchResult(
            lane_id=cl.right_boundary_id or cl.left_boundary_id or cl.id,
            centerline_id=cl.id,
            distance=nearest_cl_info["distance"],
            projected_point=proj_point,
            heading=lane_heading,
            is_forward=is_forward,
        ).to_dict()

    def find_lanes_in_area(self, center: Tuple[float, float, float],
                           radius: float) -> List[Dict]:
        """
        查找区域内的车道

        Args:
            center: 区域中心
            radius: 半径

        Returns:
            车道列表
        """
        lanes = []

        # 使用空间索引
        candidate_lanes = set()
        for grid_key in self._get_nearby_grids(center, radius):
            candidate_lanes.update(self._lane_grid.get(grid_key, []))

        if not candidate_lanes:
            candidate_lanes = set(self.map.lane_lines.keys())

        for lane_id in candidate_lanes:
            lane = self.map.lane_lines.get(lane_id)
            if not lane:
                continue

            # 检查车道是否有在区域内的点
            for coord in lane.coordinates:
                dist = calculate_distance(center, tuple(coord))
                if dist <= radius:
                    lanes.append({
                        "lane_id": lane.id,
                        "type": lane.type,
                        "color": lane.color,
                        "length": lane.length,
                    })
                    break

        return lanes

    def find_intersections_in_area(self, center: Tuple[float, float, float],
                                    radius: float) -> List[Dict]:
        """
        查找区域内的路口

        Args:
            center: 区域中心
            radius: 半径

        Returns:
            路口列表
        """
        intersections = []
        for intersection in self.map.intersections.values():
            dist = calculate_distance(center, intersection.get_center_tuple())
            if dist <= radius:
                intersections.append({
                    "id": intersection.id,
                    "center": list(intersection.get_center_tuple()),
                    "lane_count": len(intersection.lanes),
                    "distance": round(dist, 2),
                })
        return intersections

    def get_area_statistics(self, center: Tuple[float, float, float],
                            radius: float) -> Dict:
        """
        获取区域统计信息

        Args:
            center: 区域中心
            radius: 半径

        Returns:
            统计信息
        """
        lanes = self.find_lanes_in_area(center, radius)

        # 统计类型
        lane_types: Dict[str, int] = {}
        lane_colors: Dict[str, int] = {}
        total_length = 0.0
        for lane in lanes:
            lane_types[lane["type"]] = lane_types.get(lane["type"], 0) + 1
            lane_colors[lane["color"]] = lane_colors.get(lane["color"], 0) + 1
            total_length += lane.get("length", 0)

        # 统计路口
        intersections = self.find_intersections_in_area(center, radius)

        # 统计中心线
        centerline_count = 0
        for cl in self.map.centerlines.values():
            for coord in cl.coordinates:
                dist = calculate_distance(center, tuple(coord))
                if dist <= radius:
                    centerline_count += 1
                    break

        # 统计交通标志
        signs = self.get_traffic_signs_in_area(center, radius)

        return {
            "center": list(center),
            "radius": radius,
            "lane_count": len(lanes),
            "centerline_count": centerline_count,
            "intersection_count": len(intersections),
            "traffic_sign_count": len(signs),
            "lane_types": lane_types,
            "lane_colors": lane_colors,
            "total_lane_length": round(total_length, 1),
        }

    # ========== 交通标志查询 ==========

    def get_traffic_signs_in_area(self, center: Tuple[float, float, float],
                                   radius: float) -> List[Dict]:
        """
        获取区域内的交通标志

        Args:
            center: 区域中心
            radius: 半径

        Returns:
            交通标志列表
        """
        signs = []
        for sign_id, sign in self.map.traffic_signs.items():
            if sign.position_3d:
                dist = calculate_distance(center, tuple(sign.position_3d))
                if dist <= radius:
                    signs.append({
                        "id": sign.id,
                        "category": sign.category,
                        "function": sign.function,
                        "distance": round(dist, 2),
                    })
        return signs

    def get_direction_signs_in_area(self, center: Tuple[float, float, float],
                                     radius: float) -> List[Dict]:
        """
        获取区域内的方向指示标志

        Args:
            center: 区域中心
            radius: 半径

        Returns:
            方向指示标志列表
        """
        all_signs = self.get_traffic_signs_in_area(center, radius)
        direction_signs = []
        for sign in all_signs:
            if sign.get("category") == "lane_direction":
                direction_signs.append(sign)
        return direction_signs

    # ========== 路径查询 ==========

    def find_path_between_lanes(self, start_lane_id: str, end_lane_id: str,
                                 max_depth: int = 10) -> Optional[List[str]]:
        """
        查找两个车道之间的路径

        Args:
            start_lane_id: 起始车道ID
            end_lane_id: 目标车道ID
            max_depth: 最大搜索深度

        Returns:
            车道ID列表（路径）
        """
        # 找到对应的中心线
        start_cl = self._find_centerline_by_boundary(start_lane_id)
        end_cl = self._find_centerline_by_boundary(end_lane_id)

        if not start_cl or not end_cl:
            return None

        start_id = start_cl.id
        end_id = end_cl.id

        if start_id == end_id:
            return [start_lane_id]

        # BFS 搜索
        visited: Set[str] = set()
        queue: List[Tuple[str, List[str]]] = [(start_id, [start_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id in visited:
                continue

            visited.add(current_id)

            current_cl = self.map.get_centerline(current_id)
            if not current_cl:
                continue

            for succ_id in current_cl.successor_ids:
                if succ_id == end_id:
                    return path + [succ_id]

                if succ_id not in visited:
                    queue.append((succ_id, path + [succ_id]))

        return None

    # ========== 辅助方法 ==========

    def _find_centerline_by_boundary(self, lane_id: str) -> Optional[Centerline]:
        """根据边界ID查找中心线"""
        for cl in self.map.centerlines.values():
            if cl.left_boundary_id == lane_id or cl.right_boundary_id == lane_id:
                return cl
        return None

    def get_map_summary(self) -> Dict:
        """获取地图概要"""
        return {
            "version": self.map.version,
            "type": self.map.type,
            "frames_processed": self.map.frames_processed,
            "total_lanes": self.map.get_lane_count(),
            "total_centerlines": self.map.get_centerline_count(),
            "total_intersections": self.map.get_intersection_count(),
            "total_traffic_signs": len(self.map.traffic_signs),
            "lane_type_distribution": self.map.get_lane_type_statistics(),
            "lane_color_distribution": self.map.get_lane_color_statistics(),
        }

    def get_all_lane_ids(self) -> List[str]:
        """获取所有车道ID"""
        return list(self.map.lane_lines.keys())

    def get_all_centerline_ids(self) -> List[str]:
        """获取所有中心线ID"""
        return list(self.map.centerlines.keys())

    def get_all_intersection_ids(self) -> List[str]:
        """获取所有路口ID"""
        return list(self.map.intersections.keys())