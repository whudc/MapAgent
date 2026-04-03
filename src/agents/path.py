"""
路径规划 Agent

负责路径规划，包括：
- 多路径搜索
- 车流状态分析
- 行驶建议生成
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agents.base import BaseAgent, AgentContext
from models.agent_io import PathQuery, PathResult, PathInfo
from utils.geo import calculate_distance, get_polyline_heading


@dataclass
class PathNode:
    """路径节点"""
    centerline_id: str
    coordinates: List[Tuple[float, float, float]]
    g_cost: float = 0  # 从起点到当前节点的成本
    h_cost: float = 0  # 启发式估计到终点的成本
    parent: Optional['PathNode'] = None

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost


class PathAgent(BaseAgent):
    """
    路径规划 Agent

    功能:
    - 多路径搜索
    - 车流状态分析
    - 行驶建议生成
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "path_agent"

    def get_tools(self) -> List[Dict]:
        """返回路径规划相关的工具"""
        return [
            {
                "name": "find_path",
                "description": "查找从起点到终点的路径",
                "parameters": {
                    "start_x": {"type": "number", "description": "起点X坐标"},
                    "start_y": {"type": "number", "description": "起点Y坐标"},
                    "end_x": {"type": "number", "description": "终点X坐标"},
                    "end_y": {"type": "number", "description": "终点Y坐标"},
                    "max_paths": {"type": "number", "description": "最大返回路径数", "default": 3}
                },
                "handler": self._find_path
            },
            {
                "name": "get_route_advice",
                "description": "获取路径行驶建议",
                "parameters": {
                    "path": {"type": "array", "description": "路径中心线ID列表"}
                },
                "handler": self._get_route_advice
            },
            {
                "name": "estimate_travel_time",
                "description": "估算行程时间",
                "parameters": {
                    "path": {"type": "array", "description": "路径中心线ID列表"},
                    "speed": {"type": "number", "description": "平均速度(m/s)", "default": 10}
                },
                "handler": self._estimate_travel_time
            },
            {
                "name": "find_nearby_destination",
                "description": "查找附近的目的地（如路口）",
                "parameters": {
                    "x": {"type": "number", "description": "中心X坐标"},
                    "y": {"type": "number", "description": "中心Y坐标"},
                    "radius": {"type": "number", "description": "搜索半径(米)", "default": 100}
                },
                "handler": self._find_nearby_destination
            },
        ]

    def get_system_prompt(self) -> str:
        return """你是一个路径规划专家，专门规划最优行驶路线。

你的职责包括：
1. 规划从起点到终点的路径
2. 分析路径上的车流状况
3. 提供行驶建议和注意事项

回答时请：
- 提供多条可选路径
- 说明每条路径的特点
- 给出最佳推荐和理由"""

    def process(self, query: str, **kwargs) -> Dict:
        """
        处理路径规划查询

        Args:
            query: 用户查询
            **kwargs: 额外参数 (origin, destination 等)

        Returns:
            PathResult 字典
        """
        path_query = self._parse_query(query, **kwargs)
        result = self._plan_path(path_query)
        return result

    def _parse_query(self, query: str, **kwargs) -> PathQuery:
        """解析查询参数"""
        return PathQuery(
            question=query,
            origin=kwargs.get("origin"),
            destination=kwargs.get("destination"),
            destination_name=kwargs.get("destination_name"),
            preferences=kwargs.get("preferences")
        )

    def _plan_path(self, query: PathQuery) -> Dict:
        """执行路径规划"""
        result = PathResult(advice="")

        if not query.origin or not query.destination:
            result.advice = "请提供起点和终点坐标"
            return result.to_dict()

        # 找到起点和终点附近的车道
        start_pos = query.origin if len(query.origin) == 3 else (*query.origin, 0)
        end_pos = query.destination if len(query.destination) == 3 else (*query.destination, 0)

        start_match = self.map_api.find_nearest_centerline(start_pos, max_distance=50)
        end_match = self.map_api.find_nearest_centerline(end_pos, max_distance=50)

        if not start_match:
            result.advice = "起点附近没有找到可用道路"
            return result.to_dict()

        if not end_match:
            result.advice = "终点附近没有找到可用道路"
            return result.to_dict()

        start_cl_id = start_match['centerline_id']
        end_cl_id = end_match['centerline_id']

        # 使用 A* 搜索路径
        paths = self._search_paths(start_cl_id, end_cl_id, max_paths=3)

        if not paths:
            # 尝试反向搜索
            paths = self._search_paths(end_cl_id, start_cl_id, max_paths=3)
            if paths:
                paths = [list(reversed(p)) for p in paths]

        if not paths:
            result.advice = "未能找到连接起点和终点的路径，请检查坐标是否正确"
            return result.to_dict()

        # 构建路径信息
        path_infos = []
        for i, path_cl_ids in enumerate(paths):
            path_info = self._build_path_info(path_cl_ids, start_pos, end_pos)
            path_info.id = f"path_{i}"
            path_infos.append(path_info)

        result.paths = [p.to_dict() for p in path_infos]
        result.best_path = path_infos[0].to_dict() if path_infos else None
        result.distance = path_infos[0].distance if path_infos else 0
        result.estimated_time = path_infos[0].estimated_time if path_infos else 0

        # 生成建议
        result.advice = self._generate_advice(path_infos, query)

        return result.to_dict()

    def _search_paths(self, start_id: str, end_id: str, max_paths: int = 3) -> List[List[str]]:
        """使用 BFS 搜索多条路径"""
        all_paths = []
        queue = [(start_id, [start_id])]
        visited_count = defaultdict(int)
        max_visits = 3  # 每个节点最多访问次数

        while queue and len(all_paths) < max_paths:
            current_id, path = queue.pop(0)

            # 限制路径长度
            if len(path) > 20:
                continue

            # 限制节点访问次数
            if visited_count[current_id] >= max_visits:
                continue
            visited_count[current_id] += 1

            if current_id == end_id:
                all_paths.append(path)
                continue

            cl_info = self.map_api.get_centerline_info(current_id)
            if not cl_info:
                continue

            for succ_id in cl_info.get('successor_ids', []):
                if succ_id not in path:  # 避免环路
                    queue.append((succ_id, path + [succ_id]))

        return all_paths

    def _build_path_info(self, centerline_ids: List[str],
                          start_pos: Tuple, end_pos: Tuple) -> PathInfo:
        """构建路径信息"""
        all_coords = []
        lane_ids = []
        total_length = 0.0

        for cl_id in centerline_ids:
            cl_info = self.map_api.get_centerline_info(cl_id)
            if cl_info and cl_info.get('coordinates'):
                coords = [tuple(c) for c in cl_info['coordinates']]
                all_coords.extend(coords)

                # 计算长度
                for i in range(len(coords) - 1):
                    total_length += calculate_distance(coords[i], coords[i + 1])

                # 记录车道
            if cl_info:
                if cl_info.get('right_boundary_id'):
                    lane_ids.append(cl_info['right_boundary_id'])
                elif cl_info.get('left_boundary_id'):
                    lane_ids.append(cl_info['left_boundary_id'])

        # 估算时间（假设平均速度 10 m/s = 36 km/h）
        estimated_time = total_length / 10.0  # 秒

        return PathInfo(
            id="",
            waypoints=all_coords,
            distance=total_length,
            estimated_time=estimated_time,
            lane_ids=lane_ids
        )

    def _generate_advice(self, path_infos: List[PathInfo], query: PathQuery) -> str:
        """生成行驶建议"""
        if not path_infos:
            return "无法生成路径建议"

        best_path = path_infos[0]
        advice_parts = []

        # 基本信息建议
        distance_km = best_path.distance / 1000
        time_min = best_path.estimated_time / 60
        advice_parts.append(f"推荐路径：总距离 {distance_km:.1f} 公里，预计用时 {time_min:.0f} 分钟")

        # 分析路径特点
        lane_types = set()
        for lane_id in best_path.lane_ids:
            lane_info = self.map_api.get_lane_info(lane_id)
            if lane_info:
                lane_types.add(lane_info['type'])

        if 'double_solid' in lane_types:
            advice_parts.append("注意：路径中存在双实线路段，禁止变道")
        if 'dashed' in lane_types:
            advice_parts.append("部分路段为虚线，可按需变道")

        # 检查路口
        intersection_count = 0
        for wp in best_path.waypoints[:10]:  # 检查前10个点
            ints = self.map_api.find_intersections_in_area(wp, radius=20)
            intersection_count += len(ints)

        if intersection_count > 0:
            advice_parts.append(f"路径经过约 {intersection_count} 个路口，请注意观察")

        return "；".join(advice_parts)

    # ========== 工具处理函数 ==========

    def _find_path(self, start_x: float, start_y: float,
                    end_x: float, end_y: float, max_paths: int = 3) -> Dict:
        """查找路径"""
        result = self._plan_path(PathQuery(
            question="",
            origin=(start_x, start_y, 0),
            destination=(end_x, end_y, 0)
        ))
        return result

    def _get_route_advice(self, path: List[str]) -> Dict:
        """获取路径建议"""
        if not path:
            return {"advice": "无效路径"}

        path_info = self._build_path_info(path, (0, 0, 0), (0, 0, 0))
        advice = self._generate_advice([path_info], PathQuery(question=""))

        return {
            "path_length": f"{path_info.distance:.0f}米",
            "estimated_time": f"{path_info.estimated_time/60:.0f}分钟",
            "advice": advice
        }

    def _estimate_travel_time(self, path: List[str], speed: float = 10) -> Dict:
        """估算行程时间"""
        if not path:
            return {"time": 0, "distance": 0}

        total_distance = 0
        for cl_id in path:
            cl_info = self.map_api.get_centerline_info(cl_id)
            if cl_info and cl_info.get('coordinates'):
                coords = [tuple(c) for c in cl_info['coordinates']]
                for i in range(len(coords) - 1):
                    total_distance += calculate_distance(coords[i], coords[i + 1])

        time_seconds = total_distance / speed
        time_minutes = time_seconds / 60

        return {
            "distance_meters": round(total_distance, 1),
            "time_seconds": round(time_seconds, 1),
            "time_minutes": round(time_minutes, 1),
            "assumed_speed_ms": speed
        }

    def _find_nearby_destination(self, x: float, y: float, radius: float = 100) -> Dict:
        """查找附近的目的地"""
        position = (x, y, 0)

        # 查找附近路口
        intersections = self.map_api.find_intersections_in_area(position, radius)

        # 查找附近车道
        lanes = self.map_api.find_lanes_in_area(position, radius)

        destinations = []
        for i, inter in enumerate(intersections[:5]):
            destinations.append({
                "type": "intersection",
                "id": inter['id'],
                "distance": inter['distance'],
                "description": f"路口 {inter['id']}"
            })

        return {
            "position": [x, y],
            "radius": radius,
            "destinations": destinations,
            "total_intersections": len(intersections),
            "total_lanes": len(lanes)
        }