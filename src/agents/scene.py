"""
场景理解 Agent

负责理解道路场景，包括：
- 车道数量和类型
- 路口结构
- 交通规则信息
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agents.base import BaseAgent, AgentContext
from models.agent_io import SceneQuery, SceneResult


class SceneAgent(BaseAgent):
    """
    场景理解 Agent

    功能:
    - 查询车道数量、类型
    - 分析路口结构
    - 返回交通规则信息
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "scene_agent"

    def get_tools(self) -> List[Dict]:
        """返回场景理解相关的工具"""
        return [
            {
                "name": "get_lane_count_by_type",
                "description": "获取指定区域内各类型车道的数量",
                "parameters": {
                    "x": {"type": "number", "description": "中心点X坐标"},
                    "y": {"type": "number", "description": "中心点Y坐标"},
                    "radius": {"type": "number", "description": "查询半径(米)", "default": 100}
                },
                "handler": self._get_lane_count_by_type
            },
            {
                "name": "get_intersection_structure",
                "description": "获取路口的详细结构信息",
                "parameters": {
                    "intersection_id": {"type": "string", "description": "路口ID"}
                },
                "handler": self._get_intersection_structure
            },
            {
                "name": "get_nearby_traffic_signs",
                "description": "获取附近的交通标志",
                "parameters": {
                    "x": {"type": "number", "description": "中心点X坐标"},
                    "y": {"type": "number", "description": "中心点Y坐标"},
                    "radius": {"type": "number", "description": "查询半径(米)", "default": 50}
                },
                "handler": self._get_nearby_traffic_signs
            },
            {
                "name": "analyze_road_scene",
                "description": "分析指定位置的道路场景",
                "parameters": {
                    "x": {"type": "number", "description": "位置X坐标"},
                    "y": {"type": "number", "description": "位置Y坐标"},
                    "z": {"type": "number", "description": "位置Z坐标", "default": 0},
                    "radius": {"type": "number", "description": "分析半径(米)", "default": 100}
                },
                "handler": self._analyze_road_scene
            },
        ]

    def get_system_prompt(self) -> str:
        return """你是一个场景理解专家，专门分析道路场景。

你的职责包括：
1. 分析车道数量、类型和分布
2. 分析路口结构和交通组织
3. 识别交通规则和限制

回答时请：
- 使用清晰、专业的语言
- 提供具体的数据和信息
- 给出实用的建议"""

    def process(self, query: str, **kwargs) -> Dict:
        """
        处理场景理解查询

        Args:
            query: 用户查询
            **kwargs: 额外参数 (location, intersection_id 等)

        Returns:
            SceneResult 字典
        """
        # 解析查询参数
        scene_query = self._parse_query(query, **kwargs)

        # 执行分析
        result = self._analyze_scene(scene_query)

        return result

    def _parse_query(self, query: str, **kwargs) -> SceneQuery:
        """解析查询参数"""
        location = kwargs.get("location")
        radius = kwargs.get("radius", 100.0)
        intersection_id = kwargs.get("intersection_id")
        lane_id = kwargs.get("lane_id")

        return SceneQuery(
            question=query,
            location=location,
            radius=radius,
            intersection_id=intersection_id,
            lane_id=lane_id
        )

    def _analyze_scene(self, query: SceneQuery) -> Dict:
        """执行场景分析"""
        result = SceneResult(summary="")

        # 如果有指定路口
        if query.intersection_id:
            int_info = self.map_api.get_intersection_info(query.intersection_id)
            if int_info:
                result.intersection_info = int_info
                result.summary = f"路口 {query.intersection_id} 包含 {int_info['lane_count']} 条关联车道。"

                # 获取车道类型统计
                lane_types = {}
                for lane_id in int_info['lanes']:
                    lane_info = self.map_api.get_lane_info(lane_id)
                    if lane_info:
                        lane_types[lane_info['type']] = lane_types.get(lane_info['type'], 0) + 1
                result.lane_types = lane_types
                result.lane_count = int_info['lane_count']

        # 如果有指定位置
        elif query.location:
            position = query.location
            if len(position) == 2:
                position = (position[0], position[1], 0)

            # 获取区域统计
            stats = self.map_api.get_area_statistics(position, query.radius)
            result.lane_count = stats['lane_count']
            result.lane_types = stats['lane_types']

            # 获取交通标志
            signs = self.map_api.get_traffic_signs_in_area(position, query.radius)
            result.nearby_signs = [s['id'] for s in signs]

            # 生成摘要
            result.summary = f"位置 ({position[0]:.1f}, {position[1]:.1f}) 周围 {query.radius}米范围内"
            result.summary += f"共有 {stats['lane_count']} 条车道"
            if stats['intersection_count'] > 0:
                result.summary += f"，{stats['intersection_count']} 个路口"

            # 交通规则提示
            rules = self._generate_traffic_rules(stats, signs)
            result.traffic_rules = rules

        # 如果有指定车道
        elif query.lane_id:
            lane_info = self.map_api.get_lane_info(query.lane_id)
            if lane_info:
                result.summary = f"车道 {query.lane_id}: 类型={lane_info['type']}, 颜色={lane_info['color']}"
                result.lane_count = 1
                result.lane_types = {lane_info['type']: 1}

        else:
            # 返回全局信息
            summary = self.map_api.get_map_summary()
            result.summary = f"地图共包含 {summary['total_lanes']} 条车道，{summary['total_intersections']} 个路口"
            result.lane_count = summary['total_lanes']
            result.lane_types = summary['lane_type_distribution']

        return result.to_dict()

    def _generate_traffic_rules(self, stats: Dict, signs: List[Dict]) -> List[str]:
        """生成交通规则提示"""
        rules = []

        # 根据车道类型
        lane_types = stats.get('lane_types', {})
        if lane_types.get('double_solid', 0) > 0:
            rules.append("存在双实线，禁止跨越")
        if lane_types.get('curb', 0) > 0:
            rules.append("注意路缘，不要越界")

        # 根据交通标志
        for sign in signs:
            if sign.get('category') == 'lane_direction':
                rules.append("注意车道方向指示标志")
            elif sign.get('category') == 'warning':
                rules.append("前方有警告标志，请注意")

        return rules

    # ========== 工具处理函数 ==========

    def _get_lane_count_by_type(self, x: float, y: float, radius: float = 100) -> Dict:
        """获取区域内的车道类型统计"""
        position = (x, y, 0)
        lanes = self.map_api.find_lanes_in_area(position, radius)

        lane_types = {}
        for lane in lanes:
            t = lane['type']
            lane_types[t] = lane_types.get(t, 0) + 1

        return {
            "position": [x, y],
            "radius": radius,
            "total_lanes": len(lanes),
            "lane_types": lane_types
        }

    def _get_intersection_structure(self, intersection_id: str) -> Dict:
        """获取路口结构"""
        info = self.map_api.get_intersection_info(intersection_id)
        if not info:
            return {"error": f"路口 {intersection_id} 不存在"}

        # 获取每条车道的详细信息
        lane_details = []
        for lane_id in info['lanes']:
            lane_info = self.map_api.get_lane_info(lane_id)
            if lane_info:
                lane_details.append({
                    "id": lane_id,
                    "type": lane_info['type'],
                    "color": lane_info['color']
                })

        return {
            "intersection_id": intersection_id,
            "center": info['center'],
            "lane_count": info['lane_count'],
            "lanes": lane_details
        }

    def _get_nearby_traffic_signs(self, x: float, y: float, radius: float = 50) -> Dict:
        """获取附近的交通标志"""
        position = (x, y, 0)
        signs = self.map_api.get_traffic_signs_in_area(position, radius)

        return {
            "position": [x, y],
            "radius": radius,
            "sign_count": len(signs),
            "signs": signs
        }

    def _analyze_road_scene(self, x: float, y: float, z: float = 0, radius: float = 100) -> Dict:
        """分析道路场景"""
        position = (x, y, z)

        # 获取统计信息
        stats = self.map_api.get_area_statistics(position, radius)

        # 获取最近车道
        nearest_lane = self.map_api.find_nearest_lane(position, radius)

        # 获取最近路口
        intersections = self.map_api.find_intersections_in_area(position, radius)

        # 获取交通标志
        signs = self.map_api.get_traffic_signs_in_area(position, radius)

        return {
            "position": [x, y, z],
            "radius": radius,
            "statistics": {
                "lane_count": stats['lane_count'],
                "intersection_count": stats['intersection_count'],
                "lane_types": stats['lane_types']
            },
            "nearest_lane": nearest_lane,
            "nearby_intersections": intersections,
            "traffic_signs": signs
        }