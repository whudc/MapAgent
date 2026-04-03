"""
行为分析 Agent

负责分析车辆行为，包括：
- 车辆位置匹配车道
- 行为预测（转向、变道）
- 责任分析
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agents.base import BaseAgent, AgentContext
from models.agent_io import BehaviorQuery, BehaviorResult


class VehicleAction(str, Enum):
    """车辆行为"""
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    U_TURN = "u_turn"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    STOP = "stop"
    SLOW_DOWN = "slow_down"
    ACCELERATE = "accelerate"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BehaviorAgent(BaseAgent):
    """
    行为分析 Agent

    功能:
    - 车辆位置匹配车道
    - 行为预测（转向、变道）
    - 责任分析
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "behavior_agent"

    def get_tools(self) -> List[Dict]:
        """返回行为分析相关的工具"""
        return [
            {
                "name": "match_vehicle_to_lane",
                "description": "将车辆匹配到最近的车道",
                "parameters": {
                    "x": {"type": "number", "description": "车辆X坐标"},
                    "y": {"type": "number", "description": "车辆Y坐标"},
                    "heading": {"type": "number", "description": "车辆航向角(度)", "default": 0}
                },
                "handler": self._match_vehicle_to_lane
            },
            {
                "name": "predict_vehicle_action",
                "description": "预测车辆下一步行为",
                "parameters": {
                    "x": {"type": "number", "description": "车辆X坐标"},
                    "y": {"type": "number", "description": "车辆Y坐标"},
                    "heading": {"type": "number", "description": "车辆航向角(度)"},
                    "speed": {"type": "number", "description": "车辆速度(m/s)", "default": 0}
                },
                "handler": self._predict_vehicle_action
            },
            {
                "name": "analyze_collision_risk",
                "description": "分析碰撞风险",
                "parameters": {
                    "vehicle1_x": {"type": "number", "description": "车辆1 X坐标"},
                    "vehicle1_y": {"type": "number", "description": "车辆1 Y坐标"},
                    "vehicle1_heading": {"type": "number", "description": "车辆1航向角"},
                    "vehicle2_x": {"type": "number", "description": "车辆2 X坐标"},
                    "vehicle2_y": {"type": "number", "description": "车辆2 Y坐标"},
                    "vehicle2_heading": {"type": "number", "description": "车辆2航向角"}
                },
                "handler": self._analyze_collision_risk
            },
            {
                "name": "get_lane_change_possibility",
                "description": "判断车辆变道的可能性",
                "parameters": {
                    "x": {"type": "number", "description": "车辆X坐标"},
                    "y": {"type": "number", "description": "车辆Y坐标"},
                    "heading": {"type": "number", "description": "车辆航向角(度)"}
                },
                "handler": self._get_lane_change_possibility
            },
        ]

    def get_system_prompt(self) -> str:
        return """你是一个车辆行为分析专家，专门预测和分析车辆行为。

你的职责包括：
1. 分析车辆当前所在车道
2. 预测车辆下一步行为（转向、变道等）
3. 评估车辆行为的风险等级
4. 分析事故责任

回答时请：
- 基于数据给出判断
- 说明推理依据
- 给出置信度估计"""

    def process(self, query: str, **kwargs) -> Dict:
        """
        处理行为分析查询

        Args:
            query: 用户查询
            **kwargs: 额外参数 (vehicle_id, location, heading 等)

        Returns:
            BehaviorResult 字典
        """
        behavior_query = self._parse_query(query, **kwargs)
        result = self._analyze_behavior(behavior_query)
        return result

    def _parse_query(self, query: str, **kwargs) -> BehaviorQuery:
        """解析查询参数"""
        return BehaviorQuery(
            question=query,
            vehicle_id=kwargs.get("vehicle_id"),
            location=kwargs.get("location"),
            heading=kwargs.get("heading"),
            speed=kwargs.get("speed"),
            context=kwargs.get("context")
        )

    def _analyze_behavior(self, query: BehaviorQuery) -> Dict:
        """执行行为分析"""
        result = BehaviorResult(
            predicted_action=VehicleAction.UNKNOWN.value,
            confidence=0.0,
            reasoning=""
        )

        location = query.location
        heading = query.heading
        speed = query.speed

        if not location:
            result.reasoning = "未提供车辆位置信息"
            return result.to_dict()

        # 匹配到车道
        match_result = self.map_api.match_vehicle_to_lane(
            position=location,
            heading=heading,
            max_distance=20.0
        )

        if not match_result:
            result.reasoning = "未能匹配到车道"
            result.predicted_action = VehicleAction.UNKNOWN.value
            return result.to_dict()

        # 基于车道信息预测行为
        centerline_id = match_result['centerline_id']
        cl_info = self.map_api.get_centerline_info(centerline_id)

        if not cl_info:
            result.reasoning = "无法获取车道信息"
            return result.to_dict()

        # 分析行为
        action, confidence, reasoning = self._predict_action_from_lane(
            cl_info, heading, speed, match_result
        )

        result.predicted_action = action
        result.confidence = confidence
        result.reasoning = reasoning

        # 评估风险
        result.risk_level = self._assess_risk(action, speed, cl_info)

        return result.to_dict()

    def _predict_action_from_lane(self, cl_info: Dict, heading: Optional[float],
                                   speed: Optional[float], match_result: Dict) -> Tuple[str, float, str]:
        """基于车道信息预测行为"""
        reasoning_parts = []
        confidence = 0.5

        # 检查车道边界类型
        left_boundary = cl_info.get('left_boundary_id')
        right_boundary = cl_info.get('right_boundary_id')

        # 检查车道线类型
        lane_type = "unknown"
        if right_boundary:
            lane_info = self.map_api.get_lane_info(right_boundary)
            if lane_info:
                lane_type = lane_info['type']

        # 基本预测
        action = VehicleAction.STRAIGHT.value

        # 根据车道线类型判断
        if lane_type == "double_solid":
            action = VehicleAction.STRAIGHT.value
            confidence = 0.8
            reasoning_parts.append("双实线，不允许变道")
        elif lane_type == "dashed":
            confidence = 0.6
            reasoning_parts.append("虚线，可以变道")
        elif lane_type == "left_dashed_right_solid":
            confidence = 0.7
            reasoning_parts.append("左虚右实，可向左变道")

        # 根据速度判断
        if speed is not None:
            if speed < 0.5:
                action = VehicleAction.STOP.value
                confidence = 0.9
                reasoning_parts.append(f"速度很低({speed:.1f}m/s)，可能停车")
            elif speed < 5:
                reasoning_parts.append(f"速度较低({speed:.1f}m/s)")

        # 根据车道拓扑判断转向
        successors = cl_info.get('successor_ids', [])
        if len(successors) > 1:
            reasoning_parts.append("车道分流，可能转向")
            confidence = 0.6

        predecessors = cl_info.get('predecessor_ids', [])
        if len(predecessors) > 1:
            reasoning_parts.append("车道合流，注意侧方来车")

        reasoning = "；".join(reasoning_parts) if reasoning_parts else "车辆沿车道行驶"
        return action, confidence, reasoning

    def _assess_risk(self, action: str, speed: Optional[float], cl_info: Dict) -> str:
        """评估风险等级"""
        risk = RiskLevel.LOW.value

        # 高速变道高风险
        if action in [VehicleAction.CHANGE_LANE_LEFT.value, VehicleAction.CHANGE_LANE_RIGHT.value]:
            if speed and speed > 15:
                risk = RiskLevel.HIGH.value
            elif speed and speed > 10:
                risk = RiskLevel.MEDIUM.value

        # 分流区域中等风险
        if len(cl_info.get('successor_ids', [])) > 1:
            risk = RiskLevel.MEDIUM.value

        # 合流区域中高风险
        if len(cl_info.get('predecessor_ids', [])) > 1:
            risk = RiskLevel.HIGH.value if risk == RiskLevel.MEDIUM.value else RiskLevel.MEDIUM.value

        return risk

    # ========== 工具处理函数 ==========

    def _match_vehicle_to_lane(self, x: float, y: float, heading: float = 0) -> Dict:
        """将车辆匹配到车道"""
        position = (x, y, 0)

        match = self.map_api.match_vehicle_to_lane(position, heading, max_distance=20)

        if match:
            return {
                "success": True,
                "centerline_id": match['centerline_id'],
                "distance_to_lane": match['distance'],
                "lane_heading": match['heading'],
                "is_forward": match['is_forward'],
                "projected_point": match['projected_point']
            }
        else:
            return {
                "success": False,
                "message": "未能在20米内找到匹配车道"
            }

    def _predict_vehicle_action(self, x: float, y: float, heading: float, speed: float = 0) -> Dict:
        """预测车辆行为"""
        result = self._analyze_behavior(BehaviorQuery(
            question="",
            location=(x, y, 0),
            heading=heading,
            speed=speed
        ))
        return result

    def _analyze_collision_risk(self, vehicle1_x: float, vehicle1_y: float, vehicle1_heading: float,
                                 vehicle2_x: float, vehicle2_y: float, vehicle2_heading: float) -> Dict:
        """分析碰撞风险"""
        # 计算两车距离
        dist = math.sqrt((vehicle1_x - vehicle2_x)**2 + (vehicle1_y - vehicle2_y)**2)

        # 计算相对航向角
        heading_diff = abs(vehicle1_heading - vehicle2_heading)
        if heading_diff > 180:
            heading_diff = 360 - heading_diff

        # 判断是否同向
        is_same_direction = heading_diff < 30

        # 风险评估
        if dist < 10:
            risk = "high"
        elif dist < 30:
            risk = "medium"
        else:
            risk = "low"

        return {
            "distance": round(dist, 1),
            "heading_difference": round(heading_diff, 1),
            "is_same_direction": is_same_direction,
            "collision_risk": risk,
            "warning": "注意保持车距" if risk != "low" else "安全"
        }

    def _get_lane_change_possibility(self, x: float, y: float, heading: float) -> Dict:
        """判断变道可能性"""
        match = self.map_api.match_vehicle_to_lane((x, y, 0), heading, max_distance=20)

        if not match:
            return {"can_change_left": False, "can_change_right": False, "reason": "未匹配到车道"}

        cl_info = self.map_api.get_centerline_info(match['centerline_id'])
        if not cl_info:
            return {"can_change_left": False, "can_change_right": False, "reason": "无法获取车道信息"}

        # 检查左边界
        can_left = False
        left_reason = ""
        if cl_info.get('left_boundary_id'):
            left_lane = self.map_api.get_lane_info(cl_info['left_boundary_id'])
            if left_lane:
                if left_lane['type'] in ['dashed', 'double_dashed']:
                    can_left = True
                    left_reason = "左侧虚线，可变道"
                else:
                    left_reason = f"左侧{left_lane['type']}，不可变道"
        else:
            left_reason = "无左边界"

        # 检查右边界
        can_right = False
        right_reason = ""
        if cl_info.get('right_boundary_id'):
            right_lane = self.map_api.get_lane_info(cl_info['right_boundary_id'])
            if right_lane:
                if right_lane['type'] in ['dashed', 'double_dashed', 'left_dashed_right_solid']:
                    can_right = True
                    right_reason = "右侧虚线，可变道"
                else:
                    right_reason = f"右侧{right_lane['type']}，不可变道"
        else:
            right_reason = "无右边界"

        return {
            "can_change_left": can_left,
            "can_change_right": can_right,
            "left_reason": left_reason,
            "right_reason": right_reason
        }