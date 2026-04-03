"""
Agent 输入输出数据模型

定义各 Agent 的请求和响应结构
"""

from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """用户意图类型"""
    SCENE_UNDERSTANDING = "scene"      # 场景理解类
    BEHAVIOR_ANALYSIS = "behavior"     # 行为分析类
    PATH_PLANNING = "path"             # 路径规划类
    GENERAL_QUERY = "general"          # 通用查询类
    UNKNOWN = "unknown"                # 无法识别


class Intent(BaseModel):
    """识别出的意图"""
    type: IntentType = Field(..., description="意图类型")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度")
    entities: Dict[str, Any] = Field(default_factory=dict, description="提取的实体")
    sub_agent: Optional[str] = Field(None, description="对应的子Agent")

    def is_scene_query(self) -> bool:
        return self.type == IntentType.SCENE_UNDERSTANDING

    def is_behavior_query(self) -> bool:
        return self.type == IntentType.BEHAVIOR_ANALYSIS

    def is_path_query(self) -> bool:
        return self.type == IntentType.PATH_PLANNING


# ==================== 场景理解 Agent ====================

class SceneQuery(BaseModel):
    """场景理解查询"""
    question: str = Field(..., description="用户问题")
    location: Optional[Tuple[float, float, float]] = Field(None, description="查询位置坐标")
    radius: float = Field(default=100.0, description="查询半径(米)")
    intersection_id: Optional[str] = Field(None, description="指定路口ID")
    lane_id: Optional[str] = Field(None, description="指定车道ID")


class SceneResult(BaseModel):
    """场景理解结果"""
    summary: str = Field(default="", description="场景概述")
    lane_count: Optional[int] = Field(None, description="车道数")
    lane_types: Optional[Dict[str, int]] = Field(None, description="各类型车道数量")
    intersection_info: Optional[Dict[str, Any]] = Field(None, description="路口详细信息")
    traffic_rules: Optional[List[str]] = Field(None, description="交通规则提示")
    nearby_signs: Optional[List[str]] = Field(None, description="附近交通标志")

    def to_response_text(self) -> str:
        """生成自然语言回复"""
        parts = [self.summary]
        if self.lane_count:
            parts.append(f"共有 {self.lane_count} 条车道。")
        if self.lane_types:
            type_desc = ", ".join(f"{k}: {v}条" for k, v in self.lane_types.items())
            parts.append(f"车道类型分布: {type_desc}。")
        if self.traffic_rules:
            parts.append("交通规则提示: " + "; ".join(self.traffic_rules))
        return " ".join(parts)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.model_dump(exclude_none=True)


# ==================== 行为分析 Agent ====================

class VehicleBehavior(str, Enum):
    """车辆行为类型"""
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    U_TURN = "u_turn"
    CHANGE_LANE_LEFT = "change_lane_left"
    CHANGE_LANE_RIGHT = "change_lane_right"
    STOP = "stop"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BehaviorQuery(BaseModel):
    """行为分析查询"""
    question: str = Field(..., description="用户问题")
    vehicle_id: Optional[str] = Field(None, description="目标车辆ID")
    location: Optional[Tuple[float, float, float]] = Field(None, description="车辆位置")
    heading: Optional[float] = Field(None, description="航向角")
    speed: Optional[float] = Field(None, description="速度")
    context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")


class BehaviorResult(BaseModel):
    """行为分析结果"""
    predicted_action: str = Field(default="unknown", description="预测行为")
    confidence: float = Field(default=0.0, description="置信度 0-1")
    reasoning: str = Field(default="", description="推理依据")
    risk_level: str = Field(default="low", description="风险等级")
    liability_analysis: Optional[Dict[str, Any]] = Field(None, description="责任分析")

    def to_response_text(self) -> str:
        """生成自然语言回复"""
        parts = [f"预测行为: {self.predicted_action}，置信度 {self.confidence:.0%}。"]
        parts.append(f"依据: {self.reasoning}")
        if self.risk_level != "low":
            parts.append(f"风险等级: {self.risk_level}")
        return " ".join(parts)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.model_dump(exclude_none=True)


# ==================== 路径规划 Agent ====================

class PathQuery(BaseModel):
    """路径规划查询"""
    question: str = Field(..., description="用户问题")
    origin: Optional[Tuple[float, float, float]] = Field(None, description="起点坐标")
    destination: Optional[Tuple[float, float, float]] = Field(None, description="终点坐标")
    destination_name: Optional[str] = Field(None, description="目的地名称")
    preferences: Optional[Dict[str, Any]] = Field(None, description="偏好设置")


class PathInfo(BaseModel):
    """单条路径信息"""
    id: str = Field(default="", description="路径ID")
    waypoints: List[Tuple[float, float, float]] = Field(default_factory=list, description="路径点列表")
    distance: float = Field(default=0.0, description="距离(米)")
    estimated_time: float = Field(default=0.0, description="预估时间(秒)")
    lane_ids: List[str] = Field(default_factory=list, description="经过的车道ID")

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "waypoints": [list(w) for w in self.waypoints],
            "distance": self.distance,
            "estimated_time": self.estimated_time,
            "lane_ids": self.lane_ids
        }


class PathResult(BaseModel):
    """路径规划结果"""
    paths: List[Dict] = Field(default_factory=list, description="推荐路径列表")
    best_path: Optional[Dict] = Field(None, description="最佳路径")
    advice: str = Field(default="", description="车流建议")
    estimated_time: float = Field(default=0.0, description="预估时间(分钟)")
    distance: float = Field(default=0.0, description="总距离(米)")

    def to_response_text(self) -> str:
        """生成自然语言回复"""
        if self.best_path:
            parts = [f"推荐路径: 距离 {self.distance:.0f}米，预估时间 {self.estimated_time:.1f}分钟。"]
        else:
            parts = ["未找到有效路径。"]
        if self.advice:
            parts.append(self.advice)
        return " ".join(parts)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.model_dump(exclude_none=True)


# ==================== 通用响应 ====================

class AgentResponse(BaseModel):
    """Agent通用响应"""
    success: bool = Field(default=True, description="是否成功")
    intent: Optional[Intent] = Field(None, description="识别的意图")
    result: Optional[Any] = Field(None, description="结果数据")
    response_text: str = Field(default="", description="自然语言回复")
    error: Optional[str] = Field(None, description="错误信息")