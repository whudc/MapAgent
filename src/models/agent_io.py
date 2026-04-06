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
    TRAFFIC_FLOW_RECONSTRUCTION = "traffic_flow"  # 交通流重建类
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


# ==================== 交通流重建 Agent ====================

class VehicleState(BaseModel):
    """单个车辆在某帧的状态"""
    frame_id: int = Field(..., description="帧ID")
    timestamp: Optional[float] = Field(None, description="时间戳")
    vehicle_id: int = Field(..., description="车辆ID")
    vehicle_type: str = Field(default="Car", description="车辆类型")
    position: Tuple[float, float, float] = Field(..., description="位置坐标")
    size: Tuple[float, float, float] = Field(default=(4.0, 2.0, 1.5), description="尺寸")
    rotation: Tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="旋转角度")
    velocity: Tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="速度")
    heading: Optional[float] = Field(None, description="航向角(度)")
    speed: Optional[float] = Field(None, description="速度大小(m/s)")
    matched_lane: Optional[str] = Field(None, description="匹配的车道ID")
    behavior: Optional[str] = Field(None, description="推断的行为")

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "position": list(self.position),
            "size": list(self.size),
            "rotation": list(self.rotation),
            "velocity": list(self.velocity),
            "heading": self.heading,
            "speed": self.speed,
            "matched_lane": self.matched_lane,
            "behavior": self.behavior
        }


class VehicleTrajectory(BaseModel):
    """单个车辆的完整轨迹"""
    vehicle_id: int = Field(..., description="车辆ID")
    vehicle_type: str = Field(default="Car", description="车辆类型")
    states: List[VehicleState] = Field(default_factory=list, description="轨迹状态序列")
    start_frame: int = Field(default=0, description="起始帧")
    end_frame: int = Field(default=0, description="结束帧")
    total_distance: float = Field(default=0.0, description="总行驶距离")
    avg_speed: float = Field(default=0.0, description="平均速度")
    behaviors: List[str] = Field(default_factory=list, description="行为序列")

    def to_dict(self) -> Dict:
        return {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": self.vehicle_type,
            "states": [s.to_dict() for s in self.states],
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "total_distance": self.total_distance,
            "avg_speed": self.avg_speed,
            "behaviors": self.behaviors
        }


class FrameData(BaseModel):
    """单帧交通流数据"""
    frame_id: int = Field(..., description="帧ID")
    timestamp: Optional[float] = Field(None, description="时间戳")
    vehicles: List[VehicleState] = Field(default_factory=list, description="该帧所有车辆状态")
    vehicle_count: int = Field(default=0, description="车辆数量")
    ego_position: Optional[Tuple[float, float, float]] = Field(None, description="自车位置")
    ego_velocity: Optional[Tuple[float, float, float]] = Field(None, description="自车速度")

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "vehicle_count": self.vehicle_count,
            "ego_position": list(self.ego_position) if self.ego_position else None,
            "ego_velocity": list(self.ego_velocity) if self.ego_velocity else None
        }


class TrafficFlowQuery(BaseModel):
    """交通流重建查询"""
    question: str = Field(..., description="用户问题")
    detection_path: str = Field(..., description="检测结果目录路径")
    start_frame: Optional[int] = Field(None, description="起始帧")
    end_frame: Optional[int] = Field(None, description="结束帧")
    output_path: Optional[str] = Field(None, description="输出结果路径")
    use_llm_inference: bool = Field(default=True, description="是否使用LLM推理补充")


class TrafficFlowResult(BaseModel):
    """交通流重建结果"""
    success: bool = Field(default=True, description="是否成功")
    trajectories: List[Dict] = Field(default_factory=list, description="所有车辆轨迹")
    frames: List[Dict] = Field(default_factory=list, description="帧数据序列")
    total_frames: int = Field(default=0, description="总帧数")
    total_vehicles: int = Field(default=0, description="车辆总数")
    duration_seconds: float = Field(default=0.0, description="时长(秒)")
    summary: str = Field(default="", description="交通流概述")
    output_file: Optional[str] = Field(None, description="输出文件路径")
    reconstruction_stats: Optional[Dict[str, Any]] = Field(None, description="重建统计信息")

    def to_response_text(self) -> str:
        """生成自然语言回复"""
        parts = [f"交通流重建完成。"]
        parts.append(f"共处理 {self.total_frames} 帧，{self.total_vehicles} 辆车辆。")
        parts.append(f"时长: {self.duration_seconds:.1f} 秒。")
        if self.summary:
            parts.append(self.summary)
        if self.output_file:
            parts.append(f"结果已保存至: {self.output_file}")
        return " ".join(parts)

    def to_dict(self) -> Dict:
        return self.model_dump(exclude_none=True)