"""
交通流重建 Agent

基于检测结果重建交通流轨迹，结合规则推理和LLM推理
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math
import json
from pathlib import Path
import sys

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agents.base import BaseAgent, AgentContext
from models.agent_io import (
    TrafficFlowQuery, TrafficFlowResult,
    VehicleState, VehicleTrajectory, FrameData
)
from utils.detection_loader import DetectionLoader, FrameDetection, DetectedObject


class TrafficFlowAgent(BaseAgent):
    """
    交通流重建 Agent

    功能：
    - 加载检测结果数据
    - 基于规则推理重建车辆轨迹
    - 使用LLM补充推理异常/缺失数据
    - 分析整体交通流
    - 保存重建结果
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "traffic_flow_agent"
        self._loader: Optional[DetectionLoader] = None
        self._trajectories: Dict[int, VehicleTrajectory] = {}
        self._frames: List[FrameData] = []

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
                "name": "reconstruct_traffic_flow",
                "description": "重建交通流轨迹",
                "parameters": {
                    "start_frame": {"type": "integer", "description": "起始帧ID", "default": None},
                    "end_frame": {"type": "integer", "description": "结束帧ID", "default": None},
                    "use_llm": {"type": "boolean", "description": "是否使用LLM补充推理", "default": True}
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
            }
        ]

    def get_system_prompt(self) -> str:
        return """你是一个交通流分析专家，专门重建和分析车辆轨迹。

你的职责包括：
1. 加载和处理检测结果数据
2. 重建车辆轨迹序列
3. 分析车辆行为（变道、转向、跟车等）
4. 评估交通流状态
5. 对于缺失或异常数据，进行合理推断

回答时请：
- 基于数据给出分析结果
- 说明推理依据和置信度
- 提供详细的轨迹信息"""

    def process(self, query: str, **kwargs) -> Dict:
        """
        处理交通流重建请求

        Args:
            query: 用户问题
            **kwargs: 额外参数 (detection_path, start_frame, end_frame 等)

        Returns:
            TrafficFlowResult 字典
        """
        # 解析查询
        flow_query = self._parse_query(query, **kwargs)

        # 加载检测结果
        if not self._loader:
            if not flow_query.detection_path:
                return TrafficFlowResult(
                    success=False,
                    summary="请提供检测结果目录路径"
                ).to_dict()

            try:
                self._loader = DetectionLoader(flow_query.detection_path)
            except Exception as e:
                return TrafficFlowResult(
                    success=False,
                    summary=f"加载检测结果失败: {str(e)}"
                ).to_dict()

        # 执行重建
        result = self._reconstruct(flow_query)

        # 保存结果
        if flow_query.output_path and result.success:
            self._save_result(flow_query.output_path, result)

        return result.to_dict()

    def _parse_query(self, query: str, **kwargs) -> TrafficFlowQuery:
        """解析查询参数"""
        # 尝试从query中提取路径
        detection_path = kwargs.get('detection_path')

        if not detection_path:
            # 使用简单模式匹配提取路径
            import re
            match = re.search(r'(?:位于|在|路径(?:是)?[:\s])([^\s,，。]+)', query)
            if match:
                detection_path = match.group(1).strip()

        return TrafficFlowQuery(
            question=query,
            detection_path=detection_path or "",
            start_frame=kwargs.get('start_frame'),
            end_frame=kwargs.get('end_frame'),
            output_path=kwargs.get('output_path'),
            use_llm_inference=kwargs.get('use_llm', True)
        )

    def _reconstruct(self, query: TrafficFlowQuery) -> TrafficFlowResult:
        """执行交通流重建"""
        if not self._loader:
            return TrafficFlowResult(success=False, summary="检测结果未加载")

        # 运行跟踪算法（确保跟踪ID已分配）
        self._loader.run_tracking(query.start_frame, query.end_frame)

        # 加载帧数据
        frames = self._loader.load_frames(query.start_frame, query.end_frame)

        if not frames:
            return TrafficFlowResult(success=False, summary="未找到有效的帧数据")

        # 重建轨迹
        self._trajectories = {}
        self._frames = []

        # 车辆类型
        vehicle_types = ['Car', 'Truck', 'Bus', 'Suv', 'Non_motor_rider', 'Motorcycle']

        # 遍历帧，使用跟踪ID构建轨迹
        for frame in frames:
            frame_data = self._process_frame(frame)
            self._frames.append(frame_data)

            # 更新轨迹（使用tracking_id）
            for vehicle_state in frame_data.vehicles:
                vid = vehicle_state.vehicle_id  # 这里已经是tracking_id
                if vid not in self._trajectories:
                    self._trajectories[vid] = VehicleTrajectory(
                        vehicle_id=vid,
                        vehicle_type=vehicle_state.vehicle_type,
                        states=[]
                    )
                self._trajectories[vid].states.append(vehicle_state)

        # 计算轨迹统计和推断行为
        for vid, traj in self._trajectories.items():
            if traj.states:
                traj.start_frame = traj.states[0].frame_id
                traj.end_frame = traj.states[-1].frame_id
                traj.total_distance = self._calculate_total_distance(traj.states)
                traj.avg_speed = self._calculate_avg_speed(traj.states)
                traj.behaviors = self._infer_behaviors(traj.states)

        # 使用LLM补充推理（如果启用）
        if query.use_llm_inference and self._is_llm_available():
            self._llm_inference(query)

        # 构建结果
        result = TrafficFlowResult(
            success=True,
            trajectories=[t.to_dict() for t in self._trajectories.values()],
            frames=[f.to_dict() for f in self._frames],
            total_frames=len(self._frames),
            total_vehicles=len(self._trajectories),
            duration_seconds=self._calculate_duration(frames),
            summary=self._generate_summary()
        )

        return result

    def _process_frame(self, frame: FrameDetection) -> FrameData:
        """处理单帧数据"""
        vehicles = []

        # 车辆类型
        vehicle_types = ['Car', 'Truck', 'Bus', 'Suv', 'Non_motor_rider', 'Motorcycle']

        for obj in frame.objects:
            if obj.type in vehicle_types:
                # 计算航向角和速度
                heading = self._calculate_heading(obj.rotation)
                speed = obj.speed

                # 尝试匹配车道
                matched_lane = None
                if self.map_api:
                    try:
                        match_result = self.map_api.match_vehicle_to_lane(
                            obj.location, heading, max_distance=20
                        )
                        if match_result:
                            matched_lane = match_result.get('centerline_id')
                    except Exception:
                        pass

                # 使用tracking_id作为vehicle_id
                vehicle_id = obj.tracking_id if obj.tracking_id > 0 else obj.id

                vehicle_state = VehicleState(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    vehicle_id=vehicle_id,
                    vehicle_type=obj.type,
                    position=obj.location,
                    size=obj.size,
                    rotation=obj.rotation,
                    velocity=obj.velocity,
                    heading=heading,
                    speed=speed,
                    matched_lane=matched_lane
                )
                vehicles.append(vehicle_state)

        return FrameData(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            vehicles=vehicles,
            vehicle_count=len(vehicles),
            ego_position=frame.ego_position,
            ego_velocity=frame.ego_velocity
        )

    def _calculate_heading(self, rotation: Tuple[float, float, float]) -> float:
        """计算航向角（转换到0-360度范围）"""
        raw_heading = rotation[2]  # Z轴旋转
        # 转换到0-360度
        heading_deg = math.degrees(raw_heading) if raw_heading < 0 else math.degrees(raw_heading)
        heading_deg = heading_deg % 360
        if heading_deg < 0:
            heading_deg += 360
        return heading_deg

    def _calculate_total_distance(self, states: List[VehicleState]) -> float:
        """计算总行驶距离"""
        total = 0.0
        for i in range(1, len(states)):
            prev = states[i-1].position
            curr = states[i].position
            dist = math.sqrt(
                (curr[0] - prev[0])**2 +
                (curr[1] - prev[1])**2
            )
            total += dist
        return total

    def _calculate_avg_speed(self, states: List[VehicleState]) -> float:
        """计算平均速度"""
        if not states:
            return 0.0
        speeds = [s.speed for s in states if s.speed is not None]
        if speeds:
            return sum(speeds) / len(speeds)
        return 0.0

    def _infer_behaviors(self, states: List[VehicleState]) -> List[str]:
        """推断行为序列（规则推理）"""
        behaviors = []

        for i, state in enumerate(states):
            if i == 0:
                behaviors.append("start")
                continue

            prev = states[i-1]
            behavior = self._infer_single_behavior(prev, state)
            behaviors.append(behavior)

        return behaviors

    def _infer_single_behavior(self, prev: VehicleState, curr: VehicleState) -> str:
        """推断单步行为"""
        # 计算位置变化
        dx = curr.position[0] - prev.position[0]
        dy = curr.position[1] - prev.position[1]
        dist = math.sqrt(dx**2 + dy**2)

        # 计算航向角变化
        heading_change = 0
        if prev.heading and curr.heading:
            diff = abs(curr.heading - prev.heading)
            if diff > 180:
                diff = 360 - diff
            heading_change = diff

        # 判断行为
        if dist < 0.5:  # 几乎静止
            return "stop"

        # 根据航向角变化判断转向
        if heading_change > 30:
            if curr.heading > prev.heading:
                return "left_turn"
            else:
                return "right_turn"

        # 根据车道变化判断变道
        if prev.matched_lane and curr.matched_lane:
            if prev.matched_lane != curr.matched_lane:
                return "change_lane"

        # 速度变化判断
        if prev.speed and curr.speed:
            if curr.speed > prev.speed * 1.2:
                return "accelerate"
            elif curr.speed < prev.speed * 0.8:
                return "slow_down"

        return "straight"

    def _calculate_duration(self, frames: List[FrameDetection]) -> float:
        """计算时长"""
        if len(frames) < 2:
            return 0.0

        first_ts = frames[0].timestamp
        last_ts = frames[-1].timestamp

        if first_ts and last_ts:
            return last_ts - first_ts

        # 假设10Hz
        return len(frames) / 10.0

    def _generate_summary(self) -> str:
        """生成交通流摘要"""
        if not self._trajectories:
            return "未重建任何轨迹"

        parts = []

        # 统计车辆类型
        type_counts = {}
        for traj in self._trajectories.values():
            type_counts[traj.vehicle_type] = type_counts.get(traj.vehicle_type, 0) + 1

        type_str = ", ".join(f"{t}: {c}辆" for t, c in type_counts.items())
        parts.append(f"车辆类型分布: {type_str}")

        # 统计行为
        behavior_counts = {"straight": 0, "left_turn": 0, "right_turn": 0,
                          "change_lane": 0, "stop": 0, "accelerate": 0, "slow_down": 0}
        for traj in self._trajectories.values():
            for b in traj.behaviors:
                if b in behavior_counts:
                    behavior_counts[b] += 1

        behavior_str = ", ".join(f"{b}: {c}次" for b, c in behavior_counts.items() if c > 0)
        parts.append(f"行为统计: {behavior_str}")

        return "。".join(parts)

    def _llm_inference(self, query: TrafficFlowQuery):
        """使用LLM补充推理"""
        # 检查LLM客户端是否有效
        if not self._is_llm_available():
            print("LLM不可用，跳过LLM推理优化")
            return

        # 对缺失帧或异常数据进行LLM推理
        try:
            for vid, traj in self._trajectories.items():
                # 检测缺失帧
                missing_frames = self._detect_missing_frames(traj)

                if missing_frames:
                    # 使用LLM推断缺失状态
                    inferred_states = self._llm_infer_missing(traj, missing_frames)
                    # 插入推断的状态
                    for state in inferred_states:
                        self._insert_state(traj, state)
        except Exception as e:
            print(f"LLM推理出错，已跳过: {e}")

    def _is_llm_available(self) -> bool:
        """检查LLM是否可用"""
        if not self.llm_client:
            return False

        # 检查是否有有效的API key配置
        config = getattr(self.llm_client, 'config', None)
        if config:
            api_key = getattr(config, 'api_key', None)
            # 检查是否是无效的dummy key
            if not api_key or api_key == 'dummy':
                return False

        return True

    def _detect_missing_frames(self, traj: VehicleTrajectory) -> List[int]:
        """检测缺失帧"""
        if len(traj.states) < 2:
            return []

        all_frame_ids = self._loader.get_frame_ids()
        traj_frame_ids = [s.frame_id for s in traj.states]

        missing = []
        start_idx = all_frame_ids.index(traj_frame_ids[0]) if traj_frame_ids[0] in all_frame_ids else 0
        end_idx = all_frame_ids.index(traj_frame_ids[-1]) if traj_frame_ids[-1] in all_frame_ids else len(all_frame_ids)

        for i in range(start_idx, end_idx + 1):
            if all_frame_ids[i] not in traj_frame_ids:
                missing.append(all_frame_ids[i])

        return missing

    def _llm_infer_missing(self, traj: VehicleTrajectory, missing_frames: List[int]) -> List[VehicleState]:
        """使用LLM推断缺失帧状态"""
        if not self._is_llm_available() or not missing_frames:
            return []

        # 准备上下文
        context = {
            "vehicle_id": traj.vehicle_id,
            "vehicle_type": traj.vehicle_type,
            "known_states": [s.to_dict() for s in traj.states[:5]],  # 前5个已知状态
            "missing_frames": missing_frames[:3]  # 最多推断3个缺失帧
        }

        prompt = f"""基于以下车辆轨迹信息，推断缺失帧的车辆状态：

车辆ID: {traj.vehicle_id}
车辆类型: {traj.vehicle_type}
已知状态: {json.dumps(context['known_states'], ensure_ascii=False)}
缺失帧: {missing_frames[:3]}

请为每个缺失帧推断合理的：
1. 位置 (基于前一帧位置和速度)
2. 航向角 (保持一致或小幅变化)
3. 速度 (平滑过渡)

以JSON数组格式返回推断的状态。"""

        try:
            response = self.llm_client.chat_simple(prompt)

            # 尝试解析LLM返回的JSON
            # 这里简化处理，实际需要更复杂的解析逻辑
            # 对于每个缺失帧，基于前一帧线性推断
            inferred = []
            for mf in missing_frames[:3]:
                # 找到最近的已知帧
                prev_state = None
                for s in traj.states:
                    if s.frame_id < mf:
                        prev_state = s
                    elif s.frame_id > mf:
                        break

                if prev_state:
                    # 简单线性推断
                    dt = 0.1  # 假设0.1秒间隔
                    vx, vy, vz = prev_state.velocity
                    new_pos = (
                        prev_state.position[0] + vx * dt,
                        prev_state.position[1] + vy * dt,
                        prev_state.position[2]
                    )

                    inferred_state = VehicleState(
                        frame_id=mf,
                        timestamp=prev_state.timestamp + dt if prev_state.timestamp else None,
                        vehicle_id=traj.vehicle_id,
                        vehicle_type=traj.vehicle_type,
                        position=new_pos,
                        velocity=prev_state.velocity,
                        heading=prev_state.heading,
                        speed=prev_state.speed,
                        behavior="inferred"
                    )
                    inferred.append(inferred_state)

            return inferred

        except Exception as e:
            print(f"LLM推理失败: {e}")
            return []

    def _insert_state(self, traj: VehicleTrajectory, state: VehicleState):
        """将状态插入轨迹的正确位置"""
        # 找到插入位置
        insert_idx = 0
        for i, s in enumerate(traj.states):
            if s.frame_id < state.frame_id:
                insert_idx = i + 1
            else:
                break

        traj.states.insert(insert_idx, state)

    def _save_result(self, output_path: str, result: TrafficFlowResult):
        """保存结果到文件"""
        try:
            output_file = Path(output_path)
            if not output_file.is_absolute():
                output_file = Path(self._loader.detection_path).parent / output_file

            with open(output_file, 'w') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            result.output_file = str(output_file)

        except Exception as e:
            print(f"保存结果失败: {e}")

    # ========== 工具处理函数 ==========

    def _load_detection_results(self, path: str) -> Dict:
        """加载检测结果"""
        try:
            self._loader = DetectionLoader(path)
            summary = self._loader.get_summary()
            return {
                "success": True,
                "summary": summary,
                "message": f"成功加载 {summary['total_frames']} 帧检测结果"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _reconstruct_traffic_flow(self, start_frame: int = None,
                                   end_frame: int = None,
                                   use_llm: bool = True) -> Dict:
        """重建交通流"""
        if not self._loader:
            return {"success": False, "error": "请先加载检测结果"}

        query = TrafficFlowQuery(
            question="重建交通流",
            detection_path=str(self._loader.detection_path),
            start_frame=start_frame,
            end_frame=end_frame,
            use_llm_inference=use_llm
        )

        result = self._reconstruct(query)
        return result.to_dict()

    def _get_trajectory_by_id(self, vehicle_id: int) -> Dict:
        """获取指定车辆轨迹"""
        if vehicle_id in self._trajectories:
            return {
                "success": True,
                "trajectory": self._trajectories[vehicle_id].to_dict()
            }
        return {
            "success": False,
            "error": f"未找到车辆 {vehicle_id} 的轨迹"
        }

    def _analyze_vehicle_behavior(self, vehicle_id: int, frame_id: int) -> Dict:
        """分析车辆行为"""
        if vehicle_id not in self._trajectories:
            return {"success": False, "error": "车辆不存在"}

        traj = self._trajectories[vehicle_id]
        for state in traj.states:
            if state.frame_id == frame_id:
                return {
                    "success": True,
                    "vehicle_id": vehicle_id,
                    "frame_id": frame_id,
                    "position": state.position,
                    "heading": state.heading,
                    "speed": state.speed,
                    "matched_lane": state.matched_lane,
                    "behavior": state.behavior
                }

        return {"success": False, "error": "帧不存在"}

    def _save_reconstruction_result(self, output_path: str = "reconstruction_result.json") -> Dict:
        """保存重建结果"""
        if not self._trajectories:
            return {"success": False, "error": "没有重建结果"}

        result = TrafficFlowResult(
            success=True,
            trajectories=[t.to_dict() for t in self._trajectories.values()],
            frames=[f.to_dict() for f in self._frames],
            total_frames=len(self._frames),
            total_vehicles=len(self._trajectories),
            summary=self._generate_summary()
        )

        self._save_result(output_path, result)

        return {
            "success": True,
            "output_file": result.output_file,
            "message": f"结果已保存至 {result.output_file}"
        }

    def _get_traffic_flow_summary(self) -> Dict:
        """获取交通流摘要"""
        return {
            "success": True,
            "total_frames": len(self._frames),
            "total_vehicles": len(self._trajectories),
            "summary": self._generate_summary(),
            "vehicle_ids": list(self._trajectories.keys())
        }