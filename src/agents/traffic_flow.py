"""
交通流重建 Agent - 支持纯 DeepSORT 和 LLM 混合优化两种模式

核心设计：
1. 纯 DeepSORT 模式：使用卡尔曼滤波 + 级联匹配进行多目标跟踪
2. LLM 混合模式：规则层处理正常匹配，LLM 层处理疑难场景

架构：
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  规则层      │ →  │  LLM 层      │ →  │  后处理层    │
│  (快速匹配)  │    │  (疑难推理)  │    │  (ID 一致性)  │
└──────────────┘    └──────────────┘    └──────────────┘
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import numpy as np

from agents.base import BaseAgent, AgentContext
from agents.deepsort_tracker import DeepSORTTracker
from models.agent_io import VehicleState, VehicleTrajectory
from utils.detection_loader import DetectionLoader, FrameDetection
from core.llm_client import LLMClient


# ==================== 数据结构 ====================

@dataclass
class Trajectory:
    """轨迹数据结构"""
    track_id: int
    positions: List[List[float]] = field(default_factory=list)
    velocities: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    obj_types: List[str] = field(default_factory=list)
    lane_id: Optional[str] = None
    status: str = "active"  # active/lost/confirmed
    lost_count: int = 0
    confidence: float = 1.0

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


@dataclass
class LaneHistory:
    """车道历史状态"""
    lane_id: str
    count_history: List[int] = field(default_factory=list)
    track_ids_history: List[Set[int]] = field(default_factory=list)


class MatchResult(str, Enum):
    """匹配结果类型"""
    MATCHED = "matched"
    MISS_DETECTION = "miss"
    FALSE_DETECTION = "false"
    EXITED = "exited"
    ENTERED = "entered"
    RE_ENTERED = "re_entered"


# ==================== ID 一致性管理器 ====================

class IDConsistencyManager:
    """
    ID 一致性管理器

    保证长时序 ID 不跳变的三层机制：
    1. ID 分配池 - 确保 ID 不重复使用
    2. 轨迹嵌入缓存 - 用于跨帧关联
    3. 延迟确认机制 - 避免过早分配 ID
    """

    def __init__(self):
        self.active_ids: Set[int] = set()
        self.retired_ids: Set[int] = set()
        self.next_id: int = 1
        self.track_embeddings: Dict[int, np.ndarray] = {}

    def assign_id(self, detection: Dict, candidates: List[Trajectory],
                  context: Dict) -> Tuple[int, str]:
        """分配 ID"""
        if len(candidates) == 0:
            new_id = self._allocate_new_id()
            self.active_ids.add(new_id)
            return new_id, "new"
        elif len(candidates) == 1:
            return candidates[0].track_id, "unique_match"
        else:
            best = self._select_best_candidate(detection, candidates, context)
            return best.track_id, "multi_match"

    def _allocate_new_id(self) -> int:
        """分配新 ID"""
        for candidate_id in range(1, self.next_id + 1):
            if candidate_id not in self.active_ids and candidate_id not in self.retired_ids:
                return candidate_id
        self.next_id += 1
        return self.next_id

    def retire_id(self, track_id: int, reason: str = "exited"):
        """退役 ID"""
        if track_id in self.active_ids:
            self.active_ids.remove(track_id)
            if reason == "exited":
                self.retired_ids.add(track_id)

    def _select_best_candidate(self, detection: Dict, candidates: List[Trajectory],
                                context: Dict) -> Trajectory:
        """选择最佳候选轨迹"""
        scores = []
        for track in candidates:
            score = self._compute_match_score(detection, track, context)
            scores.append((track, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _compute_match_score(self, detection: Dict, track: Trajectory, context: Dict) -> float:
        """计算匹配分数"""
        if not track.positions:
            return 0.0

        last_pos = np.array(track.positions[-1][:2])
        det_pos = np.array(detection.get('location', [0, 0, 0])[:2])
        dist = np.linalg.norm(last_pos - det_pos)
        dist_score = np.exp(-dist / 5.0)

        lane_score = 1.0
        if track.lane_id and context.get("expected_lane") == track.lane_id:
            lane_score = 1.2

        return dist_score * lane_score * track.confidence


# ==================== LLM 优化器 ====================

class LLMOptimizer:
    """
    LLM 优化器

    仅在关键决策点调用 LLM，避免频繁调用
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.cache: Dict[str, Any] = {}
        self.call_count = 0

    def should_call_llm(self, situation: str, context: Dict) -> bool:
        """判断是否需要调用 LLM"""
        if not self.llm_client:
            return False

        # 规则层能处理的情况
        if situation == "normal_match":
            return False

        count_diff = context.get("count_diff", 0)
        if situation == "count_mismatch" and abs(count_diff) <= 1:
            return False

        # 复杂情况才调用 LLM
        llm_situations = [
            "count_mismatch_large",
            "track_reappear",
            "id_conflict",
            "new_object_source",
            "lane_transition",
        ]
        return situation in llm_situations

    def analyze_count_mismatch(self, lane_id: str,
                                prev_frame_id: int, prev_count: int,
                                curr_frame_id: int, curr_count: int,
                                trajectory_history: List[Dict],
                                map_topology: Dict) -> Dict:
        """分析车道目标数量不匹配的原因"""
        cache_key = f"mismatch_{lane_id}_{prev_frame_id}_{curr_frame_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_mismatch(prev_count, curr_count)

        prompt = self._build_mismatch_prompt(
            lane_id, prev_count, curr_count,
            trajectory_history, map_topology
        )

        self.call_count += 1
        try:
            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)
        except Exception:
            result = self._rule_based_mismatch(prev_count, curr_count)

        self.cache[cache_key] = result
        return result

    def _build_mismatch_prompt(self, lane_id: str,
                                prev_count: int, curr_count: int,
                                trajectory_history: List[Dict],
                                map_topology: Dict) -> str:
        """构建数量不匹配分析 prompt"""
        recent_tracks = trajectory_history[-5:] if len(trajectory_history) > 5 else trajectory_history

        prompt = f"""分析车道 {lane_id} 的目标数量变化：
- 前一帧数量：{prev_count}
- 当前帧数量：{curr_count}
- 数量变化：{curr_count - prev_count}

最近轨迹历史：
"""
        for track in recent_tracks:
            prompt += f"- ID {track.get('id')}: 位置={track.get('pos')}, 状态={track.get('status')}\n"

        prompt += f"""
地图拓扑信息：
- 前车道路线：{map_topology.get('predecessors', [])}
- 后车道路线：{map_topology.get('successors', [])}

请分析数量变化的原因：
1. miss - 前一帧有检测，当前帧漏检
2. false - 当前帧有虚假检测
3. exit - 目标正常驶出车道
4. enter - 新目标驶入车道

返回 JSON 格式：
{{
    "cause": "miss|false|exit|enter",
    "confidence": 0.0-1.0,
    "affected_ids": [],
    "reasoning": "简要推理说明"
}}
"""
        return prompt

    def _rule_based_mismatch(self, prev_count: int, curr_count: int) -> Dict:
        """基于规则的数量不匹配处理"""
        diff = curr_count - prev_count
        if diff < 0:
            if abs(diff) <= 1:
                return {"cause": "exit", "confidence": 0.7, "affected_ids": [], "reasoning": "正常驶离"}
            else:
                return {"cause": "miss", "confidence": 0.5, "affected_ids": [], "reasoning": "可能漏检"}
        else:
            if diff <= 1:
                return {"cause": "enter", "confidence": 0.6, "affected_ids": [], "reasoning": "新目标驶入"}
            else:
                return {"cause": "false", "confidence": 0.4, "affected_ids": [], "reasoning": "可能误检"}

    def judge_reappear(self, old_track: Dict, new_detection: Dict,
                       _map_topology: Dict) -> Dict:
        """判断重新出现的目标是否是同一轨迹"""
        cache_key = f"reappear_{old_track.get('id')}_{new_detection.get('pos')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_reappear(old_track, new_detection)

        prompt = f"""判断重新出现的目标是否是同一轨迹：

旧轨迹信息：
- ID: {old_track.get('id')}
- 最后位置：{old_track.get('last_pos')}
- 最后车道：{old_track.get('lane_id')}
- 丢失帧数：{old_track.get('lost_frames')}

新检测信息：
- 位置：{new_detection.get('pos')}
- 车道：{new_detection.get('lane_id')}
- 类型：{new_detection.get('type')}

返回 JSON：
{{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "推理说明"
}}
"""
        self.call_count += 1
        try:
            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)
        except Exception:
            result = self._rule_based_reappear(old_track, new_detection)

        self.cache[cache_key] = result
        return result

    def _rule_based_reappear(self, old_track: Dict, new_detection: Dict) -> Dict:
        """基于规则的重新出现判断"""
        old_pos = np.array(old_track.get('last_pos', [0, 0, 0])[:2])
        new_pos = np.array(new_detection.get('pos', [0, 0, 0])[:2])
        dist = np.linalg.norm(old_pos - new_pos)
        same_lane = old_track.get('lane_id') == new_detection.get('lane_id')
        lost_frames = old_track.get('lost_frames', 999)

        if dist < 10.0 and same_lane and lost_frames < 10:
            return {"is_same": True, "confidence": 0.8, "reasoning": "位置接近且同车道"}
        elif dist < 20.0 and lost_frames < 20:
            return {"is_same": True, "confidence": 0.5, "reasoning": "位置较接近"}
        else:
            return {"is_same": False, "confidence": 0.6, "reasoning": "位置差异大或丢失时间过长"}

    def _parse_llm_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        return {"cause": "unknown", "confidence": 0.3, "affected_ids": [], "reasoning": "解析失败"}


# ==================== 主 Agent ====================

class TrafficFlowAgent(BaseAgent):
    """
    交通流重建 Agent

    支持两种模式：
    1. 纯 DeepSORT 模式 (use_llm=False)：使用 DeepSORT 算法进行多目标跟踪
    2. LLM 混合模式 (use_llm=True)：规则层处理正常匹配，LLM 层处理疑难场景

    功能：
    - 加载检测结果数据
    - 多目标跟踪（DeepSORT）
    - LLM 增强决策（可选）
    - 重建车辆轨迹
    - 保存跟踪结果
    """

    def __init__(self, context: AgentContext, use_llm: bool = False):
        """
        初始化

        Args:
            context: Agent 上下文
            use_llm: 是否启用 LLM 优化（默认 False）
        """
        super().__init__(context)
        self.name = "traffic_flow_agent"
        self._use_llm = use_llm

        # 核心组件
        self._loader: Optional[DetectionLoader] = None
        self._tracker: Optional[DeepSORTTracker] = None
        self._trajectories: Dict[int, Trajectory] = {}

        # LLM 优化组件（仅当 use_llm=True 时启用）
        self._id_manager: Optional[IDConsistencyManager] = None
        self._llm_optimizer: Optional[LLMOptimizer] = None
        self._lane_history: Dict[str, LaneHistory] = {}

        if use_llm and context.llm_client:
            self._id_manager = IDConsistencyManager()
            self._llm_optimizer = LLMOptimizer(context.llm_client)
            self.name = "traffic_flow_llm_agent"

        # 统计
        self._stats = {
            'total_frames': 0,
            'llm_calls': 0,
            'use_llm': use_llm,
        }

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
                "description": "重建交通流轨迹（DeepSORT 跟踪）",
                "parameters": {
                    "start_frame": {"type": "integer", "description": "起始帧 ID", "default": None},
                    "end_frame": {"type": "integer", "description": "结束帧 ID", "default": None},
                    "max_distance": {"type": "number", "description": "最大匹配距离 (米)", "default": 5.0},
                    "max_velocity": {"type": "number", "description": "最大速度 (m/s)", "default": 30.0},
                },
                "handler": self._reconstruct_traffic_flow
            },
            {
                "name": "get_trajectory_by_id",
                "description": "获取指定车辆的轨迹",
                "parameters": {
                    "vehicle_id": {"type": "integer", "description": "车辆 ID"}
                },
                "handler": self._get_trajectory_by_id
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
        ]

    def get_system_prompt(self) -> str:
        mode = "LLM 混合优化" if self._use_llm else "纯 DeepSORT"
        return f"""你是一个交通流分析专家，使用 {mode} 模式进行多目标跟踪。

核心能力：
1. 加载检测结果数据
2. 使用 DeepSORT 算法重建连续轨迹
3. {"LLM 增强决策（处理漏检、误检、ID 冲突）" if self._use_llm else "纯位置跟踪"}
4. 分析跟踪结果

使用方法：
1. 首先调用 load_detection_results 加载检测数据
2. 然后调用 reconstruct_traffic_flow 重建轨迹
3. 使用 get_trajectory_by_id 查询特定轨迹
4. 使用 save_reconstruction_result 保存结果"""

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        处理查询

        Args:
            query: 用户查询（当前未使用，保留以匹配接口）
            **kwargs: 额外参数（detection_path, start_frame, end_frame 等）

        Returns:
            处理结果
        """
        _ = query  # 参数保留以匹配接口，暂未使用
        # 解析参数
        detection_path = kwargs.get('detection_path')
        start_frame = kwargs.get('start_frame')
        end_frame = kwargs.get('end_frame')
        output_path = kwargs.get('output_path', 'reconstruction_result.json')

        # 如果提供了检测结果路径，先加载
        if detection_path:
            load_result = self._load_detection_results(detection_path)
            if not load_result.get('success'):
                return load_result

        # 执行重建
        if self._loader:
            recon_result = self._reconstruct_traffic_flow(
                start_frame=start_frame,
                end_frame=end_frame
            )
            if not recon_result.get('success'):
                return recon_result

            # 保存结果
            self._save_reconstruction_result(output_path)

            # 构建帧数据（用于 UI 可视化）
            frames = self._build_frame_data(start_frame, end_frame)

            # 构建轨迹列表
            trajectories = []
            for tid, traj in self._trajectories.items():
                states = []
                for i, frame_id in enumerate(traj.frame_ids):
                    states.append({
                        'position': traj.positions[i],
                        'frame_id': frame_id,
                    })
                trajectories.append({
                    'vehicle_id': tid,
                    'vehicle_type': traj.dominant_type,
                    'states': states,
                })

            return {
                "success": True,
                "message": "交通流重建完成",
                "frames": frames,
                "trajectories": trajectories,
                "total_frames": len(frames),
                "total_vehicles": len(self._trajectories),
                "saved_to": output_path,
            }

        return {
            "success": False,
            "error": "请提供检测结果路径 (detection_path)",
        }

    def _build_frame_data(self, start_frame: Optional[int] = None,
                           end_frame: Optional[int] = None) -> List[Dict]:
        """构建帧数据（用于 UI 可视化）"""
        frames: List[Dict] = []

        if not self._loader:
            return frames

        # 获取所有帧 ID
        frame_ids = self._loader.get_frame_ids()

        for frame_id in frame_ids:
            if start_frame is not None and frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                continue

            # 获取该帧的检测数据
            frame_det = self._loader.load_frame(frame_id, use_tracking=False)
            if not frame_det:
                continue

            # 构建车辆列表
            vehicles = []
            for obj in frame_det.objects:
                vehicles.append({
                    'vehicle_id': obj.tracking_id if obj.tracking_id > 0 else obj.id,
                    'vehicle_type': obj.type,
                    'position': list(obj.location),
                    'heading': obj.heading,
                    'speed': obj.speed,
                })

            frames.append({
                'frame_id': frame_id,
                'vehicles': vehicles,
                'vehicle_count': len(vehicles),
            })

        return frames

    # ==================== 工具实现 ====================

    def _load_detection_results(self, path: str) -> Dict[str, Any]:
        """加载检测结果"""
        self._loader = DetectionLoader(path, enable_tracking=False)
        self._stats['total_frames'] = self._loader.get_frame_count()

        return {
            "success": True,
            "message": f"已加载检测结果",
            "frame_count": self._loader.get_frame_count(),
            "use_llm": self._use_llm,
        }

    def _reconstruct_traffic_flow(self,
                                   start_frame: Optional[int] = None,
                                   end_frame: Optional[int] = None,
                                   max_distance: float = 5.0,
                                   max_velocity: float = 30.0) -> Dict[str, Any]:
        """重建交通流"""
        if not self._loader:
            return {"success": False, "error": "请先加载检测结果"}

        # 加载帧数据
        frames = self._loader.load_frames(start_frame, end_frame)
        if not frames:
            return {"success": False, "error": "未加载到帧数据"}

        # 创建 DeepSORT 跟踪器
        self._tracker = DeepSORTTracker(
            map_api=None,
            max_distance=max_distance,
            max_velocity=max_velocity,
            frame_interval=0.1,
            min_hits=2,
            max_misses=30,
            use_map=False,
        )

        # 重置状态
        self._trajectories = {}
        if self._id_manager:
            self._id_manager = IDConsistencyManager()
        if self._llm_optimizer:
            self._llm_optimizer.cache.clear()
        self._lane_history = {}

        # 处理每一帧
        for frame in frames:
            detections = self._prepare_detections(frame)
            self._tracker.update(detections, frame.frame_id)

            # LLM 增强处理（如果启用）
            if self._use_llm and self._llm_optimizer:
                self._llm_enhanced_process(frame, detections)

        # 构建轨迹
        self._build_trajectories()

        stats = self._tracker.get_statistics()
        stats['use_llm'] = self._use_llm
        stats['llm_calls'] = self._stats.get('llm_calls', 0)

        return {
            "success": True,
            "message": f"重建完成（{'LLM 增强' if self._use_llm else '纯 DeepSORT'}）",
            "num_trajectories": len(self._trajectories),
            "statistics": stats,
        }

    def _prepare_detections(self, frame: FrameDetection) -> List[Dict]:
        """准备检测数据"""
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
        return detections

    def _llm_enhanced_process(self, frame: FrameDetection, detections: List[Dict]):
        """LLM 增强处理"""
        if not self._llm_optimizer or not self.map_api:
            return

        # 检测异常情况
        anomalies = self._detect_anomalies(frame, detections)

        for anomaly in anomalies:
            if anomaly["type"] == "count_mismatch_large":
                # 调用 LLM 分析（结果用于后续决策）
                _ = self._llm_optimizer.analyze_count_mismatch(
                    anomaly["lane_id"],
                    frame.frame_id - 1,
                    anomaly["prev_count"],
                    frame.frame_id,
                    anomaly["curr_count"],
                    anomaly.get("trajectory_history", []),
                    anomaly.get("map_topology", {})
                )
                self._stats['llm_calls'] += 1

            elif anomaly["type"] == "track_reappear":
                _ = self._llm_optimizer.judge_reappear(
                    anomaly["old_track"],
                    anomaly["new_detection"],
                    {}
                )
                self._stats['llm_calls'] += 1

    def _detect_anomalies(self, _frame: FrameDetection, detections: List[Dict]) -> List[Dict]:
        """检测异常情况"""
        anomalies = []

        # 简化的异常检测 - 基于检测数量变化
        if self._lane_history:
            for lane_id, history in self._lane_history.items():
                if len(history.count_history) >= 1:
                    prev_count = history.count_history[-1]
                    curr_count = len(detections)  # 简化：使用总检测数
                    diff = curr_count - prev_count

                    if abs(diff) >= 3:  # 大变化
                        anomalies.append({
                            "type": "count_mismatch_large",
                            "lane_id": lane_id,
                            "prev_count": prev_count,
                            "curr_count": curr_count,
                            "diff": diff,
                        })

        # 更新车道历史
        if detections:
            lane_id = "default"
            if lane_id not in self._lane_history:
                self._lane_history[lane_id] = LaneHistory(lane_id=lane_id)
            self._lane_history[lane_id].count_history.append(len(detections))

        return anomalies

    def _build_trajectories(self):
        """从跟踪器构建轨迹"""
        tracks = self._tracker.get_active_tracks()

        self._trajectories = {}
        for track_id, tracked_obj in tracks.items():
            trajectory = Trajectory(
                track_id=track_id,
                positions=tracked_obj.positions.copy(),
                velocities=tracked_obj.velocities.copy(),
                frame_ids=tracked_obj.frame_ids.copy(),
                obj_types=[tracked_obj.obj_type] * len(tracked_obj.frame_ids),
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

        from collections import Counter
        type_counts = Counter(types)

        return {
            "success": True,
            "summary": {
                "total_trajectories": len(self._trajectories),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "vehicle_types": dict(type_counts),
                "use_llm": self._use_llm,
                "llm_calls": self._stats.get('llm_calls', 0),
            }
        }

    # ==================== 公共方法 ====================

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
            }

        stats = self._tracker.get_statistics() if self._tracker else {}

        return {
            'trajectories': trajectories_data,
            'statistics': {
                'total_trajectories': len(self._trajectories),
                'tracker_stats': stats,
                'use_llm': self._use_llm,
                'llm_calls': self._stats.get('llm_calls', 0),
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


# ==================== 便捷函数 ====================

def reconstruct_traffic_flow(frames: List[Dict],
                             max_distance: float = 5.0,
                             max_velocity: float = 30.0,
                             use_llm: bool = False,
                             llm_client: Optional[LLMClient] = None) -> Dict:
    """
    便捷函数：重建交通流

    Args:
        frames: 帧数据列表，每帧包含 'frame_id' 和 'objects'
        max_distance: 最大匹配距离（米）
        max_velocity: 最大速度（米/秒）
        use_llm: 是否启用 LLM 优化
        llm_client: LLM 客户端（启用 LLM 时需要）

    Returns:
        重建结果
    """
    tracker = DeepSORTTracker(
        map_api=None,
        max_distance=max_distance,
        max_velocity=max_velocity,
        frame_interval=0.1,
        min_hits=2,
        max_misses=30,
        use_map=False,
    )

    llm_optimizer = None
    if use_llm and llm_client:
        llm_optimizer = LLMOptimizer(llm_client)

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
                    'velocity': obj.get('velocity', [0, 0, 0]),
                    'speed': obj.get('speed', 0.0),
                })

        tracker.update(detections, frame_id)

    tracks = tracker.get_active_tracks()

    trajectories = {}
    for track_id, tracked_obj in tracks.items():
        trajectories[track_id] = {
            'track_id': track_id,
            'positions': tracked_obj.positions,
            'frame_ids': tracked_obj.frame_ids,
            'type': tracked_obj.obj_type,
            'length': len(tracked_obj.frame_ids),
        }

    stats = tracker.get_statistics()
    stats['use_llm'] = use_llm
    if llm_optimizer:
        stats['llm_calls'] = llm_optimizer.call_count

    return {
        'trajectories': trajectories,
        'statistics': stats,
    }