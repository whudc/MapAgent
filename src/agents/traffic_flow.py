"""
交通流重建 Agent - 支持纯 DeepSORT 和 LLM 混合优化两种模式

核心设计：
1. 纯 DeepSORT 模式：使用卡尔曼滤波 + 级联匹配进行多目标跟踪
2. LLM 混合模式：规则层处理正常匹配，LLM 层处理疑难场景
3. 地图约束：利用车道信息约束车辆匹配，保证车道内车辆数量守恒

架构：
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  规则层      │ →  │  LLM 层      │ →  │  后处理层    │
│  (快速匹配)  │    │  (疑难推理)  │    │  (ID 一致性)  │
│  + 车道约束  │    │  (数量分析)  │    │  (轨迹平滑)  │
└──────────────┘    └──────────────┘    └──────────────┘

优化原则：
1. 前后帧同一车道内的车辆数量应该基本相同
2. 数量变化时需要重点分析（miss/false/enter/exit）
3. 被遮挡的目标通过 LLM+ 地图拓扑进行推理
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from pathlib import Path
import numpy as np

from agents.base import BaseAgent, AgentContext
from agents.deepsort_tracker import DeepSORTTracker, TrackedObject, Detection, TrackState
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


@dataclass
class LaneConstrainedTrack:
    """车道约束的轨迹"""
    track_id: int
    lane_id: Optional[str] = None
    positions: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    predicted_pos: Optional[List[float]] = None
    lost_count: int = 0
    last_update_frame: int = 0
    confidence: float = 1.0
    llm_verified: bool = False  # 是否经过 LLM 验证


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
    LLM 优化器 - 增强版

    仅在关键决策点调用 LLM，避免频繁调用
    核心优化：
    1. 车道数量守恒分析
    2. 遮挡目标推理
    3. ID 冲突解决
    4. ID 跳变分析（新增）
    """

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 progress_callback: Optional[callable] = None):
        self.llm_client = llm_client
        self.cache: Dict[str, Any] = {}
        self.call_count = 0
        # 车道级统计
        self.lane_stats: Dict[str, Dict] = {}
        # 进度回调（用于推送推理过程到前端）
        self.progress_callback = progress_callback
        # ID 跳变分析缓存
        self._id_jump_cache = {}

    def _notify_progress(self, event_type: str, data: Dict):
        """发送进度通知"""
        if self.progress_callback:
            self.progress_callback(event_type, data)

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
            "occlusion_analysis",  # 遮挡分析
        ]
        return situation in llm_situations

    def analyze_lane_count_conservation(self,
                                        lane_id: str,
                                        prev_frame_id: int,
                                        curr_frame_id: int,
                                        prev_tracks: List[LaneConstrainedTrack],
                                        curr_detections: List[Dict],
                                        map_api: Optional[Any] = None) -> Dict:
        """
        分析车道数量守恒

        原则：前后帧同一车道内的车辆数量应该基本相同
        差异超过阈值时才需要 LLM 分析
        """
        cache_key = f"lane_conservation_{lane_id}_{prev_frame_id}_{curr_frame_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prev_count = len(prev_tracks)
        curr_count = len(curr_detections)
        diff = curr_count - prev_count

        # 规则层能处理的小变化
        if abs(diff) <= 1:
            if diff < 0:
                return {"cause": "exit", "confidence": 0.7, "affected_ids": [],
                        "reasoning": "目标正常驶离车道", "action": "keep_tracks"}
            else:
                return {"cause": "enter", "confidence": 0.6, "affected_ids": [],
                        "reasoning": "新目标驶入车道", "action": "create_new"}

        # 需要 LLM 分析的大变化
        if not self.llm_client:
            return self._rule_based_lane_analysis(lane_id, prev_tracks, curr_detections, diff)

        # 通知前端开始车道分析
        self._notify_progress("lane_analysis_start", {
            "lane_id": lane_id,
            "prev_count": prev_count,
            "curr_count": curr_count,
            "diff": diff,
            "frame_range": [prev_frame_id, curr_frame_id]
        })

        prompt = self._build_lane_conservation_prompt(
            lane_id, prev_frame_id, curr_frame_id,
            prev_tracks, curr_detections, map_api
        )

        self.call_count += 1
        try:
            # 通知前端 LLM 思考中
            self._notify_progress("llm_thinking", {
                "analysis_type": "lane_count_conservation",
                "lane_id": lane_id,
                "prompt_preview": prompt[:200] + "..."
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # 通知前端分析结果
            self._notify_progress("lane_analysis_result", {
                "lane_id": lane_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            self._notify_progress("lane_analysis_error", {
                "lane_id": lane_id,
                "error": error_detail
            })
            result = self._rule_based_lane_analysis(lane_id, prev_tracks, curr_detections, diff)

        self.cache[cache_key] = result
        return result

    def _build_lane_conservation_prompt(self, lane_id: str,
                                         prev_frame_id: int,
                                         curr_frame_id: int,
                                         prev_tracks: List[LaneConstrainedTrack],
                                         curr_detections: List[Dict],
                                         map_api: Optional[Any]) -> str:
        """构建车道守恒分析 prompt"""
        prev_count = len(prev_tracks)
        curr_count = len(curr_detections)
        diff = curr_count - prev_count

        prompt = f"""【车道数量守恒分析】

车道 ID: {lane_id}
帧范围：{prev_frame_id} → {curr_frame_id}
前一帧车辆数：{prev_count}
当前帧检测数：{curr_count}
数量变化：{diff:+d} ({'减少' if diff < 0 else '增加'})

【前一帧轨迹信息】"""

        for track in prev_tracks[:10]:  # 最多显示 10 条
            pos = track.positions[-1] if track.positions else [0, 0, 0]
            pred_pos = track.predicted_pos or pos
            prompt += f"""
- ID {track.track_id}:
  最后位置：({pos[0]:.1f}, {pos[1]:.1f})
  预测位置：({pred_pos[0]:.1f}, {pred_pos[1]:.1f})
  丢失帧数：{track.lost_count}
  置信度：{track.confidence:.2f}"""

        prompt += f"""

【当前帧检测信息】"""
        for i, det in enumerate(curr_detections[:10]):  # 最多显示 10 个
            pos = det.get('location', [0, 0, 0])
            prompt += f"""
- 检测{i}: 位置 ({pos[0]:.1f}, {pos[1]:.1f}), 类型={det.get('type', 'Unknown')}"""

        # 添加地图拓扑信息
        if map_api:
            try:
                lane_info = map_api.get_lane_info(lane_id)
                if lane_info:
                    prompt += f"""

【地图拓扑信息】
- 车道类型：{lane_info.get('type', 'unknown')}
- 前车道路线：{lane_info.get('predecessor_ids', [])}
- 后车道路线：{lane_info.get('successor_ids', [])}"""
            except:
                pass

        prompt += """

【分析任务】
请分析数量变化的原因，选择最可能的情况：

1. **miss (漏检)** - 车辆存在但未被检测到（遮挡、传感器失效）
2. **false (误检)** - 当前帧有虚假检测
3. **exit (驶离)** - 车辆正常驶出当前车道
4. **enter (驶入)** - 新车辆从其他车道驶入

对于减少的车辆，请判断是否是：
- 被其他车辆遮挡
- 驶出车道范围
- 检测器漏检

对于增加的车辆，请判断是否是：
- 从相邻车道变道而来
- 新进入检测范围
- 虚假检测

返回 JSON 格式：
{
    "cause": "miss|false|exit|enter|mixed",
    "confidence": 0.0-1.0,
    "affected_track_ids": [1, 2, 3],  // 受影响的轨迹 ID
    "action": "keep|remove|merge|interpolate",
    "reasoning": "简要推理说明"
}
"""
        return prompt

    def _rule_based_lane_analysis(self, lane_id: str,
                                   prev_tracks: List[LaneConstrainedTrack],
                                   curr_detections: List[Dict],
                                   diff: int) -> Dict:
        """基于规则的车道分析"""
        if diff < -1:  # 数量减少
            # 检查是否有轨迹预测位置在检测范围内但未匹配
            unmatched_tracks = [t for t in prev_tracks if t.lost_count < 5]
            if len(unmatched_tracks) > 0:
                return {
                    "cause": "miss",
                    "confidence": 0.6,
                    "affected_track_ids": [t.track_id for t in unmatched_tracks],
                    "action": "interpolate",  # 插值保持
                    "reasoning": f"{len(unmatched_tracks)} 条轨迹预测存在但检测丢失，可能遮挡"
                }
            return {"cause": "exit", "confidence": 0.7, "affected_track_ids": [],
                    "action": "keep_tracks", "reasoning": "目标正常驶离"}
        else:  # 数量增加
            return {"cause": "enter", "confidence": 0.6, "affected_track_ids": [],
                    "action": "create_new", "reasoning": "新目标驶入"}

    def analyze_occlusion(self,
                          lost_track: LaneConstrainedTrack,
                          nearby_tracks: List[LaneConstrainedTrack],
                          curr_detections: List[Dict],
                          map_api: Optional[Any] = None) -> Dict:
        """
        分析遮挡情况

        当轨迹丢失时，判断是否被其他车辆遮挡
        """
        cache_key = f"occlusion_{lost_track.track_id}_{lost_track.last_update_frame}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_occlusion(lost_track, nearby_tracks, curr_detections)

        # 通知前端开始遮挡分析
        self._notify_progress("occlusion_analysis_start", {
            "track_id": lost_track.track_id,
            "lost_frames": lost_track.lost_count,
            "lane_id": lost_track.lane_id,
            "nearby_count": len(nearby_tracks)
        })

        prompt = f"""【遮挡分析】

丢失轨迹信息：
- ID: {lost_track.track_id}
- 最后位置：{lost_track.positions[-1] if lost_track.positions else 'N/A'}
- 预测位置：{lost_track.predicted_pos or 'N/A'}
- 丢失帧数：{lost_track.lost_count}
- 车道：{lost_track.lane_id}

【附近轨迹】"""
        for track in nearby_tracks[:5]:
            pos = track.positions[-1] if track.positions else [0, 0, 0]
            prompt += f"""
- ID {track.track_id}: 位置 ({pos[0]:.1f}, {pos[1]:.1f}), 置信度 {track.confidence:.2f}"""

        prompt += """

【当前帧检测】"""
        for det in curr_detections[:5]:
            pos = det.get('location', [0, 0, 0])
            prompt += f"""
- 位置 ({pos[0]:.1f}, {pos[1]:.1f}), 类型={det.get('type', 'Unknown')}"""

        prompt += """

请分析：
1. 丢失的轨迹是否被附近车辆遮挡？
2. 如果是遮挡，预测何时重新出现？
3. 建议的处理方式是什么？

返回 JSON:
{
    "is_occluded": true/false,
    "occluder_id": 轨迹 ID 或 null,
    "confidence": 0.0-1.0,
    "predicted_reappear_frames": 1-10,
    "action": "keep|interpolate|remove",
    "reasoning": "推理说明"
}
"""
        self.call_count += 1
        try:
            # 通知前端 LLM 思考中
            self._notify_progress("llm_thinking", {
                "analysis_type": "occlusion",
                "track_id": lost_track.track_id,
                "prompt_preview": prompt[:200] + "..."
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # 通知前端分析结果
            self._notify_progress("occlusion_analysis_result", {
                "track_id": lost_track.track_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            self._notify_progress("occlusion_analysis_error", {
                "track_id": lost_track.track_id,
                "error": error_detail
            })
            result = self._rule_based_occlusion(lost_track, nearby_tracks, curr_detections)

        self.cache[cache_key] = result
        return result

    def _rule_based_occlusion(self, lost_track: LaneConstrainedTrack,
                               nearby_tracks: List[LaneConstrainedTrack],
                               curr_detections: List[Dict]) -> Dict:
        """基于规则的遮挡分析"""
        if not lost_track.predicted_pos:
            return {"is_occluded": False, "action": "remove", "confidence": 0.5}

        pred_pos = np.array(lost_track.predicted_pos[:2])

        # 检查是否有 nearby tracks 在预测位置附近
        for track in nearby_tracks:
            if track.track_id == lost_track.track_id:
                continue
            if not track.positions:
                continue
            track_pos = np.array(track.positions[-1][:2])
            dist = np.linalg.norm(pred_pos - track_pos)

            # 如果距离很近（< 5 米），可能是遮挡
            if dist < 5.0:
                return {
                    "is_occluded": True,
                    "occluder_id": track.track_id,
                    "confidence": 0.7,
                    "predicted_reappear_frames": 3,
                    "action": "keep",
                    "reasoning": f"预测位置被轨迹{track.track_id}遮挡"
                }

        return {
            "is_occluded": False,
            "confidence": 0.5,
            "action": "interpolate",
            "reasoning": "未发现明显遮挡物"
        }

    def analyze_id_jumping(self,
                           track_id: int,
                           track_history: List[Dict],
                           recent_detections: List[Dict],
                           frame_id: int,
                           map_api: Optional[Any] = None) -> Dict:
        """
        分析 ID 跳变问题

        当轨迹出现 ID 跳变时，使用 LLM 分析是否应该保持原 ID

        Args:
            track_id: 轨迹 ID
            track_history: 轨迹历史（最近 5-10 帧）
            recent_detections: 最近检测
            frame_id: 当前帧
            map_api: 地图 API

        Returns:
            分析结果
        """
        cache_key = f"id_jump_{track_id}_{frame_id}"
        if cache_key in self._id_jump_cache:
            return self._id_jump_cache[cache_key]

        if not self.llm_client:
            return self._rule_based_id_judge(track_history, recent_detections)

        # 通知前端开始 ID 分析
        self._notify_progress("id_analysis_start", {
            "track_id": track_id,
            "frame_id": frame_id,
            "history_length": len(track_history)
        })

        # 构建 prompt
        prompt = f"""【ID 一致性分析】

轨迹 ID: {track_id}
当前帧：{frame_id}

【轨迹历史】（最近 {len(track_history)} 帧）"""
        for i, pos in enumerate(track_history[-5:]):
            prompt += f"""
- 帧 {pos.get('frame_id', '?')}: 位置 ({pos.get('pos', [0,0])[:2]}), 置信度 {pos.get('confidence', 1.0)}"""

        prompt += f"""

【当前帧检测】"""
        for i, det in enumerate(recent_detections[:5]):
            prompt += f"""
- 检测{i}: 位置 ({det.get('location', [0,0])[:2]}), 类型={det.get('type', 'Unknown')}"""

        # 添加地图信息
        if map_api:
            try:
                pos = track_history[-1].get('pos', [0, 0]) if track_history else [0, 0]
                lane_info = map_api.find_nearest_lane(pos[:2])
                if lane_info:
                    prompt += f"""

【地图信息】
- 所在车道：{lane_info.get('lane_id', 'unknown')}
- 车道类型：{lane_info.get('type', 'unknown')}"""
            except:
                pass

        prompt += """

【分析任务】
判断当前帧的检测是否应该继承该轨迹的 ID：

1. **保持 ID** - 检测与轨迹历史连续，应保持原 ID
2. **新目标** - 检测是新的车辆，应分配新 ID
3. **ID 合并** - 多个检测实际是同一车辆，应合并 ID

返回 JSON:
{
    "decision": "keep_id|new_target|merge_ids",
    "confidence": 0.0-1.0,
    "target_detection_idx": 0,  // 推荐的检测索引
    "reasoning": "推理说明"
}
"""
        self.call_count += 1
        try:
            self._notify_progress("llm_thinking", {
                "analysis_type": "id_consistency",
                "track_id": track_id
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            self._notify_progress("id_analysis_result", {
                "track_id": track_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            self._notify_progress("id_analysis_error", {
                "track_id": track_id,
                "error": error_detail
            })
            result = self._rule_based_id_judge(track_history, recent_detections)

        self._id_jump_cache[cache_key] = result
        return result

    def _rule_based_id_judge(self, track_history: List[Dict],
                              recent_detections: List[Dict]) -> Dict:
        """基于规则的 ID 判断"""
        if not track_history or not recent_detections:
            return {"decision": "new_target", "confidence": 0.5, "reasoning": "数据不足"}

        # 获取轨迹最后位置
        last_pos = np.array(track_history[-1].get('pos', [0, 0])[:2])

        # 找到最近的检测
        min_dist = float('inf')
        best_det_idx = -1
        for i, det in enumerate(recent_detections):
            det_pos = np.array(det.get('location', [0, 0])[:2])
            dist = np.linalg.norm(last_pos - det_pos)
            if dist < min_dist:
                min_dist = dist
                best_det_idx = i

        # 距离小于阈值，认为应保持 ID
        if min_dist < 3.0:
            return {
                "decision": "keep_id",
                "confidence": min(0.9, 1.0 - min_dist / 10.0),
                "target_detection_idx": best_det_idx,
                "reasoning": f"检测距离轨迹末端 {min_dist:.2f} 米，应继承 ID"
            }
        else:
            return {
                "decision": "new_target",
                "confidence": min(0.9, 1.0 - 3.0 / min_dist),
                "reasoning": f"最近检测距离 {min_dist:.2f} 米，可能是新目标"
            }

    def judge_reappear(self, old_track: Dict, new_detection: Dict,
                       map_topology: Dict) -> Dict:
        """判断重新出现的目标是否是同一轨迹"""
        cache_key = f"reappear_{old_track.get('id')}_{new_detection.get('pos')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_reappear(old_track, new_detection)

        # 检查地图拓扑
        old_lane = old_track.get('lane_id')
        new_lane = new_detection.get('lane_id')
        lane_consistent = old_lane == new_lane

        # 检查车道连接关系
        if map_topology and old_lane:
            successors = map_topology.get('successors', {}).get(old_lane, [])
            lane_consistent = lane_consistent or (new_lane in successors)

        prompt = f"""判断重新出现的目标是否是同一轨迹：

旧轨迹信息：
- ID: {old_track.get('id')}
- 最后位置：{old_track.get('last_pos')}
- 最后车道：{old_track.get('lane_id')}
- 丢失帧数：{old_track.get('lost_frames')}
- 预测位置：{old_track.get('predicted_pos')}

新检测信息：
- 位置：{new_detection.get('pos')}
- 车道：{new_detection.get('lane_id')}
- 类型：{new_detection.get('type')}

车道一致性：{"是" if lane_consistent else "否"}
"""
        if map_topology and old_lane:
            prompt += f"- {old_lane} 的后继车道：{map_topology.get('successors', {}).get(old_lane, [])}\n"

        prompt += """
返回 JSON：
{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "推理说明"
}
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
        # LLM 推理过程回调
        self._llm_progress_callback = None

        if use_llm and context.llm_client:
            self._id_manager = IDConsistencyManager()
            self._llm_optimizer = LLMOptimizer(context.llm_client, self._notify_llm_progress)
            self.name = "traffic_flow_llm_agent"

        # 统计
        self._stats = {
            'total_frames': 0,
            'llm_calls': 0,
            'use_llm': use_llm,
        }

    def set_llm_progress_callback(self, callback: callable):
        """设置 LLM 进度回调"""
        self._llm_progress_callback = callback

    def _notify_llm_progress(self, event_type: str, data: Dict):
        """通知 LLM 进度"""
        if self._llm_progress_callback:
            self._llm_progress_callback(event_type, data)

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
                           end_frame: Optional[int] = None,
                           use_interpolation: bool = True) -> List[Dict]:
        """
        构建帧数据（用于 UI 可视化）

        使用 DeepSORT 跟踪器的 track_id 作为车辆 ID，保证 ID 连续性
        支持插值帧，减少闪烁

        Args:
            start_frame: 起始帧 ID
            end_frame: 结束帧 ID
            use_interpolation: 是否使用插值（减少闪烁）

        Returns:
            帧数据列表
        """
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

            # 获取该帧的检测数据（不使用 DetectionLoader 的跟踪 ID）
            frame_det = self._loader.load_frame(frame_id, use_tracking=False)
            if not frame_det:
                continue

            # 构建 DeepSORT 轨迹的位置到 track_id 的映射
            track_id_map = {}  # (x, y) -> track_id
            active_tracks = {}  # track_id -> (position, is_interpolated)

            if self._tracker:
                for track_id, track in self._tracker.tracks.items():
                    if track.state.name != 'DELETED' and track.frame_ids:
                        # 找到该轨迹在当前帧的位置
                        for i, fid in enumerate(track.frame_ids):
                            if fid == frame_id:
                                pos = tuple(track.positions[i][:2])
                                track_id_map[pos] = track_id
                                active_tracks[track_id] = (list(pos), False)

            # 使用插值轨迹填充丢失的目标（减少闪烁）
            if use_interpolation and hasattr(self._tracker, '_interpolated_tracks'):
                for track_id, interp_data in self._tracker._interpolated_tracks.items():
                    for interp in interp_data:
                        if interp['frame_id'] == frame_id and interp.get('is_interpolated'):
                            if track_id not in active_tracks:
                                pos = tuple(interp['position'][:2])
                                active_tracks[track_id] = (list(pos), True)

            # 构建车辆列表
            vehicles = []

            # 1. 首先添加有检测匹配的车辆
            matched_positions = set()
            for obj in frame_det.objects:
                obj_pos = tuple(obj.location[:2])

                vehicle_id = None

                # 尝试精确位置匹配
                if obj_pos in track_id_map:
                    vehicle_id = track_id_map[obj_pos]
                    matched_positions.add(obj_pos)
                else:
                    # 尝试最近邻匹配（在 1 米范围内）
                    for track_pos, track_id in track_id_map.items():
                        dist = ((obj_pos[0] - track_pos[0]) ** 2 +
                               (obj_pos[1] - track_pos[1]) ** 2) ** 0.5
                        if dist < 1.0:
                            vehicle_id = track_id
                            matched_positions.add(track_pos)
                            break

                # 如果没有匹配的轨迹，使用原始 tracking_id 或检测 ID
                if vehicle_id is None:
                    vehicle_id = obj.tracking_id if obj.tracking_id > 0 else obj.id

                vehicles.append({
                    'vehicle_id': vehicle_id,
                    'vehicle_type': obj.type,
                    'position': list(obj.location),
                    'heading': obj.heading,
                    'speed': obj.speed,
                    'is_interpolated': False,
                })

            # 2. 添加插值车辆（轨迹存在但检测丢失）
            for track_id, (pos, is_interp) in active_tracks.items():
                # 检查该位置是否已有车辆
                has_vehicle = False
                for v in vehicles:
                    v_pos = tuple(v['position'][:2])
                    if np.linalg.norm(np.array(v_pos) - np.array(pos)) < 0.5:
                        has_vehicle = True
                        break

                if not has_vehicle and is_interp:
                    # 添加插值车辆
                    track = self._tracker.tracks.get(track_id)
                    vehicles.append({
                        'vehicle_id': track_id,
                        'vehicle_type': track.obj_type if track else 'Unknown',
                        'position': list(pos),
                        'heading': 0.0,
                        'speed': 0.0,
                        'is_interpolated': True,
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
                                   max_velocity: float = 30.0,
                                   use_lane_constraint: bool = True,
                                   use_interpolation: bool = True) -> Dict[str, Any]:
        """
        重建交通流 - 增强版

        Args:
            start_frame: 起始帧 ID
            end_frame: 结束帧 ID
            max_distance: 最大匹配距离（米）
            max_velocity: 最大速度（米/秒）
            use_lane_constraint: 是否使用车道约束（默认 True）
            use_interpolation: 是否使用轨迹插值（默认 True）

        Returns:
            重建结果
        """
        if not self._loader:
            return {"success": False, "error": "请先加载检测结果"}

        # 加载帧数据
        frames = self._loader.load_frames(start_frame, end_frame)
        if not frames:
            return {"success": False, "error": "未加载到帧数据"}

        # 创建增强版跟踪器（车道感知 + 插值）
        from .lane_aware_tracker import LaneAwareTracker

        self._tracker = LaneAwareTracker(
            map_api=self.map_api if use_lane_constraint else None,
            max_distance=max_distance,
            max_velocity=max_velocity,
            frame_interval=0.1,
            min_hits=2,
            max_misses=30,
            use_map=use_lane_constraint,
            lane_weight=0.3,
            max_lane_distance=3.0,
            interpolation_enabled=use_interpolation,
            max_interpolation_frames=5,
        )

        # 重置状态
        self._trajectories = {}
        if self._id_manager:
            self._id_manager = IDConsistencyManager()
        if self._llm_optimizer:
            self._llm_optimizer.cache.clear()
        self._lane_history = {}

        # 上一帧的车道统计（用于数量守恒分析）
        prev_lane_tracks: Dict[str, List] = {}

        # 处理每一帧
        for frame in frames:
            detections = self._prepare_detections(frame)
            self._tracker.update(detections, frame.frame_id)

            # LLM 增强处理（如果启用）
            if self._use_llm and self._llm_optimizer and self.map_api:
                self._llm_enhanced_process_with_lanes(frame, detections, prev_lane_tracks)
            elif self._use_llm:
                # 调试输出
                if not self._llm_optimizer:
                    print(f"[DEBUG] Frame {frame.frame_id}: _llm_optimizer is None")
                if not self.map_api:
                    print(f"[DEBUG] Frame {frame.frame_id}: map_api is None")

            # 更新车道统计
            if self._use_llm:
                prev_lane_tracks = self._get_lane_constrained_tracks()

        # 构建轨迹（包含插值帧）
        self._build_trajectories_with_interpolation()

        stats = self._tracker.get_statistics()
        stats['use_llm'] = self._use_llm
        stats['llm_calls'] = self._stats.get('llm_calls', 0)
        stats['use_lane_constraint'] = use_lane_constraint
        stats['use_interpolation'] = use_interpolation
        # 调试信息
        stats['_debug'] = {
            'self._use_llm': self._use_llm,
            'self._llm_optimizer': self._llm_optimizer is not None,
            'self.map_api': self.map_api is not None,
        }

        return {
            "success": True,
            "message": f"重建完成（{'LLM 增强 + 车道约束' if self._use_llm else '纯 DeepSORT'}）",
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

    def _get_lane_constrained_tracks(self) -> Dict[str, List]:
        """获取按车道分组的轨迹"""
        lane_tracks: Dict[str, List] = {}

        if not self._tracker:
            return lane_tracks

        for track_id, track in self._tracker.tracks.items():
            if track.state == TrackState.DELETED:
                continue

            # 获取轨迹的车道 ID
            lane_id = getattr(self._tracker, '_track_lanes', {}).get(track_id, 'unknown')

            if lane_id not in lane_tracks:
                lane_tracks[lane_id] = []

            lane_tracks[lane_id].append({
                'track_id': track_id,
                'last_position': track.last_position,
                'predicted_position': track.predicted_location().tolist() if track.kf_mean is not None else None,
                'lost_count': track.time_since_update,
                'confidence': track.confidence if hasattr(track, 'confidence') else 1.0,
            })

        return lane_tracks

    def _llm_enhanced_process_with_lanes(self, frame: FrameDetection,
                                          detections: List[Dict],
                                          prev_lane_tracks: Dict[str, List]):
        """
        LLM 增强处理 - 车道感知版本

        核心优化：
        1. 车道数量守恒分析
        2. 遮挡目标推理
        3. ID 一致性分析（防止 ID 跳变）
        """
        print(f"[DEBUG] _llm_enhanced_process_with_lanes called for frame {frame.frame_id}: llm_optimizer={self._llm_optimizer is not None}, map_api={self.map_api is not None}, detections={len(detections)}")

        if not self._llm_optimizer or not self.map_api:
            return

        # 获取当前车道统计
        curr_lane_tracks = self._get_lane_constrained_tracks()

        # 分析每条车道的数量变化
        for lane_id in set(prev_lane_tracks.keys()) | set(curr_lane_tracks.keys()):
            prev_tracks = prev_lane_tracks.get(lane_id, [])
            curr_tracks = curr_lane_tracks.get(lane_id, [])

            prev_count = len(prev_tracks)
            curr_count = len(curr_tracks)
            diff = curr_count - prev_count

            # 小变化用规则处理（变化量 <=1 时不需要 LLM）
            if abs(diff) <= 1:
                continue

            # 大变化调用 LLM 分析
            llm_result = self._llm_optimizer.analyze_lane_count_conservation(
                lane_id=lane_id,
                prev_frame_id=frame.frame_id - 1,
                curr_frame_id=frame.frame_id,
                prev_tracks=[self._dict_to_lane_track(t) for t in prev_tracks],
                curr_detections=[d for d in detections if self._get_detection_lane(d) == lane_id],
                map_api=self.map_api
            )

            self._stats['llm_calls'] += 1

            # 根据 LLM 建议处理
            if llm_result.get('action') == 'interpolate':
                # 插值丢失的轨迹
                affected_ids = llm_result.get('affected_track_ids', [])
                for track_id in affected_ids:
                    self._interpolate_track(track_id, frame.frame_id)

        # 遮挡分析 - 对丢失的轨迹进行分析
        # 条件：轨迹已确认且丢失帧数在 1-10 之间
        for track_id, track in self._tracker.tracks.items():
            if track.state != TrackState.DELETED and 0 < track.time_since_update <= 10:
                # 轨迹丢失，检查是否被遮挡
                nearby_tracks = self._get_nearby_tracks(track, radius=5.0)

                llm_result = self._llm_optimizer.analyze_occlusion(
                    lost_track=self._track_to_lane_track(track),
                    nearby_tracks=[self._track_to_lane_track(t) for t in nearby_tracks],
                    curr_detections=detections,
                    map_api=self.map_api
                )

                self._stats['llm_calls'] += 1

                if llm_result.get('is_occluded') and llm_result.get('action') == 'keep':
                    # 保持被遮挡的轨迹
                    track.confidence = 0.5  # 降低置信度但不删除

        # 【新增】ID 一致性分析 - 防止 ID 跳变
        # 对每条活跃的轨迹，强制每帧进行 ID 一致性分析
        print(f"[DEBUG] Starting ID consistency analysis for frame {frame.frame_id}: {len(self._tracker.tracks)} tracks")
        for track_id, track in self._tracker.tracks.items():
            if track.state == TrackState.DELETED:
                continue
            if not track.positions or len(track.positions) < 2:
                continue  # 轨迹太短，不分析

            # 构建轨迹历史
            track_history = []
            for i, (pos, frame_id_hist) in enumerate(zip(track.positions[-5:], track.frame_ids[-5:])):
                track_history.append({
                    'frame_id': frame_id_hist,
                    'pos': pos,
                    'confidence': getattr(track, 'confidence', 1.0)
                })

            # 找到轨迹当前位置附近的检测
            last_pos = track.positions[-1][:2] if track.positions else [0, 0]
            nearby_detections = []
            for det in detections:
                det_pos = det.get('location', [0, 0])[:2]
                dist = np.linalg.norm(np.array(last_pos) - np.array(det_pos))
                if dist < 5.0:  # 5 米范围内的检测
                    nearby_detections.append(det)

            print(f"[DEBUG] Track {track_id}: analyzing with {len(nearby_detections)} nearby detections")

            # 强制每帧调用 LLM 分析 ID 一致性
            llm_result = self._llm_optimizer.analyze_id_jumping(
                track_id=track_id,
                track_history=track_history,
                recent_detections=nearby_detections,
                frame_id=frame.frame_id,
                map_api=self.map_api
            )
            self._stats['llm_calls'] += 1
            print(f"[DEBUG] Track {track_id}: LLM call completed, llm_calls={self._stats['llm_calls']}")

            # 根据 LLM 建议处理
            if llm_result.get('decision') == 'keep_id':
                det_idx = llm_result.get('target_detection_idx', -1)
                if det_idx >= 0 and det_idx < len(nearby_detections):
                    # 确保轨迹与正确的检测关联
                    pass  # LLM 已确认当前匹配正确

    def _dict_to_lane_track(self, d: Dict) -> LaneConstrainedTrack:
        """将字典转换为车道约束轨迹"""
        return LaneConstrainedTrack(
            track_id=d.get('track_id', 0),
            lane_id='unknown',
            positions=[d.get('last_position', [0, 0, 0])] if d.get('last_position') else [],
            frame_ids=[],
            predicted_pos=d.get('predicted_position'),
            lost_count=d.get('lost_count', 0),
            confidence=d.get('confidence', 1.0)
        )

    def _track_to_lane_track(self, track: TrackedObject) -> LaneConstrainedTrack:
        """将 TrackedObject 转换为 LaneConstrainedTrack"""
        lane_id = getattr(self._tracker, '_track_lanes', {}).get(track.track_id, 'unknown')
        return LaneConstrainedTrack(
            track_id=track.track_id,
            lane_id=lane_id,
            positions=track.positions,
            frame_ids=track.frame_ids,
            predicted_pos=track.predicted_location().tolist() if track.kf_mean is not None else None,
            lost_count=track.time_since_update,
            confidence=getattr(track, 'confidence', 1.0)
        )

    def _get_detection_lane(self, detection: Dict) -> str:
        """获取检测所在的车道 ID"""
        if not self.map_api:
            return 'unknown'

        pos = detection.get('location', [0, 0, 0])[:2]
        try:
            nearest = self.map_api.find_nearest_lane(pos)
            if nearest and nearest.get('distance', float('inf')) < 3.0:
                return nearest.get('lane_id', 'unknown')
        except Exception:
            pass
        return 'unknown'

    def _get_nearby_tracks(self, track: TrackedObject, radius: float = 5.0) -> List[TrackedObject]:
        """获取指定轨迹附近的其他轨迹"""
        if not track.last_position:
            return []

        track_pos = np.array(track.last_position[:2])
        nearby = []

        for other_id, other in self._tracker.tracks.items():
            if other_id == track.track_id:
                continue
            if other.state == TrackState.DELETED:
                continue
            if not other.last_position:
                continue

            other_pos = np.array(other.last_position[:2])
            dist = np.linalg.norm(track_pos - other_pos)

            if dist < radius:
                nearby.append(other)

        return nearby

    def _interpolate_track(self, track_id: int, frame_id: int):
        """对轨迹进行插值"""
        if track_id not in self._tracker.tracks:
            return

        track = self._tracker.tracks[track_id]

        # 使用卡尔曼预测位置进行插值
        pred_pos = track.predicted_location()
        pred_vel = track.predicted_velocity()

        # 添加插值数据
        track.positions.append(pred_pos.tolist())
        track.velocities.append(pred_vel.tolist())
        track.frame_ids.append(frame_id)

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

    def _build_trajectories_with_interpolation(self):
        """从跟踪器构建轨迹（包含插值帧）"""
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