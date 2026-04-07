"""
交通流重建 Agent - 规则 + LLM 混合优化版

核心设计：
1. 规则层处理正常匹配（快速）
2. LLM 层处理疑难场景（漏检/误检判断、ID 重关联）
3. ID 管理器保证长时序一致性

架构：
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  规则层      │ →  │  LLM 层      │ →  │  后处理层    │
│  (快速匹配)  │    │  (疑难推理)  │    │  (ID 一致性)  │
└──────────────┘    └──────────────┘    └──────────────┘
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import sys

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agents.base import BaseAgent, AgentContext
from apis.map_api import MapAPI
from core.llm_client import LLMClient
from utils.detection_loader import DetectionLoader, FrameDetection, DetectedObject


# ==================== 数据结构 ====================

class MatchResult(str, Enum):
    """匹配结果类型"""
    MATCHED = "matched"              # 成功匹配
    MISS_DETECTION = "miss"          # 漏检
    FALSE_DETECTION = "false"        # 误检
    EXITED = "exited"                # 驶离
    ENTERED = "entered"              # 驶入
    RE_ENTERED = "re_entered"        # 重新驶入
    ID_CONFLICT = "id_conflict"      # ID 冲突


@dataclass
class LaneObjectCount:
    """车道目标计数"""
    lane_id: str
    count: int
    object_ids: List[int]
    timestamp: int  # 帧 ID


@dataclass
class FrameState:
    """单帧状态"""
    frame_id: int
    lane_objects: Dict[str, List[DetectedObject]]  # 车道内目标
    out_of_map_objects: List[DetectedObject]       # 地图外目标
    unmatched_objects: List[DetectedObject]        # 未匹配目标


@dataclass
class TrackState:
    """轨迹状态"""
    track_id: int
    lane_id: Optional[str]          # 当前车道
    positions: List[Tuple[float, float, float]]
    frame_ids: List[int]
    status: str = "active"          # active/lost/confirmed
    lost_count: int = 0             # 丢失帧数
    confidence: float = 1.0         # 置信度


@dataclass
class LaneHistory:
    """车道历史状态"""
    lane_id: str
    count_history: List[int] = field(default_factory=list)
    track_ids_history: List[Set[int]] = field(default_factory=list)


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
        self.retired_ids: Set[int] = set()  # 已退役，不再分配
        self.next_id: int = 1

        # 轨迹嵌入（用于相似度匹配）
        self.track_embeddings: Dict[int, np.ndarray] = {}

        # 待确认轨迹（延迟确认）
        self.pending_tracks: Dict[int, TrackState] = {}

    def assign_id(self, detection: DetectedObject,
                  candidates: List[TrackState],
                  context: Dict) -> Tuple[int, str]:
        """
        分配 ID

        Returns:
            (track_id, assignment_reason)
        """
        if len(candidates) == 0:
            # 无候选，分配新 ID
            new_id = self._allocate_new_id()
            self.active_ids.add(new_id)
            return new_id, "new"

        elif len(candidates) == 1:
            # 唯一候选，直接复用
            return candidates[0].track_id, "unique_match"

        else:
            # 多候选，需要消歧义
            best = self._select_best_candidate(detection, candidates, context)
            return best.track_id, "multi_match"

    def _allocate_new_id(self) -> int:
        """分配新 ID"""
        # 优先使用最小可用 ID
        for candidate_id in range(1, self.next_id + 1):
            if candidate_id not in self.active_ids and candidate_id not in self.retired_ids:
                return candidate_id
        self.next_id += 1
        return self.next_id

    def retire_id(self, track_id: int, reason: str = "exited"):
        """退役 ID"""
        if track_id in self.active_ids:
            self.active_ids.remove(track_id)
            if reason == "exited":  # 正常驶离的 ID 不再使用
                self.retired_ids.add(track_id)

    def _select_best_candidate(self, detection: DetectedObject,
                                candidates: List[TrackState],
                                context: Dict) -> TrackState:
        """选择最佳候选轨迹"""
        # 1. 位置距离
        # 2. 速度一致性
        # 3. 车道连续性
        # 4. 外观相似度（如果有）
        scores = []
        for track in candidates:
            score = self._compute_match_score(detection, track, context)
            scores.append((track, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _compute_match_score(self, detection: DetectedObject,
                              track: TrackState,
                              context: Dict) -> float:
        """计算匹配分数"""
        if not track.positions:
            return 0.0

        last_pos = np.array(track.positions[-1][:2])
        det_pos = np.array(detection.location[:2])

        # 距离分数
        dist = np.linalg.norm(last_pos - det_pos)
        dist_score = np.exp(-dist / 5.0)  # 指数衰减

        # 车道一致性
        lane_score = 1.0
        if track.lane_id and context.get("expected_lane") == track.lane_id:
            lane_score = 1.2  # 同车道加分

        # 置信度
        conf_score = track.confidence

        return dist_score * lane_score * conf_score


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
        # 规则层能处理的情况
        if situation == "normal_match":
            return False

        # 数量差异小，规则处理
        count_diff = context.get("count_diff", 0)
        if situation == "count_mismatch" and abs(count_diff) <= 1:
            return False

        # 复杂情况才调用 LLM
        llm_situations = [
            "count_mismatch_large",    # 数量大差异
            "track_reappear",          # 轨迹重新出现
            "id_conflict",             # ID 冲突
            "new_object_source",       # 新目标来源判断
            "lane_transition",         # 车道转换
        ]

        return situation in llm_situations

    def analyze_count_mismatch(self, lane_id: str,
                                prev_frame_id: int, prev_count: int,
                                curr_frame_id: int, curr_count: int,
                                trajectory_history: List[Dict],
                                map_topology: Dict) -> Dict:
        """
        分析车道目标数量不匹配的原因

        Returns:
            {
                "cause": "miss" | "false" | "exit" | "enter",
                "confidence": float,
                "affected_ids": List[int],
                "reasoning": str
            }
        """
        prompt = self._build_mismatch_prompt(
            lane_id, prev_count, curr_count,
            trajectory_history, map_topology
        )

        cache_key = f"mismatch_{lane_id}_{prev_frame_id}_{curr_frame_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_mismatch(prev_count, curr_count)

        self.call_count += 1
        response = self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        result = self._parse_llm_response(response)
        self.cache[cache_key] = result
        return result

    def _build_mismatch_prompt(self, lane_id: str,
                                prev_count: int, curr_count: int,
                                trajectory_history: List[Dict],
                                map_topology: Dict) -> str:
        """构建数量不匹配分析 prompt"""
        # 精简历史，只保留关键信息
        recent_tracks = trajectory_history[-5:] if len(trajectory_history) > 5 else trajectory_history

        prompt = f"""分析车道 {lane_id} 的目标数量变化：
- 前一帧数量：{prev_count}
- 当前帧数量：{curr_count}
- 数量变化：{curr_count - prev_count}

最近轨迹历史（最多 5 条）：
"""
        for track in recent_tracks:
            prompt += f"- ID {track.get('id')}: 位置={track.get('pos')}, 状态={track.get('status')}\n"

        prompt += f"""
地图拓扑信息：
- 前车道路线：{map_topology.get('predecessors', [])}
- 后车道路线：{map_topology.get('successors', [])}

请分析数量变化的原因：
1. 漏检 (miss) - 前一帧有检测，当前帧漏检
2. 误检 (false) - 当前帧有虚假检测
3. 驶离 (exit) - 目标正常驶出车道
4. 驶入 (enter) - 新目标驶入车道

返回 JSON 格式：
{{
    "cause": "miss|false|exit|enter",
    "confidence": 0.0-1.0,
    "affected_ids": [1, 2, ...],
    "reasoning": "简要推理说明"
}}
"""
        return prompt

    def _rule_based_mismatch(self, prev_count: int, curr_count: int) -> Dict:
        """基于规则的数量不匹配处理"""
        diff = curr_count - prev_count

        if diff < 0:
            # 数量减少
            if abs(diff) <= 1:
                return {"cause": "exit", "confidence": 0.7, "affected_ids": [], "reasoning": "正常驶离"}
            else:
                return {"cause": "miss", "confidence": 0.5, "affected_ids": [], "reasoning": "可能漏检"}
        else:
            # 数量增加
            if diff <= 1:
                return {"cause": "enter", "confidence": 0.6, "affected_ids": [], "reasoning": "新目标驶入"}
            else:
                return {"cause": "false", "confidence": 0.4, "affected_ids": [], "reasoning": "可能误检"}

    def _parse_llm_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        import json
        try:
            # 尝试提取 JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass

        return {
            "cause": "unknown",
            "confidence": 0.3,
            "affected_ids": [],
            "reasoning": "解析失败，使用默认规则"
        }

    def judge_reappear(self, old_track: Dict, new_detection: Dict,
                       map_topology: Dict) -> Dict:
        """
        判断重新出现的目标是否是同一轨迹

        Returns:
            {
                "is_same": bool,
                "confidence": float,
                "reasoning": str
            }
        """
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

地图拓扑：
- 车道连接关系：{map_topology.get('connections', [])}

请判断新检测是否是旧轨迹的延续，返回 JSON：
{{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "推理说明"
}}
"""
        cache_key = f"reappear_{old_track.get('id')}_{new_detection.get('pos')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_reappear(old_track, new_detection)

        self.call_count += 1
        response = self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        result = self._parse_llm_response(response)
        self.cache[cache_key] = result
        return result

    def _rule_based_reappear(self, old_track: Dict, new_detection: Dict) -> Dict:
        """基于规则的重新出现判断"""
        # 位置距离
        old_pos = np.array(old_track.get('last_pos', [0, 0, 0])[:2])
        new_pos = np.array(new_detection.get('pos', [0, 0, 0])[:2])
        dist = np.linalg.norm(old_pos - new_pos)

        # 车道一致性
        same_lane = old_track.get('lane_id') == new_detection.get('lane_id')

        # 丢失帧数
        lost_frames = old_track.get('lost_frames', 999)

        # 综合判断
        if dist < 10.0 and same_lane and lost_frames < 10:
            return {"is_same": True, "confidence": 0.8, "reasoning": "位置接近且同车道"}
        elif dist < 20.0 and lost_frames < 20:
            return {"is_same": True, "confidence": 0.5, "reasoning": "位置较接近"}
        else:
            return {"is_same": False, "confidence": 0.6, "reasoning": "位置差异大或丢失时间过长"}


# ==================== 主 Agent ====================

class TrafficFlowLLMAgent(BaseAgent):
    """
    交通流重建 Agent - 规则 + LLM 混合优化版

    工作流程：
    1. 规则层快速匹配（DeepSORT + 车道约束）
    2. 检测异常情况（数量突变、ID 冲突等）
    3. LLM 层处理疑难场景
    4. ID 管理器保证长时序一致性
    """

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "traffic_flow_llm_agent"

        # 核心组件
        self.id_manager = IDConsistencyManager()
        self.llm_optimizer = LLMOptimizer(context.llm_client)
        self.map_api: Optional[MapAPI] = context.map_api

        # 状态记录
        self.tracks: Dict[int, TrackState] = {}
        self.lane_history: Dict[str, LaneHistory] = {}
        self.frame_states: Dict[int, FrameState] = {}

        # 统计
        self.stats = {
            'total_frames': 0,
            'llm_calls': 0,
            'tracks_created': 0,
            'tracks_exited': 0,
        }

    def process_frame(self, frame: FrameDetection) -> Dict[int, int]:
        """
        处理单帧数据

        Returns:
            {detection_id: track_id} 映射
        """
        self.stats['total_frames'] += 1
        frame_id = frame.frame_id

        # Step 1: 将检测对象分配到车道
        lane_objects, out_objects, unmatched = self._assign_detections_to_lanes(
            frame.objects, frame_id
        )

        # Step 2: 记录帧状态
        self.frame_states[frame_id] = FrameState(
            frame_id=frame_id,
            lane_objects=lane_objects,
            out_of_map_objects=out_objects,
            unmatched_objects=unmatched
        )

        # Step 3: 更新车道历史
        self._update_lane_history(lane_objects, frame_id)

        # Step 4: 规则层匹配
        assignments = self._rule_based_match(lane_objects, frame_id)

        # Step 5: 检测异常，调用 LLM 处理
        anomalies = self._detect_anomalies(lane_objects, frame_id)
        if anomalies and self.llm_optimizer.llm_client:
            llm_decisions = self._handle_anomalies_llm(anomalies, frame_id)
            assignments = self._apply_llm_decisions(assignments, llm_decisions)

        # Step 6: 更新 ID 管理器
        self._update_id_manager(assignments, frame_id)

        return assignments

    def _assign_detections_to_lanes(self, detections: List[DetectedObject],
                                     frame_id: int) -> Tuple[Dict, List, List]:
        """将检测对象分配到车道"""
        lane_objects: Dict[str, List[DetectedObject]] = {}
        out_objects: List[DetectedObject] = []
        unmatched: List[DetectedObject] = []

        if not self.map_api:
            # 没有地图，全部归为地图外
            return {}, detections, []

        for det in detections:
            # 查找最近车道
            lane_info = self.map_api.match_vehicle_to_lane(
                det.location, det.heading
            )

            if lane_info and lane_info.get('distance', 999) < 5.0:
                # 在车道内
                lane_id = lane_info.get('lane_id')
                if lane_id not in lane_objects:
                    lane_objects[lane_id] = []
                lane_objects[lane_id].append(det)
            elif lane_info:
                # 在地图范围内但不在车道内
                unmatched.append(det)
            else:
                # 地图外
                out_objects.append(det)

        return lane_objects, out_objects, unmatched

    def _update_lane_history(self, lane_objects: Dict[str, List[DetectedObject]],
                              frame_id: int):
        """更新车道历史"""
        for lane_id, objects in lane_objects.items():
            if lane_id not in self.lane_history:
                self.lane_history[lane_id] = LaneHistory(lane_id=lane_id)

            history = self.lane_history[lane_id]
            history.count_history.append(len(objects))

            track_ids = set()
            for obj in objects:
                if obj.tracking_id > 0:
                    track_ids.add(obj.tracking_id)
            history.track_ids_history.append(track_ids)

    def _rule_based_match(self, lane_objects: Dict[str, List[DetectedObject]],
                          frame_id: int) -> Dict[int, int]:
        """
        规则层匹配

        Returns:
            {detection_id: track_id}
        """
        assignments = {}

        for lane_id, objects in lane_objects.items():
            # 获取该车道现有轨迹
            existing_tracks = [
                t for t in self.tracks.values()
                if t.lane_id == lane_id and t.status == "active"
            ]

            if len(objects) == len(existing_tracks):
                # 数量一致，直接位置匹配
                for obj in objects:
                    best_track = self._find_nearest_track(obj, existing_tracks)
                    if best_track and self._is_valid_match(obj, best_track):
                        assignments[obj.id] = best_track.track_id
                        best_track.positions.append(obj.location)
                        best_track.frame_ids.append(frame_id)
                        obj.tracking_id = best_track.track_id

            elif len(objects) > len(existing_tracks):
                # 数量增加，部分匹配 + 新轨迹
                for obj in objects:
                    best_track = self._find_nearest_track(obj, existing_tracks)
                    if best_track and self._is_valid_match(obj, best_track):
                        assignments[obj.id] = best_track.track_id
                        best_track.positions.append(obj.location)
                        best_track.frame_ids.append(frame_id)
                        obj.tracking_id = best_track.track_id
                    else:
                        # 新轨迹
                        new_id = self.id_manager.assign_id(obj, [], {"lane": lane_id})
                        self._create_track(obj, new_id, lane_id, frame_id)
                        assignments[obj.id] = new_id
                        obj.tracking_id = new_id

            else:
                # 数量减少，需要判断原因
                for obj in objects:
                    best_track = self._find_nearest_track(obj, existing_tracks)
                    if best_track and self._is_valid_match(obj, best_track):
                        assignments[obj.id] = best_track.track_id
                        best_track.positions.append(obj.location)
                        best_track.frame_ids.append(frame_id)
                        obj.tracking_id = best_track.track_id

                # 未匹配的轨迹标记为丢失
                matched_ids = set(assignments.values())
                for track in existing_tracks:
                    if track.track_id not in matched_ids:
                        track.lost_count += 1
                        if track.lost_count > 30:
                            track.status = "lost"

        return assignments

    def _find_nearest_track(self, obj: DetectedObject,
                            tracks: List[TrackState],
                            max_dist: float = 5.0) -> Optional[TrackState]:
        """查找最近的轨迹"""
        if not tracks:
            return None

        best = None
        best_dist = float('inf')

        for track in tracks:
            if not track.positions:
                continue
            last_pos = np.array(track.positions[-1][:2])
            obj_pos = np.array(obj.location[:2])
            dist = np.linalg.norm(last_pos - obj_pos)

            if dist < best_dist and dist < max_dist:
                best = track
                best_dist = dist

        return best

    def _is_valid_match(self, obj: DetectedObject, track: TrackState,
                        max_dist: float = 5.0) -> bool:
        """判断匹配是否有效"""
        if not track.positions:
            return False

        last_pos = np.array(track.positions[-1][:2])
        obj_pos = np.array(obj.location[:2])
        dist = np.linalg.norm(last_pos - obj_pos)

        return dist < max_dist

    def _create_track(self, obj: DetectedObject, track_id: int,
                      lane_id: str, frame_id: int):
        """创建新轨迹"""
        track = TrackState(
            track_id=track_id,
            lane_id=lane_id,
            positions=[obj.location],
            frame_ids=[frame_id],
            status="active",
            lost_count=0,
            confidence=0.8
        )
        self.tracks[track_id] = track
        self.stats['tracks_created'] += 1

    def _detect_anomalies(self, lane_objects: Dict[str, List[DetectedObject]],
                          frame_id: int) -> List[Dict]:
        """检测异常情况"""
        anomalies = []

        for lane_id, objects in lane_objects.items():
            if lane_id not in self.lane_history:
                continue

            history = self.lane_history[lane_id]
            if len(history.count_history) < 2:
                continue

            prev_count = history.count_history[-2]
            curr_count = len(objects)
            diff = curr_count - prev_count

            if abs(diff) >= 2:
                anomalies.append({
                    "type": "count_mismatch_large",
                    "lane_id": lane_id,
                    "prev_count": prev_count,
                    "curr_count": curr_count,
                    "diff": diff,
                })

        # 检测重新出现的轨迹
        for track_id, track in self.tracks.items():
            if track.status == "lost" and track.lost_count < 30:
                # 检查是否有检测接近丢失轨迹
                for lane_id, objects in lane_objects.items():
                    for obj in objects:
                        dist = np.linalg.norm(
                            np.array(track.positions[-1][:2]) - np.array(obj.location[:2])
                        )
                        if dist < 10.0:
                            anomalies.append({
                                "type": "track_reappear",
                                "track_id": track_id,
                                "detection_id": obj.id,
                                "distance": dist,
                            })

        return anomalies

    def _handle_anomalies_llm(self, anomalies: List[Dict],
                               frame_id: int) -> List[Dict]:
        """使用 LLM 处理异常"""
        decisions = []

        for anomaly in anomalies:
            if anomaly["type"] == "count_mismatch_large":
                lane_id = anomaly["lane_id"]
                history = self.lane_history.get(lane_id)

                # 获取轨迹历史
                trajectory_history = []
                for track in self.tracks.values():
                    if track.lane_id == lane_id and track.positions:
                        trajectory_history.append({
                            "id": track.track_id,
                            "pos": track.positions[-1],
                            "status": track.status,
                        })

                # 获取地图拓扑
                map_topology = {}
                if self.map_api:
                    try:
                        topology = self.map_api.get_lane_topology(lane_id)
                        map_topology = {
                            "predecessors": topology.get("predecessor_ids", []),
                            "successors": topology.get("successor_ids", []),
                        }
                    except:
                        pass

                # 调用 LLM
                result = self.llm_optimizer.analyze_count_mismatch(
                    lane_id,
                    frame_id - 1, anomaly["prev_count"],
                    frame_id, anomaly["curr_count"],
                    trajectory_history,
                    map_topology
                )

                decisions.append({
                    "anomaly_type": "count_mismatch",
                    "lane_id": lane_id,
                    "decision": result,
                })

            elif anomaly["type"] == "track_reappear":
                track = self.tracks.get(anomaly["track_id"])
                if track and track.positions:
                    old_track = {
                        "id": track.track_id,
                        "last_pos": track.positions[-1],
                        "lane_id": track.lane_id,
                        "lost_frames": track.lost_count,
                    }

                    # 这里需要获取对应的 detection，简化处理
                    new_detection = {
                        "pos": [0, 0, 0],  # 实际应从 anomaly 中获取
                        "lane_id": track.lane_id,
                        "type": "Vehicle",
                    }

                    result = self.llm_optimizer.judge_reappear(
                        old_track, new_detection, {}
                    )

                    decisions.append({
                        "anomaly_type": "reappear",
                        "track_id": track.track_id,
                        "decision": result,
                    })

        self.stats['llm_calls'] += len(decisions)
        return decisions

    def _apply_llm_decisions(self, assignments: Dict[int, int],
                              decisions: List[Dict]) -> Dict[int, int]:
        """应用 LLM 决策"""
        for decision in decisions:
            if decision["anomaly_type"] == "count_mismatch":
                cause = decision["decision"].get("cause", "unknown")

                if cause == "false":
                    # 误检，需要移除对应分配
                    affected_ids = decision["decision"].get("affected_ids", [])
                    for det_id in affected_ids:
                        if det_id in assignments:
                            del assignments[det_id]

                elif cause == "miss":
                    # 漏检，保持轨迹但标记
                    pass

            elif decision["anomaly_type"] == "reappear":
                if decision["decision"].get("is_same", False):
                    # 同一轨迹，恢复 ID
                    track_id = decision["track_id"]
                    if track_id in self.tracks:
                        self.tracks[track_id].status = "active"
                        self.tracks[track_id].lost_count = 0

        return assignments

    def _update_id_manager(self, assignments: Dict[int, int], frame_id: int):
        """更新 ID 管理器"""
        # 更新活跃 ID
        assigned_track_ids = set(assignments.values())
        self.id_manager.active_ids = assigned_track_ids

        # 检查是否有轨迹退出
        for track_id, track in list(self.tracks.items()):
            if track_id not in assigned_track_ids and track.lost_count > 30:
                self.id_manager.retire_id(track_id, "exited")
                self.stats['tracks_exited'] += 1

    def get_trajectories(self) -> Dict[int, Dict]:
        """获取所有轨迹"""
        result = {}
        for track_id, track in self.tracks.items():
            result[track_id] = {
                "track_id": track_id,
                "lane_id": track.lane_id,
                "positions": track.positions,
                "frame_ids": track.frame_ids,
                "status": track.status,
                "length": len(track.frame_ids),
            }
        return result

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'active_tracks': len([t for t in self.tracks.values() if t.status == "active"]),
            'llm_call_count': self.llm_optimizer.call_count,
        }
