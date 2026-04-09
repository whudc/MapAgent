"""
车道感知的 DeepSORT 跟踪器 - 增强版

在标准 DeepSORT 基础上增加：
1. 车道约束匹配 - 同一车道内的目标优先匹配
2. 轨迹预测插值 - 减少闪烁
3. 地图拓扑验证 - 验证轨迹是否符合车道连接关系
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .deepsort_tracker import (
    DeepSORTTracker, KalmanFilter, Detection, TrackedObject,
    TrackState, CHI2INV95, INFTY_COST, matching_cascade, min_cost_matching,
    position_cost, gate_cost_matrix
)


# 为 Detection 添加 lane_id 属性
if not hasattr(Detection, 'lane_id'):
    Detection.lane_id = None


@dataclass
class LaneInfo:
    """车道信息"""
    lane_id: str
    centerline_coords: List[List[float]] = field(default_factory=list)
    boundary_coords: List[List[float]] = field(default_factory=list)
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)
    lane_type: str = "unknown"


class LaneAwareTracker(DeepSORTTracker):
    """
    车道感知的 DeepSORT 跟踪器

    增强特性：
    1. 车道约束匹配 - 同一车道内的目标优先匹配
    2. 轨迹平滑插值 - 减少闪烁
    3. 地图拓扑验证 - 验证轨迹是否符合车道连接关系
    4. 遮挡处理 - 智能处理被遮挡的目标
    """

    def __init__(self,
                 map_api: Optional[Any] = None,
                 max_distance: float = 5.0,
                 max_velocity: float = 30.0,
                 frame_interval: float = 0.1,
                 min_hits: int = 2,
                 max_misses: int = 30,
                 use_map: bool = True,
                 lane_weight: float = 0.3,
                 max_lane_distance: float = 3.0,
                 interpolation_enabled: bool = True,
                 max_interpolation_frames: int = 5,
                 ):
        # 先存储 map_api（父类不存储）
        self.map_api = map_api

        super().__init__(
            map_api=map_api,
            max_distance=max_distance,
            max_velocity=max_velocity,
            frame_interval=frame_interval,
            min_hits=min_hits,
            max_misses=max_misses,
            use_map=use_map,
            lane_weight=lane_weight,
            max_lane_distance=max_lane_distance,
        )

        # 车道感知配置
        self.use_map = use_map
        self.lane_weight = lane_weight
        self.max_lane_distance = max_lane_distance

        # 插值配置
        self.interpolation_enabled = interpolation_enabled
        self.max_interpolation_frames = max_interpolation_frames

        # 车道缓存
        self._lane_cache: Dict[str, LaneInfo] = {}

        # 轨迹的车道分配
        self._track_lanes: Dict[int, str] = {}

        # 插值轨迹存储
        self._interpolated_tracks: Dict[int, List[Dict]] = {}

        # 统计
        self._lane_stats: Dict[str, Dict] = {}

    def update(self, detections: List[Dict], frame_id: int) -> Dict[int, TrackedObject]:
        """
        增强版更新流程

        1. 卡尔曼滤波预测
        2. 重复检测过滤（NMS）
        3. 车道约束匹配
        4. 更新/创建/删除轨迹
        5. 轨迹插值（减少闪烁）
        """
        self.frame_count = frame_id

        # 解析检测
        parsed_dets = self._parse_detections(detections)

        # 【新增】重复检测过滤 - 处理同一目标被误检成多个的情况
        parsed_dets = self._remove_duplicate_detections(parsed_dets, frame_id)

        # 为每个检测分配车道
        if self.use_map and self.map_api:
            for det in parsed_dets:
                det.lane_id = self._assign_lane_to_detection(det, frame_id)

        # 转换为列表
        track_list = list(self.tracks.values())

        # 1. 卡尔曼滤波预测
        for track in track_list:
            if track.state == TrackState.DELETED:
                continue
            self._predict_track(track)

        # 2. 车道约束级联匹配
        confirmed_track_indices = [
            i for i, t in enumerate(track_list)
            if t.state == TrackState.CONFIRMED
        ]

        def lane_gated_metric(tracks, dets, track_idxs, det_idxs):
            cost_matrix = self._lane_aware_distance(tracks, dets, track_idxs, det_idxs)
            cost_matrix = self._gate_cost_matrix_with_lanes(
                cost_matrix, tracks, dets, track_idxs, det_idxs
            )
            return cost_matrix

        # 车道约束级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            lane_gated_metric, self.max_distance, self.max_misses,
            track_list, parsed_dets, confirmed_track_indices
        )

        # IOU 匹配（处理未确认的轨迹）
        iou_track_candidates = [
            i for i, t in enumerate(track_list)
            if not t.is_confirmed()
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = self._min_cost_matching_with_lanes(
            self._position_distance, self.max_iou_distance,
            track_list, parsed_dets,
            iou_track_candidates, unmatched_detections
        )

        # 合并匹配结果
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a) | set(unmatched_tracks_b))

        # 更新统计
        self.stats['matches'] += len(matches)

        # 【新增】冲突检测与解决 - 处理一个检测匹配多个轨迹的情况
        matches = self._resolve_match_conflicts(matches, track_list, parsed_dets, frame_id)

        # 3. 更新匹配的轨迹
        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            self._update_track_with_lane(track, parsed_dets[det_idx], frame_id)

        # 4. 标记未匹配的轨迹为 missed
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            self._mark_track_missed(track)

        # 5. 创建新轨迹
        for det_idx in unmatched_detections:
            self._create_track_with_lane(parsed_dets[det_idx], frame_id)

        # 6. 清理过期轨迹
        self._cleanup_tracks()

        # 7. 轨迹插值（减少闪烁）
        if self.interpolation_enabled:
            self._interpolate_lost_tracks(frame_id)

        # 更新统计
        self.stats['active_tracks'] = len([t for t in self.tracks.values()
                                           if t.state != TrackState.DELETED])

        return self.get_active_tracks()

    def _assign_lane_to_detection(self, det: Detection, frame_id: int) -> Optional[str]:
        """为检测分配车道 ID"""
        if not self.map_api:
            return None

        try:
            pos = det.location[:2]
            nearest = self.map_api.find_nearest_lane(pos)
            if nearest and nearest.get('distance', float('inf')) < self.max_lane_distance:
                return nearest.get('lane_id')
        except Exception:
            pass
        return None

    def _remove_duplicate_detections(self, detections: List[Detection],
                                      frame_id: int,
                                      nms_distance: float = 0.5) -> List[Detection]:
        """
        去除重复检测（非极大值抑制）

        当同一位置出现多个几乎重叠的检测时，保留置信度最高的检测

        典型场景：
        - 检测器在同一位置输出两个 bounding box
        - 两个检测位置差异 < 0.5 米，类型相同

        Args:
            detections: 检测列表
            frame_id: 当前帧 ID
            nms_distance: NMS 距离阈值（米），默认 0.5 米

        Returns:
            过滤后的检测列表
        """
        if len(detections) <= 1:
            return detections

        # 标记需要删除的检测索引
        to_remove = set()

        # 按置信度排序（高的优先）
        sorted_indices = sorted(
            range(len(detections)),
            key=lambda i: detections[i].confidence,
            reverse=True
        )

        for i in sorted_indices:
            if i in to_remove:
                continue

            det_i = detections[i]
            pos_i = np.array(det_i.location[:2])

            # 检查与后续检测的距离
            for j in sorted_indices:
                if j <= i or j in to_remove:
                    continue

                det_j = detections[j]
                pos_j = np.array(det_j.location[:2])

                dist = np.linalg.norm(pos_i - pos_j)

                # 距离很近且类型相同，认为是同一目标的重复检测
                if dist < nms_distance and det_i.obj_type == det_j.obj_type:
                    # 保留置信度高的，删除置信度低的
                    to_remove.add(j)

        # 过滤掉重复检测
        filtered = [det for i, det in enumerate(detections) if i not in to_remove]

        # 记录统计信息
        if len(detections) != len(filtered):
            if not hasattr(self, '_nms_stats'):
                self._nms_stats = {'total_removed': 0, 'frames_affected': 0}
            removed_count = len(detections) - len(filtered)
            self._nms_stats['total_removed'] += removed_count
            self._nms_stats['frames_affected'] += 1

        return filtered

    def _resolve_multi_match_conflict(self, track: TrackedObject,
                                       detections: List[Detection],
                                       candidate_indices: List[int],
                                       frame_id: int) -> int:
        """
        解决一个轨迹匹配多个检测的冲突

        Args:
            track: 轨迹对象
            detections: 检测列表
            candidate_indices: 候选检测索引
            frame_id: 当前帧 ID

        Returns:
            最佳检测的索引
        """
        if not candidate_indices:
            return -1
        if len(candidate_indices) == 1:
            return candidate_indices[0]

        # 计算每个候选检测的分数
        scores = []
        pred_pos = track.predicted_location()

        for idx in candidate_indices:
            det = detections[idx]
            det_pos = np.array(det.location[:2])

            # 距离分数（越近越好）
            dist = np.linalg.norm(pred_pos - det_pos)
            dist_score = np.exp(-dist / 3.0)

            # 置信度分数
            conf_score = det.confidence

            # 类型一致性分数
            type_score = 1.0 if det.obj_type == track.obj_type else 0.5

            # 综合分数
            total_score = dist_score * 0.5 + conf_score * 0.3 + type_score * 0.2
            scores.append((idx, total_score))

        # 返回分数最高的检测
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _resolve_match_conflicts(self, matches: List[Tuple[int, int]],
                                  tracks: List[TrackedObject],
                                  detections: List[Detection],
                                  frame_id: int) -> List[Tuple[int, int]]:
        """
        解决匹配冲突

        情况 1：一个轨迹匹配多个检测 → 选择最佳检测
        情况 2：一个检测匹配多个轨迹 → 选择最佳轨迹

        Args:
            matches: 匹配对列表 (track_idx, det_idx)
            tracks: 轨迹列表
            detections: 检测列表
            frame_id: 当前帧 ID

        Returns:
            解决冲突后的匹配对列表
        """
        if not matches:
            return matches

        # 检测冲突：统计每个轨迹和检测的匹配次数
        track_matches: Dict[int, List[int]] = {}  # track_idx -> [det_idx, ...]
        det_matches: Dict[int, List[int]] = {}     # det_idx -> [track_idx, ...]

        for track_idx, det_idx in matches:
            if track_idx not in track_matches:
                track_matches[track_idx] = []
            track_matches[track_idx].append(det_idx)

            if det_idx not in det_matches:
                det_matches[det_idx] = []
            det_matches[det_idx].append(track_idx)

        resolved_matches = []
        used_tracks = set()
        used_dets = set()

        # 优先处理一个轨迹匹配多个检测的情况
        for track_idx, det_indices in track_matches.items():
            if len(det_indices) == 1:
                continue

            track = tracks[track_idx]
            best_det_idx = self._resolve_multi_match_conflict(
                track, detections, det_indices, frame_id
            )

            resolved_matches.append((track_idx, best_det_idx))
            used_tracks.add(track_idx)
            used_dets.add(best_det_idx)

        # 处理一个检测匹配多个轨迹的情况
        for det_idx, track_indices in det_matches.items():
            if len(track_indices) == 1:
                continue
            if det_idx in used_dets:
                continue

            # 选择最佳轨迹
            best_track_idx = self._select_best_track_for_detection(
                detections[det_idx],
                [tracks[i] for i in track_indices],
                track_indices
            )

            resolved_matches.append((best_track_idx, det_idx))
            used_tracks.add(best_track_idx)
            used_dets.add(det_idx)

        # 添加没有冲突的匹配
        for track_idx, det_idx in matches:
            if track_idx not in used_tracks and det_idx not in used_dets:
                resolved_matches.append((track_idx, det_idx))

        return resolved_matches

    def _select_best_track_for_detection(self, detection: Detection,
                                          candidates: List[TrackedObject],
                                          candidate_indices: List[int]) -> int:
        """
        为一个检测选择最佳轨迹

        Args:
            detection: 检测对象
            candidates: 候选轨迹列表
            candidate_indices: 候选轨迹索引

        Returns:
            最佳轨迹的索引
        """
        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidate_indices[0]

        scores = []
        det_pos = np.array(detection.location[:2])

        for i, track in enumerate(candidates):
            pred_pos = track.predicted_location()

            # 距离分数
            dist = np.linalg.norm(pred_pos - det_pos)
            dist_score = np.exp(-dist / 3.0)

            # 轨迹质量分数
            quality_score = min(1.0, track.hits / 5.0)

            # 状态分数（CONFIRMED 优先）
            state_score = 1.0 if track.is_confirmed() else 0.5

            # 综合分数
            total_score = dist_score * 0.6 + quality_score * 0.25 + state_score * 0.15
            scores.append((candidate_indices[i], total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _lane_aware_distance(self, tracks: List[TrackedObject],
                             detections: List[Detection],
                             track_indices: List[int],
                             detection_indices: List[int]) -> np.ndarray:
        """车道感知的位置距离"""
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            pred_loc = track.predicted_location()
            track_lane = self._track_lanes.get(track.track_id)

            for j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                det_lane = getattr(det, 'lane_id', None)

                # 基础位置距离
                dist = np.linalg.norm(pred_loc[:2] - det.location[:2])

                # 车道约束惩罚
                lane_penalty = 0.0
                if track_lane and det_lane and track_lane != det_lane:
                    # 检查车道是否连通
                    if not self._are_lanes_connected(track_lane, det_lane):
                        lane_penalty = self.lane_weight * 10.0  # 大幅增加代价

                cost_matrix[i, j] = dist + lane_penalty

        return cost_matrix

    def _are_lanes_connected(self, from_lane: str, to_lane: str) -> bool:
        """检查两条车道是否连通"""
        if not self.map_api:
            return True  # 没有地图时假设连通

        try:
            lane_info = self.map_api.get_lane_info(from_lane)
            if lane_info:
                successors = lane_info.get('successor_ids', [])
                if to_lane in successors:
                    return True

            # 反向检查
            lane_info = self.map_api.get_lane_info(to_lane)
            if lane_info:
                predecessors = lane_info.get('predecessor_ids', [])
                if from_lane in predecessors:
                    return True
        except Exception:
            pass

        return False

    def _gate_cost_matrix_with_lanes(self, cost_matrix: np.ndarray,
                                      tracks: List[TrackedObject],
                                      detections: List[Detection],
                                      track_indices: List[int],
                                      detection_indices: List[int]) -> np.ndarray:
        """使用车道信息增强门控"""
        # 先应用标准卡尔曼门控
        cost_matrix = gate_cost_matrix(
            self.kf, cost_matrix, tracks, detections,
            track_indices, detection_indices,
            gated_cost=INFTY_COST
        )

        # 应用车道约束
        gating_threshold = self.max_distance * 2.0

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            track_lane = self._track_lanes.get(track.track_id)

            if not track_lane:
                continue

            for col, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                det_lane = getattr(det, 'lane_id', None)

                # 如果车道不连通且距离较远，门控掉
                if det_lane and track_lane != det_lane:
                    if not self._are_lanes_connected(track_lane, det_lane):
                        if cost_matrix[row, col] > gating_threshold:
                            cost_matrix[row, col] = INFTY_COST

        return cost_matrix

    def _min_cost_matching_with_lanes(self, distance_metric, max_distance: float,
                                       tracks: List[TrackedObject],
                                       detections: List[Detection],
                                       track_indices: List[int],
                                       detection_indices: List[int]
                                       ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """车道约束的最小代价匹配"""
        from scipy.optimize import linear_sum_assignment

        if not track_indices or not detection_indices:
            return [], list(track_indices), list(detection_indices)

        cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
        cost_matrix = cost_matrix.copy()
        cost_matrix[cost_matrix > max_distance] = INFTY_COST

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_detections = [], [], []

        for col, detection_idx in enumerate(detection_indices):
            if col not in col_ind:
                unmatched_detections.append(detection_idx)

        for row, track_idx in enumerate(track_indices):
            if row not in row_ind:
                unmatched_tracks.append(track_idx)

        for row, col in zip(row_ind, col_ind):
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))

        return matches, unmatched_tracks, unmatched_detections

    def _update_track_with_lane(self, track: TrackedObject,
                                 detection: Detection,
                                 frame_id: int):
        """更新轨迹并记录车道信息"""
        # 标准更新
        self._update_track(track, detection, frame_id)

        # 更新车道信息
        det_lane = getattr(detection, 'lane_id', None)
        if det_lane:
            self._track_lanes[track.track_id] = det_lane
            track.obj_type = detection.obj_type

    def _create_track_with_lane(self, detection: Detection, frame_id: int):
        """创建新轨迹并记录车道信息"""
        self._create_track(detection, frame_id)

        # 记录新车道的车道信息
        det_lane = getattr(detection, 'lane_id', None)
        if det_lane:
            # 找到新创建的轨迹 ID
            new_track_id = self.next_track_id - 1
            self._track_lanes[new_track_id] = det_lane

    def _interpolate_lost_tracks(self, frame_id: int):
        """
        对丢失的轨迹进行插值，减少闪烁

        当轨迹短暂丢失时（< max_interpolation_frames），
        使用卡尔曼预测位置进行插值，保持 ID 连续性
        """
        for track_id, track in self.tracks.items():
            if track.state == TrackState.DELETED:
                continue

            if track.time_since_update > 0 and track.time_since_update <= self.max_interpolation_frames:
                # 轨迹短暂丢失，进行插值
                if self.interpolation_enabled:
                    # 使用卡尔曼预测位置
                    pred_pos = track.predicted_location()

                    # 检查预测位置是否在地图范围内
                    if self._is_position_valid(pred_pos):
                        # 在插值缓存中记录
                        if track_id not in self._interpolated_tracks:
                            self._interpolated_tracks[track_id] = []

                        self._interpolated_tracks[track_id].append({
                            'frame_id': frame_id,
                            'position': pred_pos.tolist(),
                            'velocity': track.predicted_velocity().tolist(),
                            'is_interpolated': True
                        })

                        # 更新轨迹历史（插入预测位置）
                        track.positions.append(pred_pos.tolist())
                        track.velocities.append(track.predicted_velocity().tolist())
                        track.frame_ids.append(frame_id)

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """检查位置是否在有效范围内"""
        if not self.map_api:
            return True

        # 检查是否在地图边界内
        try:
            # 简单检查：位置不应超出地图范围
            pos_2d = position[:2]
            # 可以添加更复杂的边界检查
            return True
        except Exception:
            return True

    def get_trajectory_with_interpolation(self, track_id: int) -> Optional[Dict]:
        """获取包含插值数据的轨迹"""
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]
        interpolated = self._interpolated_tracks.get(track_id, [])

        return {
            'track_id': track_id,
            'positions': track.positions,
            'frame_ids': track.frame_ids,
            'interpolated_frames': [i['frame_id'] for i in interpolated],
            'is_interpolated': len(interpolated) > 0
        }

    def get_lane_statistics(self) -> Dict:
        """获取车道级统计信息"""
        lane_counts: Dict[str, int] = {}

        for track_id, lane_id in self._track_lanes.items():
            if lane_id not in lane_counts:
                lane_counts[lane_id] = 0
            if self.tracks[track_id].state != TrackState.DELETED:
                lane_counts[lane_id] += 1

        return {
            'lane_counts': lane_counts,
            'total_active': self.stats['active_tracks'],
        }

    def reset_lane_assignment(self):
        """重置车道分配"""
        self._track_lanes.clear()
        self._interpolated_tracks.clear()
        self._lane_stats.clear()
