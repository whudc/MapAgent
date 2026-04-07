"""
DeepSORT 跟踪器 - 简化版（纯位置跟踪）

基于官方 deep_sort 实现，仅使用位置信息进行多目标跟踪
参考：https://github.com/nwojke/deep_sort

主要特性：
1. 标准卡尔曼滤波（位置/速度）
2. 级联匹配（按 time_since_update 分层）
3. 马氏距离门控
4. 位置距离匹配
"""

import numpy as np
import scipy.linalg
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# ==================== 常量定义 ====================

# 卡方分布 95% 分位数（用于马氏距离门控）
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

INFTY_COST = 1e+5


# ==================== 枚举和数据类 ====================

class TrackState(str, Enum):
    """轨迹状态"""
    TENTATIVE = "tentative"    # 试探状态
    CONFIRMED = "confirmed"    # 确认状态
    DELETED = "deleted"        # 已删除


@dataclass
class Detection:
    """检测框数据（3D 位置格式）"""
    location: np.ndarray           # 位置 [x, y, z]
    velocity: np.ndarray           # 速度 [vx, vy, vz]
    obj_type: str = "Unknown"      # 类型
    heading: float = 0.0           # 航向角
    speed: float = 0.0             # 速度大小
    confidence: float = 1.0        # 置信度

    def to_measurement(self) -> np.ndarray:
        """转换为测量向量 [x, y, z]"""
        return np.array([
            self.location[0], self.location[1], self.location[2]
        ])


@dataclass
class TrackedObject:
    """跟踪目标（DeepSORT Track）"""
    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # 卡尔曼滤波状态（6 维：x, y, z, vx, vy, vz）
    kf_mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    kf_covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 10)

    # 历史数据
    positions: List[List[float]] = field(default_factory=list)
    velocities: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)

    # 属性
    obj_type: str = "Unknown"

    # 统计
    hits: int = 0                  # 总命中次数
    age: int = 0                   # 轨迹年龄（帧数）
    time_since_update: int = 0     # 自上次更新以来的帧数

    @property
    def last_position(self) -> Optional[List[float]]:
        return self.positions[-1] if self.positions else None

    @property
    def last_frame(self) -> Optional[int]:
        return self.frame_ids[-1] if self.frame_ids else None

    def predicted_location(self) -> np.ndarray:
        """获取预测位置"""
        return self.kf_mean[:3]

    def predicted_velocity(self) -> np.ndarray:
        """获取预测速度"""
        return self.kf_mean[3:6]

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


# ==================== 卡尔曼滤波器 ====================

class KalmanFilter:
    """
    卡尔曼滤波器（基于官方 deep_sort 实现，适配 3D 位置）

    状态向量：[x, y, z, vx, vy, vz] (6 维)
    测量向量：[x, y, z] (3 维)
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt  # 帧间隔

        # 状态转移矩阵 F (6x6)
        ndim = 3  # 位置维度
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 观测矩阵 H (3x6)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 相对不确定性权重（基于官方 deep_sort）
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """初始化轨迹"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # 初始化协方差（相对不确定性）
        ref_scale = max(np.linalg.norm(measurement[:2]), 1.0)
        std = [
            2 * self._std_weight_position * ref_scale,
            2 * self._std_weight_position * ref_scale,
            2 * self._std_weight_position * ref_scale,
            10 * self._std_weight_velocity * ref_scale,
            10 * self._std_weight_velocity * ref_scale,
            10 * self._std_weight_velocity * ref_scale]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测步骤"""
        # 使用位置来计算参考尺度（更稳定）
        ref_scale = max(np.linalg.norm(mean[:2]), 1.0)
        std_pos = [
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale]
        std_vel = [
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将状态投影到测量空间"""
        # 使用位置来计算参考尺度
        ref_scale = max(np.linalg.norm(mean[:2]), 1.0)
        std = [
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """更新步骤"""
        projected_mean, projected_cov = self.project(mean, covariance)

        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T
            innovation = measurement - projected_mean
            new_mean = mean + np.dot(innovation, kalman_gain.T)
            new_covariance = covariance - np.linalg.multi_dot((
                kalman_gain, projected_cov, kalman_gain.T))
        except scipy.linalg.LinAlgError:
            S = projected_cov + np.eye(len(projected_cov)) * 1e-4
            K = np.dot(covariance @ self._update_mat.T, np.linalg.pinv(S))
            innovation = measurement - projected_mean
            new_mean = mean + np.dot(K, innovation)
            new_covariance = covariance - np.linalg.multi_dot((
                K, projected_cov, K.T)) + np.eye(len(covariance)) * 1e-4

        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray,
                        only_position: bool = False) -> np.ndarray:
        """计算马氏距离（用于门控）"""
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        try:
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            d = measurements - mean
            squared_maha = np.sum(d * d, axis=1)

        return squared_maha


# ==================== 匹配算法 ====================

def position_cost(tracks: List[TrackedObject], detections: List[Detection],
                  track_indices: Optional[List[int]] = None,
                  detection_indices: Optional[List[int]] = None,
                  max_distance: float = 5.0) -> np.ndarray:
    """位置距离代价"""
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if track.time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        pred_loc = track.predicted_location()

        for col, det_idx in enumerate(detection_indices):
            det = detections[det_idx]
            dist = np.linalg.norm(pred_loc[:2] - det.location[:2])
            cost_matrix[row, col] = dist

    return cost_matrix


def min_cost_matching(
        distance_metric, max_distance: float,
        tracks: List[TrackedObject], detections: List[Detection],
        track_indices: Optional[List[int]] = None,
        detection_indices: Optional[List[int]] = None
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """最小代价匹配（匈牙利算法）"""
    from scipy.optimize import linear_sum_assignment

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
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


def matching_cascade(
        distance_metric, max_distance: float, cascade_depth: int,
        tracks: List[TrackedObject], detections: List[Detection],
        track_indices: Optional[List[int]] = None,
        detection_indices: Optional[List[int]] = None
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    级联匹配（基于官方 deep_sort）

    按 time_since_update 分层匹配
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections
        )
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf: KalmanFilter, cost_matrix: np.ndarray,
        tracks: List[TrackedObject], detections: List[Detection],
        track_indices: List[int], detection_indices: List[int],
        gated_cost: float = INFTY_COST,
        only_position: bool = False,
) -> np.ndarray:
    """使用卡尔曼滤波门控无效关联"""
    gating_dim = 2 if only_position else 3
    gating_threshold = CHI2INV95[gating_dim]

    if len(detection_indices) == 0:
        return cost_matrix

    measurements = np.asarray([
        detections[i].to_measurement() for i in detection_indices
    ])

    if len(measurements) == 0:
        return cost_matrix

    cost_matrix = cost_matrix.copy()

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.kf_mean, track.kf_covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix


# ==================== DeepSORT 跟踪器 ====================

class DeepSORTTracker:
    """
    DeepSORT 多目标跟踪器（简化版 - 纯位置跟踪）

    参数：
    - max_distance: 位置距离最大阈值
    - max_velocity: 最大速度（米/秒）
    - min_hits: 确认轨迹需要的连续检测次数
    - max_misses: 轨迹最大丢失帧数
    """

    def __init__(self,
                 map_api: Optional[Any] = None,
                 max_distance: float = 5.0,
                 max_velocity: float = 30.0,
                 frame_interval: float = 0.1,
                 min_hits: int = 2,
                 max_misses: int = 30,
                 use_map: bool = False,
                 lane_weight: float = 0.0,
                 max_lane_distance: float = 0.0,
                 max_iou_distance: float = 5.0,
                 budget: Optional[int] = 100,
                 ):
        self.max_distance = max_distance
        self.max_velocity = max_velocity
        self.dt = frame_interval
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.max_iou_distance = max_iou_distance

        # 卡尔曼滤波器
        self.kf = KalmanFilter(dt=frame_interval)

        # 轨迹管理
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.frame_count = 0

        # 统计
        self.stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'matches': 0,
            'misses': 0,
            'new_tracks': 0,
            'deleted_tracks': 0,
        }

    def update(self, detections: List[Dict], frame_id: int) -> Dict[int, TrackedObject]:
        """
        DeepSORT 更新流程

        1. 卡尔曼滤波预测
        2. 级联匹配
        3. 更新/创建/删除轨迹
        """
        self.frame_count = frame_id

        # 解析检测
        parsed_dets = self._parse_detections(detections)

        # 转换为列表
        track_list = list(self.tracks.values())

        # 1. 卡尔曼滤波预测
        for track in track_list:
            if track.state == TrackState.DELETED:
                continue
            self._predict_track(track)

        # 2. 级联匹配 - 只匹配 CONFIRMED 轨迹
        confirmed_track_indices = [
            i for i, t in enumerate(track_list)
            if t.state == TrackState.CONFIRMED
        ]

        def gated_metric(tracks, dets, track_idxs, det_idxs):
            cost_matrix = self._position_distance(tracks, dets, track_idxs, det_idxs)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets,
                track_idxs, det_idxs, gated_cost=INFTY_COST
            )
            return cost_matrix

        # 级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric, self.max_distance, self.max_misses,
            track_list, parsed_dets, confirmed_track_indices
        )

        # IOU 匹配（处理未确认的轨迹）
        iou_track_candidates = [
            i for i, t in enumerate(track_list)
            if not t.is_confirmed()
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            position_cost, self.max_iou_distance,
            track_list, parsed_dets,
            iou_track_candidates, unmatched_detections
        )

        # 合并匹配结果
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a) | set(unmatched_tracks_b))

        # 更新统计
        self.stats['matches'] += len(matches)

        # 3. 更新匹配的轨迹
        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            self._update_track(track, parsed_dets[det_idx], frame_id)

        # 4. 标记未匹配的轨迹为 missed
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            self._mark_track_missed(track)

        # 5. 创建新轨迹
        for det_idx in unmatched_detections:
            self._create_track(parsed_dets[det_idx], frame_id)

        # 6. 清理过期轨迹
        self._cleanup_tracks()

        # 更新统计
        self.stats['active_tracks'] = len([t for t in self.tracks.values()
                                           if t.state != TrackState.DELETED])

        return self.get_active_tracks()

    def _parse_detections(self, detections: List[Dict]) -> List[Detection]:
        """解析检测数据"""
        parsed = []
        for det in detections:
            pos = det.get('location') or det.get('position')
            if pos is None:
                continue

            vel = det.get('velocity') or [0, 0, 0]
            heading = det.get('heading') or 0.0
            speed = det.get('speed') or 0.0

            d = Detection(
                location=np.array(pos),
                velocity=np.array(vel),
                obj_type=det.get('type', 'Unknown'),
                heading=heading,
                speed=speed,
            )
            parsed.append(d)
        return parsed

    def _position_distance(self, tracks: List[TrackedObject], detections: List[Detection],
                          track_indices: List[int], detection_indices: List[int]) -> np.ndarray:
        """计算位置距离"""
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            pred_loc = track.predicted_location()

            for j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                dist = np.linalg.norm(pred_loc[:2] - det.location[:2])
                cost_matrix[i, j] = dist

        return cost_matrix

    def _predict_track(self, track: TrackedObject):
        """预测轨迹状态"""
        track.kf_mean, track.kf_covariance = self.kf.predict(
            track.kf_mean, track.kf_covariance
        )
        track.age += 1
        track.time_since_update += 1

    def _update_track(self, track: TrackedObject, detection: Detection, frame_id: int):
        """更新轨迹"""
        measurement = detection.to_measurement()
        track.kf_mean, track.kf_covariance = self.kf.update(
            track.kf_mean, track.kf_covariance, measurement
        )

        # 更新历史
        track.positions.append(detection.location.tolist())
        track.velocities.append(detection.velocity.tolist())
        track.frame_ids.append(frame_id)
        track.obj_type = detection.obj_type

        # 更新统计
        track.hits += 1
        track.time_since_update = 0

        # 状态转换
        if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits:
            track.state = TrackState.CONFIRMED

    def _mark_track_missed(self, track: TrackedObject):
        """标记轨迹丢失"""
        track.time_since_update += 1
        self.stats['misses'] += 1

        if track.state == TrackState.TENTATIVE:
            track.state = TrackState.DELETED
        elif track.time_since_update > self.max_misses:
            track.state = TrackState.DELETED
            self.stats['deleted_tracks'] += 1

    def _create_track(self, detection: Detection, frame_id: int):
        """创建新轨迹"""
        measurement = detection.to_measurement()
        mean, covariance = self.kf.initiate(measurement)

        track = TrackedObject(
            track_id=self.next_track_id,
            state=TrackState.TENTATIVE,
            kf_mean=mean,
            kf_covariance=covariance,
            positions=[detection.location.tolist()],
            velocities=[detection.velocity.tolist()],
            frame_ids=[frame_id],
            obj_type=detection.obj_type,
            hits=1,
            age=1,
            time_since_update=0,
        )

        self.tracks[self.next_track_id] = track
        self.stats['total_tracks'] += 1
        self.stats['new_tracks'] += 1
        self.next_track_id += 1

    def _cleanup_tracks(self):
        """清理已删除的轨迹"""
        to_delete = [tid for tid, t in self.tracks.items()
                    if t.state == TrackState.DELETED]
        for tid in to_delete:
            del self.tracks[tid]

    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """获取活跃轨迹"""
        return {tid: t for tid, t in self.tracks.items()
                if t.state != TrackState.DELETED}

    def get_confirmed_tracks(self) -> Dict[int, TrackedObject]:
        """获取确认状态的轨迹"""
        return {tid: t for tid, t in self.tracks.items()
                if t.state == TrackState.CONFIRMED}

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        active = self.get_active_tracks()
        lengths = [t.age for t in active.values()]

        return {
            **self.stats,
            'current_active': len(active),
            'avg_track_length': np.mean(lengths) if lengths else 0,
            'max_track_length': max(lengths) if lengths else 0,
        }
