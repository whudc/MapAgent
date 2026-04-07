# vim: expandtab:ts=4:sw=4
"""
多目标跟踪器 - 基于官方 deep_sort 实现

适配 3D 边界框数据，不使用外观特征
"""
from __future__ import absolute_import
import numpy as np

from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    多目标跟踪器

    参数
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        距离度量（可选，不使用外观特征时可传 None）
    max_iou_distance : float
        IoU 匹配最大距离
    max_age : int
        轨迹删除前最大丢失帧数
    n_init : int
        确认轨迹需要的连续检测次数
    """

    def __init__(self, metric=None, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """预测轨迹状态"""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        执行测量更新和轨迹管理

        参数
        ----------
        detections : List[Detection]
            当前帧检测列表
        """
        # 级联匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 更新轨迹集合
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 更新距离度量（如果有）
        if self.metric is not None:
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
            features, targets = [], []
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                if track.features:
                    features += track.features
                    targets += [track.track_id for _ in track.features]
                track.features = []
            if features:
                self.metric.partial_fit(
                    np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """匹配轨迹和检测"""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 使用 IOU 距离代替外观特征
            cost_matrix = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        # 分割为已确认和未确认轨迹
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 级联匹配已确认轨迹
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.max_iou_distance, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 使用 IOU 关联剩余轨迹
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """初始化新轨迹"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
