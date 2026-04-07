# vim: expandtab:ts=4:sw=4
"""
Track 类 - 基于官方 deep_sort 实现

状态空间：(x, y, a, h, vx, vy, va, vh)
"""
import numpy as np


class TrackState:
    """轨迹状态枚举"""
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    单目标轨迹

    参数
    ----------
    mean : ndarray
        初始状态均值 (8 维：x, y, a, h, vx, vy, va, vh)
    covariance : ndarray
        初始协方差 (8x8)
    track_id : int
        唯一轨迹 ID
    n_init : int
        确认轨迹需要的连续检测次数
    max_age : int
        删除前最大丢失帧数
    feature : ndarray | None
        初始特征向量
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        # 额外存储 3D 信息
        self.positions = []
        self.velocities = []
        self.frame_ids = []
        self.obj_type = "Unknown"

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """获取当前 2D 边界框 (top left x, top left y, width, height)"""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = a * h
        ret[:2] -= ret[2:] / 2  # tl = center - wh/2
        return ret

    def to_tlbr(self):
        """获取当前 2D 边界框 (min x, min y, max x, max y)"""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """使用卡尔曼滤波预测"""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """使用卡尔曼滤波更新"""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())

        # 存储 3D 信息
        if detection.location is not None:
            self.positions.append(detection.location.tolist())
        if detection.velocity is not None:
            self.velocities.append(detection.velocity.tolist())
        self.frame_ids.append(int(detection.heading))  # 临时用 heading 存 frame_id
        self.obj_type = detection.obj_type

        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """标记丢失"""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
