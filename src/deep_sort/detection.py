# vim: expandtab:ts=4:sw=4
"""
Detection 类 - 适配 3D 边界框数据

基于官方 deep_sort 实现
"""
import numpy as np


class Detection(object):
    """
    表示一个目标检测

    参数
    ----------
    tlwh : array_like
        2D 边界框格式 `(x, y, w, h)` -  xy 平面投影
    confidence : float
        检测置信度
    feature : array_like | None
        外观特征向量（可选）
    location : array_like
        3D 位置 (x, y, z)
    size : array_like
        3D 尺寸 (l, w, h)
    velocity : array_like
        3D 速度 (vx, vy, vz)
    obj_type : str
        目标类型
    heading : float
        航向角
    """

    def __init__(self, tlwh, confidence=1.0, feature=None,
                 location=None, size=None, velocity=None,
                 obj_type="Unknown", heading=0.0):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32) if feature is not None else None

        # 3D 信息（可选）
        self.location = np.asarray(location, dtype=np.float64) if location is not None else None
        self.size = np.asarray(size, dtype=np.float64) if size is not None else None
        self.velocity = np.asarray(velocity, dtype=np.float64) if velocity is not None else None
        self.obj_type = obj_type
        self.heading = float(heading)

    def to_tlbr(self):
        """转换为 (min x, min y, max x, max y) 格式"""
        ret = self.tlwh.copy()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_xyah(self):
        """转换为 (center x, center y, aspect ratio, height) 格式"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2      # center = tl + wh/2
        ret[2] /= ret[3]             # aspect ratio = w/h
        return ret

    @staticmethod
    def from_3d_object(location, size, heading, velocity=None,
                       confidence=1.0, feature=None, obj_type="Unknown"):
        """
        从 3D 边界框创建 Detection

        参数
        ----------
        location : array_like
            3D 位置 (x, y, z)
        size : array_like
            3D 尺寸 (length, width, height)
        heading : float
            航向角（弧度）
        velocity : array_like
            3D 速度 (vx, vy, vz)
        confidence : float
            置信度
        feature : array_like | None
            外观特征
        obj_type : str
            目标类型

        返回
        -------
        Detection
        """
        loc = np.array(location)
        sz = np.array(size)

        # 将 3D 边界框投影到 xy 平面
        # 使用 heading 旋转边界框，计算 xy 平面的投影尺寸
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # 旋转后的长宽投影
        projected_l = abs(sz[0] * cos_h) + abs(sz[1] * sin_h)
        projected_w = abs(sz[0] * sin_h) + abs(sz[1] * cos_h)

        # tlwh: (center_x, center_y, width, height)
        tlwh = np.array([
            loc[0] - projected_l / 2,  # top left x
            loc[1] - projected_w / 2,  # top left y
            projected_l,                # width
            projected_w                 # height
        ])

        return Detection(
            tlwh=tlwh,
            confidence=confidence,
            feature=feature,
            location=location,
            size=size,
            velocity=velocity if velocity is not None else [0, 0, 0],
            obj_type=obj_type,
            heading=heading
        )
