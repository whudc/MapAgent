# vim: expandtab:ts=4:sw=4
"""
3D 位置卡尔曼滤波器

基于官方 deep_sort 的 KalmanFilter 实现，适配 3D 位置跟踪
"""
import numpy as np
import scipy.linalg


"""
卡方分布 95% 分位数（用于马氏距离门控）
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    3D 位置卡尔曼滤波器

    状态空间：(x, y, z, vx, vy, vz) - 6 维
    测量空间：(x, y, z, vx, vy, vz) - 6 维（使用位置 + 速度作为测量）

    物体运动遵循恒速模型
    """

    def __init__(self, dt=0.1):
        self.dt = dt  # 帧间隔

        # 状态转移矩阵 (6x6)
        ndim = 3
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 观测矩阵 (6x6) - 单位矩阵，直接观测全部状态
        self._update_mat = np.eye(2 * ndim, 2 * ndim)

        # 相对不确定性权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        从测量初始化轨迹

        参数
        ----------
        measurement : ndarray
            测量向量 (x, y, z, vx, vy, vz)

        返回
        -------
        (ndarray, ndarray)
            均值向量 (6 维) 和协方差矩阵 (6x6)
        """
        mean = measurement.copy()
        # 速度初始化为 0 或使用测量速度
        std = [
            2 * self._std_weight_position * 10.0,
            2 * self._std_weight_position * 10.0,
            2 * self._std_weight_position * 10.0,
            10 * self._std_weight_velocity * 10.0,
            10 * self._std_weight_velocity * 10.0,
            10 * self._std_weight_velocity * 10.0
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        预测步骤

        参数
        ----------
        mean : ndarray
            前一时刻的均值向量
        covariance : ndarray
            前一时刻的协方差矩阵

        返回
        -------
        (ndarray, ndarray)
            预测的均值向量和协方差矩阵
        """
        # 相对过程噪声
        ref_scale = max(np.linalg.norm(mean[:2]), 1.0)
        std_pos = [
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale
        ]
        std_vel = [
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        投影到测量空间

        参数
        ----------
        mean : ndarray
            状态均值向量
        covariance : ndarray
            状态协方差矩阵

        返回
        -------
        (ndarray, ndarray)
            投影的均值和协方差
        """
        ref_scale = max(np.linalg.norm(mean[:2]), 1.0)
        std = [
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale,
            self._std_weight_position * ref_scale,
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale,
            self._std_weight_velocity * ref_scale
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        更新步骤

        参数
        ----------
        mean : ndarray
            预测的均值向量
        covariance : ndarray
            预测的协方差矩阵
        measurement : ndarray
            测量向量

        返回
        -------
        (ndarray, ndarray)
            更新后的均值和协方差
        """
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
            # Cholesky 分解失败，使用伪逆
            S = projected_cov + np.eye(len(projected_cov)) * 1e-4
            K = np.dot(covariance @ self._update_mat.T, np.linalg.pinv(S))
            innovation = measurement - projected_mean
            new_mean = mean + np.dot(K, innovation)
            new_covariance = covariance - np.linalg.multi_dot((
                K, projected_cov, K.T)) + np.eye(len(covariance)) * 1e-4

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """
        计算马氏距离

        参数
        ----------
        mean : ndarray
            状态均值
        covariance : ndarray
            状态协方差
        measurements : ndarray
            Nx6 测量矩阵
        only_position : bool
            是否只计算位置距离

        返回
        -------
        ndarray
            马氏距离数组
        """
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
            # 失败时使用欧氏距离
            d = measurements - mean
            squared_maha = np.sum(d * d, axis=1)

        return squared_maha
