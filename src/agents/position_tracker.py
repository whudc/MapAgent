"""
基于位置的跟踪器

完全依赖位置进行帧间关联，不依赖检测 ID
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TrackState(str, Enum):
    """轨迹状态"""
    TENTATIVE = "tentative"    # 试探状态（刚创建）
    CONFIRMED = "confirmed"    # 确认状态
    COASTED = "coasted"        # 滑行状态（丢失检测）
    DELETED = "deleted"        # 已删除


@dataclass
class TrackedObject:
    """跟踪目标"""
    track_id: int
    state: TrackState = TrackState.TENTATIVE

    # 位置历史
    positions: List[List[float]] = field(default_factory=list)  # [x, y, z]
    velocities: List[List[float]] = field(default_factory=list)  # [vx, vy, vz]
    frame_ids: List[int] = field(default_factory=list)

    # 属性
    obj_type: str = "Unknown"
    last_detection: Dict = field(default_factory=dict)

    # 卡尔曼滤波状态
    kf_state: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [x, y, z, vx, vy, vz]
    kf_covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 10)

    # 统计
    hit_count: int = 0       # 连续命中次数
    miss_count: int = 0      # 连续丢失次数
    total_hits: int = 0      # 总命中次数

    @property
    def last_position(self) -> Optional[List[float]]:
        if self.positions:
            return self.positions[-1]
        return None

    @property
    def last_frame(self) -> Optional[int]:
        if self.frame_ids:
            return self.frame_ids[-1]
        return None

    @property
    def predicted_position(self) -> np.ndarray:
        """预测下一帧位置"""
        if len(self.positions) >= 2:
            # 使用速度预测
            last_pos = np.array(self.positions[-1])
            last_vel = np.array(self.velocities[-1]) if self.velocities else np.zeros(3)
            return last_pos + last_vel * 0.1  # 假设 0.1s 帧间隔
        elif self.positions:
            return np.array(self.positions[-1])
        return self.kf_state[:3]

    @property
    def age(self) -> int:
        """轨迹年龄（帧数）"""
        return len(self.frame_ids)


class PositionTracker:
    """
    基于位置的跟踪器

    使用匈牙利算法进行最优匹配，不依赖检测 ID
    """

    def __init__(self,
                 max_distance: float = 5.0,      # 最大匹配距离
                 max_velocity: float = 30.0,      # 最大速度 (m/s)
                 frame_interval: float = 0.1,     # 帧间隔 (s)
                 min_hits: int = 2,               # 确认轨迹需要的最小命中次数
                 max_misses: int = 5,             # 删除轨迹的最大丢失次数
                 use_kalman: bool = True):        # 是否使用卡尔曼滤波
        self.max_distance = max_distance
        self.max_velocity = max_velocity
        self.frame_interval = frame_interval
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.use_kalman = use_kalman

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
        更新跟踪器

        Args:
            detections: 检测列表，每个元素包含 'location'/'position', 'type' 等
            frame_id: 当前帧 ID

        Returns:
            更新后的轨迹字典
        """
        self.frame_count = frame_id

        # 1. 预测现有轨迹位置
        self._predict_tracks()

        # 2. 匹配检测与轨迹
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)

        # 3. 更新匹配的轨迹
        for track_id, det_idx in matched:
            self._update_track(track_id, detections[det_idx], frame_id)

        # 4. 处理未匹配的轨迹
        for track_id in unmatched_tracks:
            self._mark_track_missed(track_id)

        # 5. 创建新轨迹
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], frame_id)

        # 6. 清理过期轨迹
        self._cleanup_tracks()

        # 更新统计
        self.stats['active_tracks'] = len([t for t in self.tracks.values()
                                           if t.state != TrackState.DELETED])

        return self.get_active_tracks()

    def _predict_tracks(self):
        """预测所有轨迹的下一帧位置"""
        for track in self.tracks.values():
            if track.state == TrackState.DELETED:
                continue

            if self.use_kalman:
                # 简化的卡尔曼预测
                track.kf_state[:3] += track.kf_state[3:] * self.frame_interval
            else:
                # 使用速度预测
                if track.velocities:
                    vel = np.array(track.velocities[-1])
                    track.kf_state[:3] = np.array(track.positions[-1]) + vel * self.frame_interval

    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        匹配检测与轨迹

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())

        # 获取活跃轨迹
        active_tracks = [(tid, t) for tid, t in self.tracks.items()
                         if t.state != TrackState.DELETED]

        if not active_tracks:
            return [], list(range(len(detections))), []

        # 构建代价矩阵
        cost_matrix = np.full((len(active_tracks), len(detections)), np.inf)

        for i, (track_id, track) in enumerate(active_tracks):
            pred_pos = track.predicted_position

            for j, det in enumerate(detections):
                det_pos = np.array(det.get('location') or det.get('position', [0, 0, 0]))

                # 计算距离
                dist = np.linalg.norm(pred_pos[:2] - det_pos[:2])  # 只考虑 xy

                # 计算速度约束
                if track.positions:
                    last_pos = np.array(track.positions[-1])
                    velocity = np.linalg.norm(det_pos[:2] - last_pos[:2]) / self.frame_interval

                    if velocity > self.max_velocity:
                        continue  # 超过最大速度，跳过

                if dist < self.max_distance:
                    cost_matrix[i, j] = dist

        # 匈牙利算法匹配（简化版：贪心）
        matched = []
        matched_tracks = set()
        matched_dets = set()

        # 按距离排序
        indices = []
        for i in range(len(active_tracks)):
            for j in range(len(detections)):
                if cost_matrix[i, j] < np.inf:
                    indices.append((cost_matrix[i, j], i, j))
        indices.sort(key=lambda x: x[0])

        # 贪心匹配
        for cost, i, j in indices:
            track_id = active_tracks[i][0]
            if track_id not in matched_tracks and j not in matched_dets:
                matched.append((track_id, j))
                matched_tracks.add(track_id)
                matched_dets.add(j)
                self.stats['matches'] += 1

        # 未匹配的检测和轨迹
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        unmatched_tracks = [active_tracks[i][0] for i in range(len(active_tracks))
                           if active_tracks[i][0] not in matched_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def _update_track(self, track_id: int, detection: Dict, frame_id: int):
        """更新匹配的轨迹"""
        track = self.tracks.get(track_id)
        if not track:
            return

        pos = detection.get('location') or detection.get('position', [0, 0, 0])

        # 计算速度
        if track.positions:
            last_pos = np.array(track.positions[-1])
            vel = (np.array(pos) - last_pos) / self.frame_interval
            track.velocities.append(vel.tolist())
        else:
            track.velocities.append([0, 0, 0])

        # 更新位置
        track.positions.append(list(pos))
        track.frame_ids.append(frame_id)
        track.last_detection = detection
        track.obj_type = detection.get('type', track.obj_type)

        # 更新卡尔曼状态
        if self.use_kalman:
            track.kf_state[:3] = np.array(pos)
            if track.velocities:
                track.kf_state[3:] = np.array(track.velocities[-1])

        # 更新计数
        track.hit_count += 1
        track.miss_count = 0
        track.total_hits += 1

        # 状态转换
        if track.state == TrackState.TENTATIVE and track.hit_count >= self.min_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.COASTED:
            track.state = TrackState.CONFIRMED

    def _mark_track_missed(self, track_id: int):
        """标记轨迹丢失"""
        track = self.tracks.get(track_id)
        if not track:
            return

        track.miss_count += 1
        track.hit_count = 0
        self.stats['misses'] += 1

        # 状态转换
        if track.state == TrackState.CONFIRMED:
            track.state = TrackState.COASTED

        # 删除长时间丢失的轨迹
        if track.miss_count > self.max_misses:
            track.state = TrackState.DELETED
            self.stats['deleted_tracks'] += 1

    def _create_track(self, detection: Dict, frame_id: int):
        """创建新轨迹"""
        pos = detection.get('location') or detection.get('position', [0, 0, 0])

        track = TrackedObject(
            track_id=self.next_track_id,
            state=TrackState.TENTATIVE,
            positions=[list(pos)],
            velocities=[[0, 0, 0]],
            frame_ids=[frame_id],
            obj_type=detection.get('type', 'Unknown'),
            last_detection=detection,
        )

        # 初始化卡尔曼状态
        track.kf_state = np.array([pos[0], pos[1], pos[2], 0, 0, 0])

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


def run_position_tracking(frames: List[Dict],
                          max_distance: float = 5.0,
                          max_velocity: float = 30.0,
                          min_hits: int = 2,
                          max_misses: int = 5) -> Dict:
    """
    运行位置跟踪

    Args:
        frames: 帧数据列表，每帧包含 'frame_id' 和 'objects'
        max_distance: 最大匹配距离
        max_velocity: 最大速度
        min_hits: 确认轨迹需要的最小命中次数
        max_misses: 删除轨迹的最大丢失次数

    Returns:
        跟踪结果
    """
    tracker = PositionTracker(
        max_distance=max_distance,
        max_velocity=max_velocity,
        min_hits=min_hits,
        max_misses=max_misses,
    )

    results = {}

    for frame in frames:
        frame_id = frame.get('frame_id', 0)
        objects = frame.get('objects', [])

        # 提取检测
        detections = []
        for obj in objects:
            det = {
                'location': obj.get('location') or obj.get('position'),
                'type': obj.get('type', 'Unknown'),
                'raw_data': obj,
            }
            if det['location']:
                detections.append(det)

        # 更新跟踪
        tracks = tracker.update(detections, frame_id)

        # 保存结果
        results[frame_id] = {
            'track_ids': list(tracks.keys()),
            'detections_count': len(detections),
        }

    # 获取最终轨迹
    final_tracks = tracker.get_active_tracks()

    return {
        'tracks': {
            tid: {
                'track_id': tid,
                'positions': t.positions,
                'velocities': t.velocities,
                'frame_ids': t.frame_ids,
                'type': t.obj_type,
                'age': t.age,
                'state': t.state.value,
            }
            for tid, t in final_tracks.items()
        },
        'statistics': tracker.get_statistics(),
        'frame_results': results,
    }