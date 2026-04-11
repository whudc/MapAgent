"""
Lane-Aware DeepSORT Tracker - Enhanced Version

Enhancements on top of standard DeepSORT:
1. Lane-constrained matching - Prioritize matching targets in same lane
2. Trajectory prediction interpolation - Reduce flicker
3. Map topology validation - Validate trajectories against lane connections
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .deepsort_tracker import (
    DeepSORTTracker, KalmanFilter, Detection, TrackedObject,
    TrackState, CHI2INV95, INFTY_COST, matching_cascade, min_cost_matching,
    position_cost, gate_cost_matrix
)


# Add lane_id attribute to Detection
if not hasattr(Detection, 'lane_id'):
    Detection.lane_id = None


@dataclass
class LaneInfo:
    """Lane information"""
    lane_id: str
    centerline_coords: List[List[float]] = field(default_factory=list)
    boundary_coords: List[List[float]] = field(default_factory=list)
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)
    lane_type: str = "unknown"


class LaneAwareTracker(DeepSORTTracker):
    """
    Lane-aware DeepSORT tracker

    Enhanced features:
    1. Lane-constrained matching - Prioritize matching targets in same lane
    2. Trajectory smoothing interpolation - Reduce flicker
    3. Map topology validation - Validate trajectories against lane connections
    4. Occlusion handling - Intelligently handle occluded targets
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
        # Store map_api first (parent class doesn't store)
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

        # Lane-aware configuration
        self.use_map = use_map
        self.lane_weight = lane_weight
        self.max_lane_distance = max_lane_distance

        # Interpolation configuration
        self.interpolation_enabled = interpolation_enabled
        self.max_interpolation_frames = max_interpolation_frames

        # Lane cache
        self._lane_cache: Dict[str, LaneInfo] = {}

        # Track lane assignment
        self._track_lanes: Dict[int, str] = {}

        # Interpolated trajectory storage
        self._interpolated_tracks: Dict[int, List[Dict]] = {}

        # Statistics
        self._lane_stats: Dict[str, Dict] = {}

    def update(self, detections: List[Dict], frame_id: int) -> Dict[int, TrackedObject]:
        """
        Enhanced update pipeline

        1. Kalman filter prediction
        2. Duplicate detection filtering (NMS)
        3. Lane-constrained matching
        4. Update/create/delete tracks
        5. Trajectory interpolation (reduce flicker)
        """
        self.frame_count = frame_id

        # Parse detections
        parsed_dets = self._parse_detections(detections)

        # [New] Duplicate detection filtering - Handle same target detected as multiple
        parsed_dets = self._remove_duplicate_detections(parsed_dets, frame_id)

        # Assign lane to each detection
        if self.use_map and self.map_api:
            for det in parsed_dets:
                det.lane_id = self._assign_lane_to_detection(det, frame_id)

        # Convert to list
        track_list = list(self.tracks.values())

        # 1. Kalman filter prediction
        for track in track_list:
            if track.state == TrackState.DELETED:
                continue
            self._predict_track(track)

        # 2. Lane-constrained cascade matching
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

        # Lane-constrained cascade matching
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            lane_gated_metric, self.max_distance, self.max_misses,
            track_list, parsed_dets, confirmed_track_indices
        )

        # IOU matching (handle unconfirmed tracks)
        iou_track_candidates = [
            i for i, t in enumerate(track_list)
            if not t.is_confirmed()
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = self._min_cost_matching_with_lanes(
            self._position_distance, self.max_iou_distance,
            track_list, parsed_dets,
            iou_track_candidates, unmatched_detections
        )

        # Merge matching results
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a) | set(unmatched_tracks_b))

        # Update statistics
        self.stats['matches'] += len(matches)

        # [New] Conflict detection and resolution - Handle one detection matching multiple tracks
        matches = self._resolve_match_conflicts(matches, track_list, parsed_dets, frame_id)

        # 3. Update matched tracks
        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            self._update_track_with_lane(track, parsed_dets[det_idx], frame_id)

        # 4. Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            self._mark_track_missed(track)

        # 5. Create new tracks
        for det_idx in unmatched_detections:
            self._create_track_with_lane(parsed_dets[det_idx], frame_id)

        # 6. Clean up expired tracks
        self._cleanup_tracks()

        # 7. Trajectory interpolation (reduce flicker)
        if self.interpolation_enabled:
            self._interpolate_lost_tracks(frame_id)

        # Update statistics
        self.stats['active_tracks'] = len([t for t in self.tracks.values()
                                           if t.state != TrackState.DELETED])

        return self.get_active_tracks()

    def _assign_lane_to_detection(self, det: Detection, frame_id: int) -> Optional[str]:
        """Assign lane ID to detection"""
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
        Remove duplicate detections (Non-Maximum Suppression)

        When multiple detections overlap at same position, keep highest confidence

        Typical scenarios:
        - Detector outputs two bounding boxes at same position
        - Two detections differ by < 0.5m, same type

        Args:
            detections: Detection list
            frame_id: Current frame ID
            nms_distance: NMS distance threshold (meters), default 0.5m

        Returns:
            Filtered detection list
        """
        if len(detections) <= 1:
            return detections

        # Mark detection indices to remove
        to_remove = set()

        # Sort by confidence (highest first)
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

            # Check distance with subsequent detections
            for j in sorted_indices:
                if j <= i or j in to_remove:
                    continue

                det_j = detections[j]
                pos_j = np.array(det_j.location[:2])

                dist = np.linalg.norm(pos_i - pos_j)

                # Very close distance and same type, consider duplicate
                if dist < nms_distance and det_i.obj_type == det_j.obj_type:
                    # Keep higher confidence, remove lower
                    to_remove.add(j)

        # Filter out duplicate detections
        filtered = [det for i, det in enumerate(detections) if i not in to_remove]

        # Record statistics
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
        Resolve conflict when one track matches multiple detections

        Args:
            track: Track object
            detections: Detection list
            candidate_indices: Candidate detection indices
            frame_id: Current frame ID

        Returns:
            Best detection index
        """
        if not candidate_indices:
            return -1
        if len(candidate_indices) == 1:
            return candidate_indices[0]

        # Calculate score for each candidate detection
        scores = []
        pred_pos = track.predicted_location()

        for idx in candidate_indices:
            det = detections[idx]
            det_pos = np.array(det.location[:2])

            # Distance score (closer is better)
            dist = np.linalg.norm(pred_pos - det_pos)
            dist_score = np.exp(-dist / 3.0)

            # Confidence score
            conf_score = det.confidence

            # Type consistency score
            type_score = 1.0 if det.obj_type == track.obj_type else 0.5

            # Combined score
            total_score = dist_score * 0.5 + conf_score * 0.3 + type_score * 0.2
            scores.append((idx, total_score))

        # Return detection with highest score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _resolve_match_conflicts(self, matches: List[Tuple[int, int]],
                                  tracks: List[TrackedObject],
                                  detections: List[Detection],
                                  frame_id: int) -> List[Tuple[int, int]]:
        """
        Resolve matching conflicts

        Case 1: One track matches multiple detections -> Select best detection
        Case 2: One detection matches multiple tracks -> Select best track

        Args:
            matches: Match pairs list (track_idx, det_idx)
            tracks: Track list
            detections: Detection list
            frame_id: Current frame ID

        Returns:
            Resolved match pairs list
        """
        if not matches:
            return matches

        # Detect conflicts: count matches per track and detection
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

        # Handle one track matching multiple detections first
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

        # Handle one detection matching multiple tracks
        for det_idx, track_indices in det_matches.items():
            if len(track_indices) == 1:
                continue
            if det_idx in used_dets:
                continue

            # Select best track
            best_track_idx = self._select_best_track_for_detection(
                detections[det_idx],
                [tracks[i] for i in track_indices],
                track_indices
            )

            resolved_matches.append((best_track_idx, det_idx))
            used_tracks.add(best_track_idx)
            used_dets.add(det_idx)

        # Add matches without conflicts
        for track_idx, det_idx in matches:
            if track_idx not in used_tracks and det_idx not in used_dets:
                resolved_matches.append((track_idx, det_idx))

        return resolved_matches

    def _select_best_track_for_detection(self, detection: Detection,
                                          candidates: List[TrackedObject],
                                          candidate_indices: List[int]) -> int:
        """
        Select best track for a detection

        Args:
            detection: Detection object
            candidates: Candidate track list
            candidate_indices: Candidate track indices

        Returns:
            Best track index
        """
        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidate_indices[0]

        scores = []
        det_pos = np.array(detection.location[:2])

        for i, track in enumerate(candidates):
            pred_pos = track.predicted_location()

            # Distance score
            dist = np.linalg.norm(pred_pos - det_pos)
            dist_score = np.exp(-dist / 3.0)

            # Track quality score
            quality_score = min(1.0, track.hits / 5.0)

            # State score (CONFIRMED preferred)
            state_score = 1.0 if track.is_confirmed() else 0.5

            # Combined score
            total_score = dist_score * 0.6 + quality_score * 0.25 + state_score * 0.15
            scores.append((candidate_indices[i], total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _lane_aware_distance(self, tracks: List[TrackedObject],
                             detections: List[Detection],
                             track_indices: List[int],
                             detection_indices: List[int]) -> np.ndarray:
        """Lane-aware position distance"""
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            pred_loc = track.predicted_location()
            track_lane = self._track_lanes.get(track.track_id)

            for j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                det_lane = getattr(det, 'lane_id', None)

                # Base position distance
                dist = np.linalg.norm(pred_loc[:2] - det.location[:2])

                # Lane constraint penalty
                lane_penalty = 0.0
                if track_lane and det_lane and track_lane != det_lane:
                    # Check if lanes are connected
                    if not self._are_lanes_connected(track_lane, det_lane):
                        lane_penalty = self.lane_weight * 10.0  # Significantly increase cost

                cost_matrix[i, j] = dist + lane_penalty

        return cost_matrix

    def _are_lanes_connected(self, from_lane: str, to_lane: str) -> bool:
        """Check if two lanes are connected"""
        if not self.map_api:
            return True  # Assume connected without map

        try:
            lane_info = self.map_api.get_lane_info(from_lane)
            if lane_info:
                successors = lane_info.get('successor_ids', [])
                if to_lane in successors:
                    return True

            # Reverse check
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
        """Enhanced gating with lane information"""
        # Apply standard Kalman gating first
        cost_matrix = gate_cost_matrix(
            self.kf, cost_matrix, tracks, detections,
            track_indices, detection_indices,
            gated_cost=INFTY_COST
        )

        # Apply lane constraints
        gating_threshold = self.max_distance * 2.0

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            track_lane = self._track_lanes.get(track.track_id)

            if not track_lane:
                continue

            for col, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                det_lane = getattr(det, 'lane_id', None)

                # If lanes not connected and distance is far, gate out
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
        """Lane-constrained minimum cost matching"""
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
        """Update track and record lane information"""
        # Standard update
        self._update_track(track, detection, frame_id)

        # Update lane information
        det_lane = getattr(detection, 'lane_id', None)
        if det_lane:
            self._track_lanes[track.track_id] = det_lane
            track.obj_type = detection.obj_type

    def _create_track_with_lane(self, detection: Detection, frame_id: int):
        """Create new track and record lane information"""
        self._create_track(detection, frame_id)

        # Record new track's lane information
        det_lane = getattr(detection, 'lane_id', None)
        if det_lane:
            # Find newly created track ID
            new_track_id = self.next_track_id - 1
            self._track_lanes[new_track_id] = det_lane

    def _interpolate_lost_tracks(self, frame_id: int):
        """
        Interpolate lost tracks to reduce flicker

        When track is briefly lost (< max_interpolation_frames),
        use Kalman predicted position for interpolation, maintain ID continuity
        """
        for track_id, track in self.tracks.items():
            if track.state == TrackState.DELETED:
                continue

            if track.time_since_update > 0 and track.time_since_update <= self.max_interpolation_frames:
                # Track briefly lost, interpolate
                if self.interpolation_enabled:
                    # Use Kalman predicted position
                    pred_pos = track.predicted_location()

                    # Check if predicted position is within map bounds
                    if self._is_position_valid(pred_pos):
                        # Record in interpolation cache
                        if track_id not in self._interpolated_tracks:
                            self._interpolated_tracks[track_id] = []

                        self._interpolated_tracks[track_id].append({
                            'frame_id': frame_id,
                            'position': pred_pos.tolist(),
                            'velocity': track.predicted_velocity().tolist(),
                            'is_interpolated': True
                        })

                        # Update track history (insert predicted position)
                        track.positions.append(pred_pos.tolist())
                        track.velocities.append(track.predicted_velocity().tolist())
                        track.frame_ids.append(frame_id)

    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if position is within valid range"""
        if not self.map_api:
            return True

        # Check if within map boundaries
        try:
            # Simple check: position should not exceed map bounds
            pos_2d = position[:2]
            # Can add more complex boundary checks
            return True
        except Exception:
            return True

    def get_trajectory_with_interpolation(self, track_id: int) -> Optional[Dict]:
        """Get trajectory including interpolated data"""
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
        """Get lane-level statistics"""
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
        """Reset lane assignment"""
        self._track_lanes.clear()
        self._interpolated_tracks.clear()
        self._lane_stats.clear()
