"""
Traffic Flow Reconstruction Agent - Supports Pure DeepSORT and LLM Hybrid Optimization Modes

Core Design:
1. Pure DeepSORT Mode: Uses Kalman filtering + cascade matching for multi-object tracking
2. LLM Hybrid Mode: Rule layer handles normal matching, LLM layer handles difficult cases
3. Map Constraints: Uses lane information to constrain vehicle matching, ensuring lane vehicle count conservation

Architecture:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Rule Layer  │ →  │  LLM Layer   │ →  │  Post-Proc   │
│  (Fast Match)│    │  (Reasoning) │    │  (ID Consist)│
│  +Lane Cstrnt│    │  (Count Anal)│    │  (Traj Smth) │
└──────────────┘    └──────────────┘    └──────────────┘

Optimization Principles:
1. Vehicle count in the same lane should be consistent across frames
2. Analyze count changes (miss/false/enter/exit)
3. Occluded targets are inferred through LLM + map topology
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import traceback
from pathlib import Path
import numpy as np
import logging
import os
from datetime import datetime

from agents.base import BaseAgent, AgentContext


# ============================================================================
# Logger Configuration
# ============================================================================

def setup_logger(log_dir: str = "logs", log_name: str = "traffic_flow") -> logging.Logger:
    """
    Configure logger

    Args:
        log_dir: Log directory
        log_name: Log name

    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("TrafficFlow")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (only show important messages)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger
_logger = setup_logger()


def log_info(msg: str):
    """Log INFO level message"""
    _logger.info(msg)


def log_debug(msg: str):
    """Log DEBUG level message"""
    _logger.debug(msg)


def log_warning(msg: str):
    """Log WARNING level message"""
    _logger.warning(msg)


def log_error(msg: str):
    """Log ERROR level message"""
    _logger.error(msg)


# ============================================================================
from agents.deepsort_tracker import DeepSORTTracker, TrackedObject, Detection, TrackState
from models.agent_io import VehicleState, VehicleTrajectory
from utils.detection_loader import DetectionLoader, FrameDetection
from core.llm_client import LLMClient


# ==================== Data Structures ====================

@dataclass
class Trajectory:
    """Trajectory data structure"""
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
        """Get position at specified frame"""
        if frame_id in self.frame_ids:
            idx = self.frame_ids.index(frame_id)
            return self.positions[idx]
        return None

    def to_vehicle_trajectory(self) -> VehicleTrajectory:
        """Convert to VehicleTrajectory format"""
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
    """Lane history state"""
    lane_id: str
    count_history: List[int] = field(default_factory=list)
    track_ids_history: List[Set[int]] = field(default_factory=list)


@dataclass
class LaneConstrainedTrack:
    """Lane-constrained trajectory"""
    track_id: int
    lane_id: Optional[str] = None
    positions: List[List[float]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    predicted_pos: Optional[List[float]] = None
    lost_count: int = 0
    last_update_frame: int = 0
    confidence: float = 1.0
    llm_verified: bool = False


class MatchResult(str, Enum):
    """Match result types"""
    MATCHED = "matched"
    MISS_DETECTION = "miss"
    FALSE_DETECTION = "false"
    EXITED = "exited"
    ENTERED = "entered"
    RE_ENTERED = "re_entered"


# ==================== ID Consistency Manager ====================

class IDConsistencyManager:
    """
    ID Consistency Manager

    Three-layer mechanism for long-term ID consistency:
    1. ID Allocation Pool - Ensure IDs are not reused
    2. Track Embedding Cache - For cross-frame association
    3. Delayed Confirmation - Avoid premature ID allocation
    """

    def __init__(self):
        self.active_ids: Set[int] = set()
        self.retired_ids: Set[int] = set()
        self.next_id: int = 1
        self.track_embeddings: Dict[int, np.ndarray] = {}

    def assign_id(self, detection: Dict, candidates: List[Trajectory],
                  context: Dict) -> Tuple[int, str]:
        """Assign ID"""
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
        """Allocate new ID"""
        for candidate_id in range(1, self.next_id + 1):
            if candidate_id not in self.active_ids and candidate_id not in self.retired_ids:
                return candidate_id
        self.next_id += 1
        return self.next_id

    def retire_id(self, track_id: int, reason: str = "exited"):
        """Retire ID"""
        if track_id in self.active_ids:
            self.active_ids.remove(track_id)
            if reason == "exited":
                self.retired_ids.add(track_id)

    def _select_best_candidate(self, detection: Dict, candidates: List[Trajectory],
                                context: Dict) -> Trajectory:
        """Select best candidate trajectory"""
        scores = []
        for track in candidates:
            score = self._compute_match_score(detection, track, context)
            scores.append((track, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _compute_match_score(self, detection: Dict, track: Trajectory, context: Dict) -> float:
        """Compute matching score"""
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


# ==================== LLM Optimizer ====================

class LLMOptimizer:
    """
    LLM Optimizer - Enhanced Version

    Call LLM only at critical decision points to avoid frequent calls
    Core optimizations:
    1. Lane count conservation analysis
    2. Occluded target inference
    3. ID conflict resolution
    4. ID jump analysis (new)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 progress_callback: Optional[callable] = None):
        self.llm_client = llm_client
        self.cache: Dict[str, Any] = {}
        self.call_count = 0
        # Lane-level statistics
        self.lane_stats: Dict[str, Dict] = {}
        # Progress callback (for pushing inference to frontend)
        self.progress_callback = progress_callback
        # ID jump analysis cache
        self._id_jump_cache = {}
        # ID jump correction records: {frame_id: {lost_track_id: jump_to_track_id}}
        self.id_jump_corrections: Dict[int, Dict[int, int]] = {}

    def _notify_progress(self, event_type: str, data: Dict):
        """Send progress notification"""
        if self.progress_callback:
            self.progress_callback(event_type, data)

    def should_call_llm(self, situation: str, context: Dict) -> bool:
        """Determine if LLM should be called"""
        if not self.llm_client:
            return False

        # Rule layer can handle normal matching
        if situation == "normal_match":
            return False

        count_diff = context.get("count_diff", 0)
        if situation == "count_mismatch" and abs(count_diff) <= 1:
            return False

        # Call LLM for complex cases
        llm_situations = [
            "count_mismatch_large",
            "track_reappear",
            "id_conflict",
            "new_object_source",
            "lane_transition",
            "occlusion_analysis",
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
        Analyze lane count conservation

        Principle: Vehicle count in the same lane should be consistent across frames
        Call LLM only when difference exceeds threshold
        """
        cache_key = f"lane_conservation_{lane_id}_{prev_frame_id}_{curr_frame_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prev_count = len(prev_tracks)
        curr_count = len(curr_detections)
        diff = curr_count - prev_count

        # Small changes handled by rule layer
        if abs(diff) <= 1:
            if diff < 0:
                return {"cause": "exit", "confidence": 0.7, "affected_ids": [],
                        "reasoning": "Target exited lane", "action": "keep_tracks"}
            else:
                return {"cause": "enter", "confidence": 0.6, "affected_ids": [],
                        "reasoning": "New target entered lane", "action": "create_new"}

        # Large changes need LLM analysis
        if not self.llm_client:
            return self._rule_based_lane_analysis(lane_id, prev_tracks, curr_detections, diff)

        # Notify frontend: starting lane analysis
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
            # Notify frontend: LLM thinking
            self._notify_progress("llm_thinking", {
                "analysis_type": "lane_count_conservation",
                "lane_id": lane_id,
                "prompt_preview": prompt[:200] + "..."
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # Notify frontend: analysis result
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
        """Build lane conservation prompt - Simplified version"""
        prev_count = len(prev_tracks)
        curr_count = len(curr_detections)
        diff = curr_count - prev_count

        prompt = f"""[Lane Count Analysis]
Lane: {lane_id}, Frames: {prev_frame_id}->{curr_frame_id}
Vehicles: {prev_count}->{curr_count} ({diff:+d})

[Previous Frame Tracks] (max 5)"""
        for track in prev_tracks[:5]:
            pos = track.positions[-1] if track.positions else [0, 0, 0]
            prompt += f"\n- ID{track.track_id}: ({pos[0]:.1f},{pos[1]:.1f})"

        prompt += f"\n\n[Current Frame Detections] (max 5)"
        for i, det in enumerate(curr_detections[:5]):
            pos = det.get('location', [0, 0, 0])
            prompt += f"\n- Detection{i}: ({pos[0]:.1f},{pos[1]:.1f})"

        prompt += """

Analyze count change cause:
- miss: missed detection (occlusion)
- false: false detection
- exit: exited
- enter: entered

Return JSON: {"cause":"miss/false/exit/enter", "confidence":0.0-1.0, "action":"keep/remove/interpolate", "reasoning":"explanation"}
"""
        return prompt

    def _rule_based_lane_analysis(self, lane_id: str,
                                   prev_tracks: List[LaneConstrainedTrack],
                                   curr_detections: List[Dict],
                                   diff: int) -> Dict:
        """Rule-based lane analysis"""
        if diff < -1:  # Count decreased
            # Check if predicted track positions exist but weren't matched
            unmatched_tracks = [t for t in prev_tracks if t.lost_count < 5]
            if len(unmatched_tracks) > 0:
                return {
                    "cause": "miss",
                    "confidence": 0.6,
                    "affected_track_ids": [t.track_id for t in unmatched_tracks],
                    "action": "interpolate",
                    "reasoning": f"{len(unmatched_tracks)} tracks predicted but not detected, possible occlusion"
                }
            return {"cause": "exit", "confidence": 0.7, "affected_track_ids": [],
                    "action": "keep_tracks", "reasoning": "Target exited normally"}
        else:  # Count increased
            return {"cause": "enter", "confidence": 0.6, "affected_track_ids": [],
                    "action": "create_new", "reasoning": "New target entered"}

    def analyze_occlusion(self,
                          lost_track: LaneConstrainedTrack,
                          nearby_tracks: List[LaneConstrainedTrack],
                          curr_detections: List[Dict],
                          map_api: Optional[Any] = None) -> Dict:
        """
        Analyze occlusion

        When track is lost, determine if occluded by other vehicles
        """
        cache_key = f"occlusion_{lost_track.track_id}_{lost_track.last_update_frame}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_occlusion(lost_track, nearby_tracks, curr_detections)

        # Notify frontend: starting occlusion analysis
        self._notify_progress("occlusion_analysis_start", {
            "track_id": lost_track.track_id,
            "lost_frames": lost_track.lost_count,
            "lane_id": lost_track.lane_id,
            "nearby_count": len(nearby_tracks)
        })

        prompt = f"""[Occlusion Analysis]
Track ID: {lost_track.track_id}
Last Position: {lost_track.positions[-1][:2] if lost_track.positions else 'N/A'}
Predicted Position: {lost_track.predicted_pos[:2] if lost_track.predicted_pos else 'N/A'}
Lost Frames: {lost_track.lost_count}

[Nearby Tracks] (max 3)"""
        for track in nearby_tracks[:3]:
            pos = track.positions[-1] if track.positions else [0, 0, 0]
            prompt += f"\n- ID{track.track_id}: ({pos[0]:.1f},{pos[1]:.1f})"

        prompt += f"\n\n[Current Detections] (max 3)"
        for det in curr_detections[:3]:
            pos = det.get('location', [0, 0, 0])
            prompt += f"\n- ({pos[0]:.1f},{pos[1]:.1f})"

        prompt += """

Determine: Is it occluded?
Return JSON: {"is_occluded":true/false, "occluder_id":ID or null, "confidence":0.0-1.0, "action":"keep/interpolate/remove", "reasoning":"explanation"}
"""
        self.call_count += 1
        try:
            # Notify frontend: LLM thinking
            self._notify_progress("llm_thinking", {
                "analysis_type": "occlusion",
                "track_id": lost_track.track_id,
                "prompt_preview": prompt[:200] + "..."
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # Notify frontend: analysis result
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
        """Rule-based occlusion analysis"""
        if not lost_track.predicted_pos:
            return {"is_occluded": False, "action": "remove", "confidence": 0.5}

        pred_pos = np.array(lost_track.predicted_pos[:2])

        # Check if any nearby tracks are near predicted position
        for track in nearby_tracks:
            if track.track_id == lost_track.track_id:
                continue
            if not track.positions:
                continue
            track_pos = np.array(track.positions[-1][:2])
            dist = np.linalg.norm(pred_pos - track_pos)

            # If distance is small (< 5m), possible occlusion
            if dist < 5.0:
                return {
                    "is_occluded": True,
                    "occluder_id": track.track_id,
                    "confidence": 0.7,
                    "predicted_reappear_frames": 3,
                    "action": "keep",
                    "reasoning": f"Predicted position occluded by track {track.track_id}"
                }

        return {
            "is_occluded": False,
            "confidence": 0.5,
            "action": "interpolate",
            "reasoning": "No obvious occluder found"
        }

    def analyze_id_jumping(self,
                           track_id: int,
                           track_history: List[Dict],
                           recent_detections: List[Dict],
                           frame_id: int,
                           track_matched: bool = False,
                           matched_det_pos: Optional[List] = None,
                           other_tracks_info: Optional[List[Dict]] = None,
                           det_to_track_map: Optional[Dict[int, int]] = None,
                           map_api: Optional[Any] = None) -> Dict:
        """
        Analyze ID consistency

        Logic:
        1. Does the track exist in current frame?
           - Yes (DeepSORT matched detection): Check if current vs previous position is reasonable
           - No (track lost): Determine if other IDs might be this track's current position

        Args:
            track_id: Track ID
            track_history: Track history (last 5-10 frames)
            recent_detections: Current frame detections (within 30m)
            frame_id: Current frame
            track_matched: Whether track was matched by DeepSORT
            matched_det_pos: Matched detection position (if track_matched=True)
            other_tracks_info: Other track info [{"track_id": int, "pos": [x,y], "matched": bool}, ...]
            det_to_track_map: Detection index to track ID mapping {det_idx: track_id}
            map_api: Map API

        Returns:
            Analysis result with inference time
        """
        import time
        start_time = time.time()

        cache_key = f"id_jump_{track_id}_{frame_id}"
        if cache_key in self._id_jump_cache:
            cached_result = self._id_jump_cache[cache_key]
            cached_result['inference_time'] = 0.0
            return cached_result

        if not self.llm_client:
            result = self._rule_based_id_judge(track_history, recent_detections)
            result['inference_time'] = 0.0
            return result

        self._notify_progress("id_analysis_start", {
            "track_id": track_id,
            "frame_id": frame_id,
            "track_matched": track_matched,
            "history_length": len(track_history)
        })

        # Build track info
        last_pos = track_history[-1].get('pos', [0, 0]) if track_history else [0, 0]
        prev_pos = track_history[-2].get('pos', [0, 0]) if len(track_history) >= 2 else last_pos

        # Calculate track movement distance (previous to current frame)
        track_movement = np.linalg.norm(np.array(last_pos[:2]) - np.array(prev_pos[:2]))

        # Build complete analysis prompt
        prompt = f"""[ID Consistency Analysis - Frame-to-Frame Tracking Verification]

Track ID: {track_id}
Current Frame: {frame_id}

[Track History] (last {len(track_history)} frames)"""
        for pos in track_history[-5:]:
            p = pos.get('pos', [0, 0])
            prompt += f"\n- Frame{pos.get('frame_id', '?')}: ({p[0]:.1f}, {p[1]:.1f})"

        prompt += f"""

[Previous Frame Position]: ({prev_pos[0]:.1f}, {prev_pos[1]:.1f})
[Track Current Position]: ({last_pos[0]:.1f}, {last_pos[1]:.1f})
[Movement Distance]: {track_movement:.2f}m

[DeepSORT Match Status]: {"Matched" if track_matched else "Unmatched (track lost)"}"""

        if track_matched and matched_det_pos:
            prompt += f"""
[Matched Detection Position]: ({matched_det_pos[0]:.1f}, {matched_det_pos[1]:.1f})"""

        # Show current frame detections (mark if matched)
        prompt += f"""

[Current Frame Nearby Detections] ({len(recent_detections)} total, within 30m)"""
        for i, det in enumerate(recent_detections[:10]):
            det_pos = det.get('location', [0, 0])
            # Check which track matched this detection
            matched_by = det_to_track_map.get(i, None) if det_to_track_map else None
            match_info = f" -> Matched by ID={matched_by}" if matched_by is not None else ""
            prompt += f"\n- Detection{i}: ({det_pos[0]:.1f}, {det_pos[1]:.1f}), type={det.get('type', 'Unknown')}{match_info}"

        # Show other track info (for checking ID preemption)
        if other_tracks_info:
            prompt += f"""

[Other Active Tracks] ({len(other_tracks_info)} total)"""
            for t_info in other_tracks_info[:10]:
                tid = t_info.get('track_id')
                tpos = t_info.get('pos', [0, 0])
                tmatched = "Matched" if t_info.get('matched') else "Lost"
                prompt += f"\n- ID{tid}: ({tpos[0]:.1f}, {tpos[1]:.1f}), status={tmatched}"

        # Different analysis tasks based on match status
        if track_matched:
            prompt += """

[Analysis Task - Match Verification]
Track matched by DeepSORT, please verify:
1. Is current position continuous with previous frame? (movement should be < 3m/frame)
2. Is matched detection reasonable?

Possible conclusions:
- "correct": Tracking correct, position continuous
- "error": Tracking error, position jumped, possible false match

Return JSON:
{"decision": "correct or error", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
        else:
            prompt += """

[Analysis Task - Lost Track Analysis]
Track lost in current frame (not matched), please analyze:
1. Did track truly disappear (exited/occluded)?
2. Could other matched track positions be continuation of this track? (check ID preemption)

If other matched track position is close to predicted position, possible ID jump.

Possible conclusions:
- "disappeared": Track truly disappeared (exited/occluded)
- "id_jump_to_X": Track occupied by ID=X, should merge
- "unknown": Cannot determine

Return JSON:
{"decision": "disappeared or id_jump_to_X or unknown", "confidence": 0.0-1.0, "jump_to_id": null or ID number, "reasoning": "brief explanation"}"""

        self.call_count += 1
        try:
            self._notify_progress("llm_thinking", {
                "analysis_type": "id_consistency",
                "track_id": track_id,
                "track_matched": track_matched
            })

            log_debug(f"Calling LLM for track {track_id}, matched={track_matched}...")
            response = self.llm_client.chat_simple(prompt)
            log_debug(f"LLM response for track {track_id}:\n{response}\n{'='*60}")

            result = self._parse_llm_response(response)
            result['inference_time'] = time.time() - start_time
            result['frame_id'] = frame_id
            result['track_matched'] = track_matched

            self._notify_progress("id_analysis_result", {
                "track_id": track_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response,
                "inference_time": result['inference_time']
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            log_error(f"LLM call failed for track {track_id}: {error_detail}")
            self._notify_progress("id_analysis_error", {
                "track_id": track_id,
                "error": error_detail[:500]
            })
            result = self._rule_based_id_judge(track_history, recent_detections)
            result['inference_time'] = 0.0
            result['frame_id'] = frame_id
            result['track_matched'] = track_matched
            result['fallback'] = True

        self._id_jump_cache[cache_key] = result
        return result

    def _rule_based_id_judge(self, track_history: List[Dict],
                              recent_detections: List[Dict]) -> Dict:
        """Rule-based ID judgment"""
        if not track_history or not recent_detections:
            return {"decision": "new_target", "confidence": 0.5, "reasoning": "Insufficient data"}

        # Get last track position
        last_pos = np.array(track_history[-1].get('pos', [0, 0])[:2])

        # Find nearest detection
        min_dist = float('inf')
        best_det_idx = -1
        for i, det in enumerate(recent_detections):
            det_pos = np.array(det.get('location', [0, 0])[:2])
            dist = np.linalg.norm(last_pos - det_pos)
            if dist < min_dist:
                min_dist = dist
                best_det_idx = i

        # Distance less than threshold, consider should keep ID
        if min_dist < 3.0:
            return {
                "decision": "keep_id",
                "confidence": min(0.9, 1.0 - min_dist / 10.0),
                "target_detection_idx": best_det_idx,
                "reasoning": f"Detection {min_dist:.2f}m from track end, should inherit ID"
            }
        else:
            return {
                "decision": "new_target",
                "confidence": min(0.9, 1.0 - 3.0 / min_dist),
                "reasoning": f"Nearest detection {min_dist:.2f}m, may be new target"
            }

    def judge_reappear(self, old_track: Dict, new_detection: Dict,
                       map_topology: Dict) -> Dict:
        """Judge if reappeared target is same trajectory"""
        cache_key = f"reappear_{old_track.get('id')}_{new_detection.get('pos')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_reappear(old_track, new_detection)

        # Check map topology
        old_lane = old_track.get('lane_id')
        new_lane = new_detection.get('lane_id')
        lane_consistent = old_lane == new_lane

        # Check lane connection
        if map_topology and old_lane:
            successors = map_topology.get('successors', {}).get(old_lane, [])
            lane_consistent = lane_consistent or (new_lane in successors)

        prompt = f"""Judge if reappeared target is same trajectory:

Old track info:
- ID: {old_track.get('id')}
- Last position: {old_track.get('last_pos')}
- Last lane: {old_track.get('lane_id')}
- Lost frames: {old_track.get('lost_frames')}
- Predicted position: {old_track.get('predicted_pos')}

New detection info:
- Position: {new_detection.get('pos')}
- Lane: {new_detection.get('lane_id')}
- Type: {new_detection.get('type')}

Lane consistency: {"Yes" if lane_consistent else "No"}
"""
        if map_topology and old_lane:
            prompt += f"- Successors of {old_lane}: {map_topology.get('successors', {}).get(old_lane, [])}\n"

        prompt += """
Return JSON:
{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "reasoning explanation"
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
        """Rule-based reappearance judgment"""
        old_pos = np.array(old_track.get('last_pos', [0, 0, 0])[:2])
        new_pos = np.array(new_detection.get('pos', [0, 0, 0])[:2])
        dist = np.linalg.norm(old_pos - new_pos)
        same_lane = old_track.get('lane_id') == new_detection.get('lane_id')
        lost_frames = old_track.get('lost_frames', 999)

        if dist < 10.0 and same_lane and lost_frames < 10:
            return {"is_same": True, "confidence": 0.8, "reasoning": "Position close and same lane"}
        elif dist < 20.0 and lost_frames < 20:
            return {"is_same": True, "confidence": 0.5, "reasoning": "Position relatively close"}
        else:
            return {"is_same": False, "confidence": 0.6, "reasoning": "Large position difference or lost too long"}

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        return {"cause": "unknown", "confidence": 0.3, "affected_ids": [], "reasoning": "Parse failed"}

    def analyze_track_quality(self,
                               track_id: int,
                               track_history: List[Dict],
                               issues: List[Dict],
                               map_api: Optional[Any] = None) -> Dict:
        """
        Analyze track quality

        Args:
            track_id: Track ID
            track_history: Track history
            issues: Detected issues
            map_api: Map API

        Returns:
            Analysis result with action recommendation
        """
        cache_key = f"track_quality_{track_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_track_quality(issues)

        # Notify frontend: starting track analysis
        self._notify_progress("track_analysis_start", {
            "track_id": track_id,
            "issues": issues,
            "history_length": len(track_history)
        })

        # Build prompt
        prompt = f"""[Track Quality Analysis]

Track ID: {track_id}
Track length: {len(track_history)} frames

[Detected Issues]"""

        for issue in issues:
            prompt += f"""
- {issue.get('type')}: {issue.get('description')}"""

        prompt += f"""

[Track History] (last {len(track_history)} frames)"""
        for i, pos in enumerate(track_history[-5:]):
            prompt += f"""
- Frame {pos.get('frame_id', '?')}: position ({pos.get('pos', [0, 0])[:2]})"""

        prompt += """

Please analyze:
1. Does this track have issues?
2. What is the root cause?
3. What is the recommended handling?

Return JSON:
{
    "action": "keep|merge|remove|interpolate",
    "confidence": 0.0-1.0,
    "reasoning": "reasoning explanation",
    "merge_with": 123  // Only when action=merge
}
"""

        self.call_count += 1
        try:
            # Notify frontend: LLM thinking
            self._notify_progress("llm_thinking", {
                "analysis_type": "track_quality",
                "track_id": track_id,
                "prompt_preview": prompt
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # Notify frontend: analysis result
            self._notify_progress("track_analysis_result", {
                "track_id": track_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            log_error(f"LLM track analysis failed for track {track_id}: {error_detail}")
            self._notify_progress("track_analysis_error", {
                "track_id": track_id,
                "error": error_detail[:500]
            })
            result = self._rule_based_track_quality(issues)

        self.cache[cache_key] = result
        return result

    def _rule_based_track_quality(self, issues: List[Dict]) -> Dict:
        """Rule-based track quality judgment"""
        if not issues:
            return {"action": "keep", "confidence": 0.9, "reasoning": "No obvious issues"}

        # Judge by issue type
        issue_types = [i.get('type') for i in issues]

        if 'speed_anomaly' in issue_types:
            return {"action": "remove", "confidence": 0.7, "reasoning": "Speed anomaly, possible false detection"}

        if 'trajectory_gap' in issue_types:
            return {"action": "interpolate", "confidence": 0.6, "reasoning": "Trajectory discontinuous, recommend interpolation"}

        if 'lost_frames' in issue_types:
            return {"action": "remove", "confidence": 0.5, "reasoning": "Too many lost frames"}

        return {"action": "keep", "confidence": 0.5, "reasoning": "Issue unclear"}

    def analyze_id_jumping_batch(self,
                                  track1_id: int,
                                  track1: TrackedObject,
                                  track2_id: int,
                                  track2: TrackedObject,
                                  jump_info: Dict,
                                  map_api: Optional[Any] = None) -> Dict:
        """
        Batch analysis of ID jumping - Judge if two tracks should merge

        Args:
            track1_id: First track ID
            track1: First track (earlier in time)
            track2_id: Second track ID
            track2: Second track (later in time)
            jump_info: Jump detection info
            map_api: Map API

        Returns:
            Analysis result with should_merge recommendation
        """
        cache_key = f"id_jump_batch_{track1_id}_{track2_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.llm_client:
            return self._rule_based_id_jumping(jump_info)

        # Notify frontend: starting ID jump analysis
        self._notify_progress("id_jumping_analysis_start", {
            "track1_id": track1_id,
            "track2_id": track2_id,
            "time_gap": jump_info.get('time_gap'),
            "distance": jump_info.get('distance')
        })

        # Build track info
        track1_info = {
            'id': track1_id,
            'frame_range': [track1.frame_ids[0], track1.frame_ids[-1]],
            'length': len(track1.positions),
            'end_pos': track1.positions[-1][:2] if track1.positions else [0, 0],
            'end_vel': track1.velocities[-1][:2] if track1.velocities else [0, 0],
        }

        track2_info = {
            'id': track2_id,
            'frame_range': [track2.frame_ids[0], track2.frame_ids[-1]],
            'length': len(track2.positions),
            'start_pos': track2.positions[0][:2] if track2.positions else [0, 0],
            'start_vel': track2.velocities[0][:2] if track2.velocities else [0, 0],
        }

        # Build prompt
        prompt = f"""[ID Jump Analysis]

Detected possible ID jump:

[Track 1] (earlier in time)
- ID: {track1_info['id']}
- Frame range: {track1_info['frame_range'][0]} -> {track1_info['frame_range'][1]}
- Length: {track1_info['length']} frames
- End position: {track1_info['end_pos']}
- End velocity: {track1_info['end_vel']}

[Track 2] (later in time)
- ID: {track2_info['id']}
- Frame range: {track2_info['frame_range'][0]} -> {track2_info['frame_range'][1]}
- Length: {track2_info['length']} frames
- Start position: {track2_info['start_pos']}
- Start velocity: {track2_info['start_vel']}

[Gap Info]
- Time interval: {jump_info.get('time_gap')} frames
- Predicted position: {jump_info.get('predicted_pos')}
- Actual start position: {jump_info.get('track2_start_pos')}
- Position deviation: {jump_info.get('distance', 0):.1f} m

Please analyze:
1. Do these tracks represent the same vehicle?
2. If yes, what is the merge confidence?

Return JSON:
{{
    "should_merge": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "reasoning explanation"
}}
"""

        self.call_count += 1
        try:
            # Notify frontend: LLM thinking
            self._notify_progress("llm_thinking", {
                "analysis_type": "id_jumping",
                "track1_id": track1_id,
                "track2_id": track2_id,
                "prompt_preview": prompt[:200] + "..."
            })

            response = self.llm_client.chat_simple(prompt)
            result = self._parse_llm_response(response)

            # Notify frontend: analysis result
            self._notify_progress("id_jumping_analysis_result", {
                "track1_id": track1_id,
                "track2_id": track2_id,
                "result": result,
                "llm_response": response[:500] if len(response) > 500 else response
            })
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            log_error(f"LLM id_jumping analysis failed: {error_detail}")
            self._notify_progress("id_jumping_analysis_error", {
                "track1_id": track1_id,
                "track2_id": track2_id,
                "error": error_detail[:500]
            })
            result = self._rule_based_id_jumping(jump_info)

        self.cache[cache_key] = result
        return result

    def _rule_based_id_jumping(self, jump_info: Dict) -> Dict:
        """Rule-based ID jump judgment"""
        time_gap = jump_info.get('time_gap', 999)
        distance = jump_info.get('distance', 999)

        # Small time gap and close distance, recommend merge
        if time_gap <= 3 and distance < 5.0:
            return {"should_merge": True, "confidence": 0.8, "reasoning": f"Time gap {time_gap} frames, distance {distance:.1f}m"}
        elif time_gap <= 5 and distance < 8.0:
            return {"should_merge": True, "confidence": 0.6, "reasoning": f"Time gap {time_gap} frames, distance {distance:.1f}m"}
        elif time_gap <= 10 and distance < 10.0:
            return {"should_merge": True, "confidence": 0.4, "reasoning": f"Time gap {time_gap} frames, distance {distance:.1f}m"}
        else:
            return {"should_merge": False, "confidence": 0.7, "reasoning": "Time/distance gap too large"}


# ==================== Main Agent ====================

class TrafficFlowAgent(BaseAgent):
    """
    Traffic Flow Reconstruction Agent

    Supports two modes:
    1. Pure DeepSORT mode (use_llm=False): Uses DeepSORT algorithm for multi-object tracking
    2. LLM hybrid mode (use_llm=True): Rule layer handles normal matching, LLM layer handles difficult cases

    Features:
    - Load detection results
    - Multi-object tracking (DeepSORT)
    - LLM enhanced decision (optional)
    - Reconstruct vehicle trajectories
    - Save tracking results
    """

    def __init__(self, context: AgentContext, use_llm: bool = False):
        """
        Initialize

        Args:
            context: Agent context
            use_llm: Whether to enable LLM optimization (default False)
        """
        super().__init__(context)
        self.name = "traffic_flow_agent"
        self._use_llm = use_llm

        # Core components
        self._loader: Optional[DetectionLoader] = None
        self._tracker: Optional[DeepSORTTracker] = None
        self._trajectories: Dict[int, Trajectory] = {}

        # LLM optimization components (only enabled when use_llm=True)
        self._id_manager: Optional[IDConsistencyManager] = None
        self._llm_optimizer: Optional[LLMOptimizer] = None
        self._lane_history: Dict[str, LaneHistory] = {}
        # LLM inference progress callback
        self._llm_progress_callback = None

        if use_llm and context.llm_client:
            self._id_manager = IDConsistencyManager()
            self._llm_optimizer = LLMOptimizer(context.llm_client, self._notify_llm_progress)
            self.name = "traffic_flow_llm_agent"

        # Statistics
        self._stats = {
            'total_frames': 0,
            'llm_calls': 0,
            'use_llm': use_llm,
        }

    def set_llm_progress_callback(self, callback: callable):
        """Set LLM progress callback"""
        self._llm_progress_callback = callback

    def _notify_llm_progress(self, event_type: str, data: Dict):
        """Notify LLM progress"""
        if self._llm_progress_callback:
            self._llm_progress_callback(event_type, data)

    def get_tools(self) -> List[Dict]:
        """Return traffic flow reconstruction tools"""
        return [
            {
                "name": "load_detection_results",
                "description": "Load detection results",
                "parameters": {
                    "path": {"type": "string", "description": "Detection results directory path"}
                },
                "handler": self._load_detection_results
            },
            {
                "name": "reconstruct_traffic_flow",
                "description": "Reconstruct traffic flow trajectories (DeepSORT tracking)",
                "parameters": {
                    "start_frame": {"type": "integer", "description": "Start frame ID", "default": None},
                    "end_frame": {"type": "integer", "description": "End frame ID", "default": None},
                    "max_distance": {"type": "number", "description": "Maximum matching distance (meters)", "default": 5.0},
                    "max_velocity": {"type": "number", "description": "Maximum velocity (m/s)", "default": 30.0},
                },
                "handler": self._reconstruct_traffic_flow
            },
            {
                "name": "get_trajectory_by_id",
                "description": "Get trajectory by vehicle ID",
                "parameters": {
                    "vehicle_id": {"type": "integer", "description": "Vehicle ID"}
                },
                "handler": self._get_trajectory_by_id
            },
            {
                "name": "save_reconstruction_result",
                "description": "Save reconstruction results",
                "parameters": {
                    "output_path": {"type": "string", "description": "Output file path", "default": "reconstruction_result.json"}
                },
                "handler": self._save_reconstruction_result
            },
            {
                "name": "get_traffic_flow_summary",
                "description": "Get traffic flow summary",
                "parameters": {},
                "handler": self._get_traffic_flow_summary
            },
        ]

    def get_system_prompt(self) -> str:
        mode = "LLM Hybrid" if self._use_llm else "Pure DeepSORT"
        return f"""You are a traffic flow analysis expert using {mode} mode for multi-object tracking.

Core capabilities:
1. Load detection results
2. Reconstruct continuous trajectories using DeepSORT algorithm
3. {"LLM enhanced decision making (handles miss, false, ID conflicts)" if self._use_llm else "Pure position tracking"}
4. Analyze tracking results

Usage:
1. First call load_detection_results to load detection data
2. Then call reconstruct_traffic_flow to reconstruct trajectories
3. Use get_trajectory_by_id to query specific trajectory
4. Use save_reconstruction_result to save results"""

    def _optimize_headings(self, trajectories: Dict[int, Any], frame_data: List[Dict],
                           use_llm: bool = False) -> Dict[int, Dict]:
        """
        优化轨迹中每一帧的朝向（heading）

        基于前后帧位置差计算运动方向，作为真实的朝向参考
        对于噪声较大的帧，可以使用 LLM 进行平滑优化

        Args:
            trajectories: 轨迹数据（按 track_id 索引）
            frame_data: 帧数据列表
            use_llm: 是否使用 LLM 进行平滑优化

        Returns:
            优化后的朝向数据 {track_id: {frame_id: heading}}
        """
        heading_optimized = {}

        for track_id, traj in trajectories.items():
            frame_ids = traj.frame_ids if hasattr(traj, 'frame_ids') else []
            positions = traj.positions if hasattr(traj, 'positions') else []

            if len(frame_ids) < 2:
                continue

            headings = {}
            for i, frame_id in enumerate(frame_ids):
                if i == 0 and len(frame_ids) > 1:
                    # 第一帧：使用第一帧和第二帧的位置差计算运动方向（向前差分）
                    curr_pos = positions[0]
                    next_pos = positions[1]

                    dx = next_pos[0] - curr_pos[0]
                    dy = next_pos[1] - curr_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist < 0.1:  # 静止
                        vel = traj.velocities[i] if i < len(traj.velocities) else [0, 0, 0]
                        heading = math.degrees(math.atan2(vel[1], vel[0])) if vel[0] != 0 or vel[1] != 0 else 0.0
                    else:
                        heading = math.degrees(math.atan2(dy, dx))
                elif i == 0:
                    # 只有一帧的情况：使用 velocity 方向
                    vel = traj.velocities[i] if i < len(traj.velocities) else [0, 0, 0]
                    heading = math.degrees(math.atan2(vel[1], vel[0])) if vel[0] != 0 or vel[1] != 0 else 0.0
                else:
                    # 计算前后帧位置差得到运动方向（向后差分）
                    prev_pos = positions[i - 1]
                    curr_pos = positions[i]

                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist < 0.1:  # 静止
                        # 静止时使用前向速度方向或默认 heading
                        vel = traj.velocities[i] if i < len(traj.velocities) else [0, 0, 0]
                        heading = math.degrees(math.atan2(vel[1], vel[0])) if vel[0] != 0 or vel[1] != 0 else 0.0
                    else:
                        # 运动方向作为朝向
                        heading = math.degrees(math.atan2(dy, dx))

                headings[frame_id] = heading

            # 如果启用 LLM，进行平滑优化
            if use_llm and self._llm_optimizer and self._llm_optimizer.llm_client and len(headings) > 2:
                headings = self._smooth_headings_with_llm(track_id, headings, frame_ids)

            heading_optimized[track_id] = headings

        return heading_optimized

    def _smooth_headings_with_llm(self, track_id: int, headings: Dict[int, float],
                                   frame_ids: List[int]) -> Dict[int, float]:
        """
        使用 LLM 对朝向序列进行平滑优化

        LLM 分析：
        1. 检测异常的朝向跳变（非物理运动）
        2. 基于轨迹平滑性进行修正
        """
        # 构建 LLM 输入
        heading_series = [(fid, headings.get(fid, 0)) for fid in frame_ids]

        prompt = f"""
分析车辆轨迹的朝向序列，识别异常的朝向跳变并进行平滑修正。

Track ID: {track_id}
朝向序列（frame_id: heading_degrees）:
{json.dumps(heading_series, indent=2)}

车辆运动特性：
1. 朝向变化应该是连续的，不会出现突然的 180 度跳变
2. 正常车辆的角速度一般不超过 30 度/帧（假设 10Hz）
3. 如果相邻帧朝向差异 > 90 度，很可能是检测噪声

请分析并返回修正后的朝向序列（JSON 格式）：
{{
    "anomalies": [{{"frame_id": X, "original": Y, "corrected": Z, "reason": "..."}}],
    "smoothed_headings": {{frame_id: heading, ...}}
}}
"""

        try:
            response = self._llm_optimizer.llm_client.chat(prompt)
            result = json.loads(response)
            return result.get('smoothed_headings', headings)
        except Exception:
            # LLM 失败时返回原始数据
            return headings

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process query

        Args:
            query: User query (currently unused, kept for interface matching)
            **kwargs: Additional parameters (detection_path, start_frame, end_frame, etc.)

        Returns:
            Processing result
        """
        _ = query  # Parameter kept for interface matching, currently unused
        # Parse parameters
        detection_path = kwargs.get('detection_path')
        start_frame = kwargs.get('start_frame')
        end_frame = kwargs.get('end_frame')
        output_path = kwargs.get('output_path', 'reconstruction_result.json')

        # If detection path provided, load first
        if detection_path:
            load_result = self._load_detection_results(detection_path)
            if not load_result.get('success'):
                return load_result

        # Execute reconstruction
        if self._loader:
            recon_result = self._reconstruct_traffic_flow(
                start_frame=start_frame,
                end_frame=end_frame
            )
            if not recon_result.get('success'):
                return recon_result

            # Save results
            self._save_reconstruction_result(output_path)

            # Build frame data (for UI visualization)
            frames = self._build_frame_data(start_frame, end_frame)

            # Build trajectory list
            trajectories = []
            for tid, traj in self._trajectories.items():
                states = []
                for i, frame_id in enumerate(traj.frame_ids):
                    vel = traj.velocities[i] if i < len(traj.velocities) else [0, 0, 0]
                    speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2) if len(vel) >= 2 else 0.0
                    states.append({
                        'position': traj.positions[i],
                        'frame_id': frame_id,
                        'velocity': vel,
                        'speed': speed,
                    })
                trajectories.append({
                    'vehicle_id': tid,
                    'vehicle_type': traj.dominant_type,
                    'states': states,
                })

            # 朝向优化：基于前后帧位置差计算运动方向
            heading_optimized = self._optimize_headings(
                self._trajectories,
                frames,
                use_llm=self._use_llm and self._llm_optimizer is not None
            )

            # 将优化后的朝向应用到帧数据和轨迹数据
            for frame in frames:
                frame_id = frame['frame_id']
                for vehicle in frame.get('vehicles', []):
                    track_id = vehicle.get('vehicle_id')
                    if track_id in heading_optimized and frame_id in heading_optimized[track_id]:
                        vehicle['heading'] = heading_optimized[track_id][frame_id]

            for traj_data in trajectories:
                track_id = traj_data['vehicle_id']
                if track_id in heading_optimized:
                    for state in traj_data.get('states', []):
                        frame_id = state['frame_id']
                        if frame_id in heading_optimized[track_id]:
                            state['heading'] = heading_optimized[track_id][frame_id]
                        else:
                            state['heading'] = 0.0

            return {
                "success": True,
                "message": "Traffic flow reconstruction completed",
                "frames": frames,
                "trajectories": trajectories,
                "total_frames": len(frames),
                "total_vehicles": len(self._trajectories),
                "saved_to": output_path,
                "statistics": recon_result.get('statistics', {}),
            }

        return {
            "success": False,
            "error": "Please provide detection results path (detection_path)",
        }

    def _build_frame_data(self, start_frame: Optional[int] = None,
                           end_frame: Optional[int] = None,
                           use_interpolation: bool = True) -> List[Dict]:
        """
        Build frame data (for UI visualization)

        Use DeepSORT tracker's track_id as vehicle ID for continuity
        Support interpolation frames to reduce flicker

        Args:
            start_frame: Start frame ID
            end_frame: End frame ID
            use_interpolation: Whether to use interpolation (reduce flicker)

        Returns:
            Frame data list
        """
        frames: List[Dict] = []

        if not self._loader:
            return frames

        # Get all frame IDs
        frame_ids = self._loader.get_frame_ids()

        for frame_id in frame_ids:
            if start_frame is not None and frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                continue

            # Get frame detection data (don't use DetectionLoader's tracking ID)
            frame_det = self._loader.load_frame(frame_id, use_tracking=False)
            if not frame_det:
                continue

            # Build DeepSORT trajectory position to track_id mapping
            track_id_map = {}  # (x, y) -> track_id
            active_tracks = {}  # track_id -> (position, is_interpolated)

            if self._tracker:
                for track_id, track in self._tracker.tracks.items():
                    if track.state.name != 'DELETED' and track.frame_ids:
                        # Find track position at current frame
                        for i, fid in enumerate(track.frame_ids):
                            if fid == frame_id:
                                pos = tuple(track.positions[i][:2])
                                track_id_map[pos] = track_id
                                active_tracks[track_id] = (list(pos), False)

            # Fill lost targets with interpolated trajectories (reduce flicker)
            if use_interpolation and hasattr(self._tracker, '_interpolated_tracks'):
                for track_id, interp_data in self._tracker._interpolated_tracks.items():
                    for interp in interp_data:
                        if interp['frame_id'] == frame_id and interp.get('is_interpolated'):
                            if track_id not in active_tracks:
                                pos = tuple(interp['position'][:2])
                                active_tracks[track_id] = (list(pos), True)

            # Build vehicle list
            vehicles = []

            # 1. First add vehicles with detection matches
            matched_positions = set()
            for obj in frame_det.objects:
                obj_pos = tuple(obj.location[:2])

                vehicle_id = None

                # Try exact position match
                if obj_pos in track_id_map:
                    vehicle_id = track_id_map[obj_pos]
                    matched_positions.add(obj_pos)
                else:
                    # Try nearest neighbor match (within 1m)
                    for track_pos, track_id in track_id_map.items():
                        dist = ((obj_pos[0] - track_pos[0]) ** 2 +
                               (obj_pos[1] - track_pos[1]) ** 2) ** 0.5
                        if dist < 1.0:
                            vehicle_id = track_id
                            matched_positions.add(track_pos)
                            break

                # If no matching trajectory, use original tracking_id or detection ID
                if vehicle_id is None:
                    vehicle_id = obj.tracking_id if obj.tracking_id > 0 else obj.id

                vehicles.append({
                    'vehicle_id': vehicle_id,
                    'vehicle_type': obj.type,
                    'position': list(obj.location),
                    'heading': obj.heading,
                    'velocity': list(obj.velocity) if hasattr(obj, 'velocity') else [0, 0, 0],
                    'speed': obj.speed,
                    'is_interpolated': False,
                })

            # 2. Add interpolated vehicles (track exists but detection lost)
            for track_id, (pos, is_interp) in active_tracks.items():
                # Check if vehicle already exists at this position
                has_vehicle = False
                for v in vehicles:
                    v_pos = tuple(v['position'][:2])
                    if np.linalg.norm(np.array(v_pos) - np.array(pos)) < 0.5:
                        has_vehicle = True
                        break

                if not has_vehicle and is_interp:
                    # Add interpolated vehicle
                    track = self._tracker.tracks.get(track_id)
                    vehicles.append({
                        'vehicle_id': track_id,
                        'vehicle_type': track.obj_type if track else 'Unknown',
                        'position': list(pos),
                        'heading': 0.0,
                        'velocity': [0, 0, 0],  # 插值车辆速度为 0
                        'speed': 0.0,
                        'is_interpolated': True,
                    })

            frames.append({
                'frame_id': frame_id,
                'vehicles': vehicles,
                'vehicle_count': len(vehicles),
            })

        return frames

    # ==================== Tool Implementations ====================

    def _load_detection_results(self, path: str) -> Dict[str, Any]:
        """Load detection results"""
        self._loader = DetectionLoader(path, enable_tracking=False)
        self._stats['total_frames'] = self._loader.get_frame_count()

        return {
            "success": True,
            "message": "Detection results loaded",
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
        Reconstruct traffic flow - Enhanced version

        Args:
            start_frame: Start frame ID
            end_frame: End frame ID
            max_distance: Maximum matching distance (meters)
            max_velocity: Maximum velocity (m/s)
            use_lane_constraint: Whether to use lane constraint (default True)
            use_interpolation: Whether to use trajectory interpolation (default True)

        Returns:
            Reconstruction result
        """
        if not self._loader:
            return {"success": False, "error": "Please load detection results first"}

        # Load frame data
        frames = self._loader.load_frames(start_frame, end_frame)
        if not frames:
            return {"success": False, "error": "No frame data loaded"}

        # Create enhanced tracker (lane-aware + interpolation)
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

        # Reset state
        self._trajectories = {}
        if self._id_manager:
            self._id_manager = IDConsistencyManager()
        if self._llm_optimizer:
            self._llm_optimizer.cache.clear()
            self._llm_optimizer._id_jump_cache.clear()
            self._llm_optimizer.id_jump_corrections.clear()
        self._lane_history = {}

        # Previous frame lane statistics (for count conservation analysis)
        prev_lane_tracks: Dict[str, List] = {}

        # LLM inference time statistics
        llm_inference_times = []

        # Process each frame - DeepSORT tracking + per-frame LLM ID consistency analysis
        for frame in frames:
            detections = self._prepare_detections(frame)
            self._tracker.update(detections, frame.frame_id)

            # Per-frame LLM ID consistency analysis (with LLM enhancement)
            if self._use_llm and self._llm_optimizer and self.map_api:
                start_time = __import__('time').time()
                self._llm_per_frame_id_consistency_analysis(frame, detections, prev_lane_tracks)
                elapsed = __import__('time').time() - start_time
                llm_inference_times.append(elapsed)

            # Update lane statistics
            if self._use_llm:
                prev_lane_tracks = self._get_lane_constrained_tracks()

        # After tracking, batch LLM analysis of complete trajectories (secondary optimization)
        # Currently commented - only keep frame-to-frame ID consistency analysis
        # if self._use_llm and self._llm_optimizer and self.map_api:
        #     self._llm_batch_analyze_trajectories(frames)

        # Build trajectories (with interpolation frames)
        self._build_trajectories_with_interpolation()

        stats = self._tracker.get_statistics()
        stats['use_llm'] = self._use_llm
        stats['llm_calls'] = self._stats.get('llm_calls', 0)
        stats['use_lane_constraint'] = use_lane_constraint
        stats['use_interpolation'] = use_interpolation

        # LLM inference time statistics
        if llm_inference_times:
            stats['llm_inference_time'] = {
                'total_calls': len(llm_inference_times),
                'total_time': sum(llm_inference_times),
                'avg_time': sum(llm_inference_times) / len(llm_inference_times),
                'min_time': min(llm_inference_times),
                'max_time': max(llm_inference_times),
            }

        # Debug info
        stats['_debug'] = {
            'self._use_llm': self._use_llm,
            'self._llm_optimizer': self._llm_optimizer is not None,
            'self.map_api': self.map_api is not None,
        }

        return {
            "success": True,
            "message": f"Reconstruction completed ({'LLM Enhanced + Lane Constraint' if self._use_llm else 'Pure DeepSORT'})",
            "num_trajectories": len(self._trajectories),
            "statistics": stats,
        }

    def _prepare_detections(self, frame: FrameDetection) -> List[Dict]:
        """Prepare detection data"""
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
        """Get lane-constrained tracks"""
        lane_tracks: Dict[str, List] = {}

        if not self._tracker:
            return lane_tracks

        for track_id, track in self._tracker.tracks.items():
            if track.state == TrackState.DELETED:
                continue

            # Get track's lane ID
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

    def _llm_per_frame_id_consistency_analysis(self, frame: FrameDetection,
                                                detections: List[Dict],
                                                prev_lane_tracks: Dict[str, List]):
        """
        [Enhanced] Per-frame LLM ID consistency analysis - Multi-Agent parallel version

        Core functionality:
        1. ID consistency analysis for each active track
        2. Distinguish whether track is matched by DeepSORT
        3. For lost tracks, check if occupied by other ID (ID jump)
        4. Use multi-Agent parallel LLM calls for speed

        Args:
            frame: Current frame data
            detections: Current frame detections
            prev_lane_tracks: Previous frame lane statistics
        """
        if not self._llm_optimizer or not self.map_api:
            return

        # Collect all active track info first (for "other tracks" info)
        all_tracks_info = []
        for tid, t in self._tracker.tracks.items():
            if t.state == TrackState.DELETED:
                continue
            if not t.positions:
                continue

            # Check if matched (time_since_update == 0 means just updated)
            is_matched = t.time_since_update == 0
            current_pos = t.positions[-1][:2] if t.positions else [0, 0]

            all_tracks_info.append({
                'track_id': tid,
                'pos': current_pos,
                'matched': is_matched,
                'last_update_frame': t.last_update_frame if hasattr(t, 'last_update_frame') else frame.frame_id
            })

        # Prepare all track data for analysis
        analysis_tasks = []
        for track_id, track in self._tracker.tracks.items():
            if track.state == TrackState.DELETED:
                continue
            if not track.positions or len(track.positions) < 2:
                continue  # Track too short, skip analysis

            # Build track history (last 5-10 frames)
            track_history = []
            start_idx = max(0, len(track.positions) - 10)
            for i in range(start_idx, len(track.positions)):
                track_history.append({
                    'frame_id': track.frame_ids[i],
                    'pos': list(track.positions[i]),
                    'confidence': getattr(track, 'confidence', 1.0)
                })

            # Check if track matched by DeepSORT
            track_matched = track.time_since_update == 0

            # Get matched detection position (if matched)
            matched_det_pos = None
            if track_matched and track.positions:
                matched_det_pos = list(track.positions[-1][:2])

            # Get other track info (exclude current track)
            other_tracks_info = [t for t in all_tracks_info if t['track_id'] != track_id]

            # Get track reference position (for filtering nearby detections)
            predicted_pos = track.predicted_location()[:2] if hasattr(track, 'predicted_location') and track.kf_mean is not None else None
            if predicted_pos is None and len(track.positions) >= 2:
                reference_pos = track.positions[-2][:2]  # Use previous frame position
            elif predicted_pos is not None:
                reference_pos = predicted_pos
            else:
                reference_pos = track.positions[-1][:2] if track.positions else [0, 0]

            # Collect detections within 30m, build detection-to-track mapping
            nearby_detections = []
            det_to_track_map = {}  # {index in nearby_detections: matched track ID}

            for det in detections:
                det_pos = det.get('location', [0, 0])[:2]
                dist = np.linalg.norm(np.array(reference_pos) - np.array(det_pos))
                if dist < 30.0:  # Detections within 30m
                    det_idx = len(nearby_detections)
                    nearby_detections.append(det)

                    # Check if detection matched by some track (via position)
                    for t_info in all_tracks_info:
                        if t_info['matched']:
                            tpos = t_info['pos']
                            det_full_pos = det.get('location', [0, 0])
                            if np.linalg.norm(np.array(tpos) - np.array(det_full_pos[:2])) < 0.5:  # Within 0.5m
                                det_to_track_map[det_idx] = t_info['track_id']
                                break

            analysis_tasks.append({
                'track_id': track_id,
                'track_history': track_history,
                'nearby_detections': nearby_detections,
                'track_matched': track_matched,
                'matched_det_pos': matched_det_pos,
                'other_tracks_info': other_tracks_info,
                'det_to_track_map': det_to_track_map,
            })

        if not analysis_tasks:
            return

        # Use multi-Agent parallel LLM calls
        results = self._parallel_llm_analysis(analysis_tasks, frame.frame_id)
        self._stats['llm_calls'] += len(results)

        # Process all results
        for track_id, llm_result in results.items():
            decision = llm_result.get('decision', '')
            track_matched = llm_result.get('track_matched', False)
            confidence = llm_result.get('confidence', 0.0)

            if track_matched:
                # Track matched, verify if correct
                if decision == 'error':
                    log_info(f"[ID Tracking Error] Track {track_id} may be falsely matched: {llm_result.get('reasoning')}")
            else:
                # Track lost, check if occupied by other ID
                if decision.startswith('id_jump_to_'):
                    try:
                        jump_to_id = int(decision.split('_')[-1])
                        log_info(f"[ID Jump Detection] Track {track_id} -> ID={jump_to_id} (confidence={confidence:.2f}): {llm_result.get('reasoning')}")

                        # Record ID jump correction (only if high confidence)
                        if confidence >= 0.9:  # Increased threshold to 0.9, reduce false positives
                            if frame.frame_id not in self._llm_optimizer.id_jump_corrections:
                                self._llm_optimizer.id_jump_corrections[frame.frame_id] = {}
                            self._llm_optimizer.id_jump_corrections[frame.frame_id][track_id] = jump_to_id
                            log_info(f"[ID Jump Record] Correction recorded: Frame{frame.frame_id}, {track_id}->{jump_to_id}")

                    except (ValueError, IndexError):
                        pass
                elif decision == 'disappeared':
                    log_info(f"[Track Disappeared] Track {track_id} disappeared: {llm_result.get('reasoning')}")

    def _parallel_llm_analysis(self, analysis_tasks: List[Dict], frame_id: int) -> Dict[int, Dict]:
        """
        Use multi-Agent parallel LLM calls for ID consistency analysis

        Args:
            analysis_tasks: Analysis task list
            frame_id: Current frame ID

        Returns:
            {track_id: llm_result} dictionary
        """
        results = {}

        # Build Agent prompt list
        agent_prompts = []
        for task in analysis_tasks:
            track_id = task['track_id']
            track_history = task['track_history']
            nearby_detections = task['nearby_detections']
            track_matched = task['track_matched']
            matched_det_pos = task['matched_det_pos']
            other_tracks_info = task['other_tracks_info']
            det_to_track_map = task['det_to_track_map']

            # Build prompt
            last_pos = track_history[-1].get('pos', [0, 0]) if track_history else [0, 0]
            prev_pos = track_history[-2].get('pos', [0, 0]) if len(track_history) >= 2 else last_pos
            track_movement = np.linalg.norm(np.array(last_pos[:2]) - np.array(prev_pos[:2]))

            prompt = f"""[ID Consistency Analysis - Frame-to-Frame Tracking Verification]

Track ID: {track_id}
Current Frame: {frame_id}

[Track History] (last {len(track_history)} frames)"""
            for pos in track_history[-5:]:
                p = pos.get('pos', [0, 0])
                prompt += f"\n- Frame{pos.get('frame_id', '?')}: ({p[0]:.1f}, {p[1]:.1f})"

            prompt += f"""

[Previous Frame Position]: ({prev_pos[0]:.1f}, {prev_pos[1]:.1f})
[Track Current Position]: ({last_pos[0]:.1f}, {last_pos[1]:.1f})
[Movement Distance]: {track_movement:.2f}m

[DeepSORT Match Status]: {"Matched" if track_matched else "Unmatched (track lost)"}"""

            if track_matched and matched_det_pos:
                prompt += f"""
[Matched Detection Position]: ({matched_det_pos[0]:.1f}, {matched_det_pos[1]:.1f})"""

            prompt += f"""

[Current Frame Nearby Detections] ({len(nearby_detections)} total, within 30m)"""
            for i, det in enumerate(nearby_detections[:10]):
                det_pos = det.get('location', [0, 0])
                matched_by = det_to_track_map.get(i, None)
                match_info = f" -> Matched by ID={matched_by}" if matched_by is not None else ""
                prompt += f"\n- Detection{i}: ({det_pos[0]:.1f}, {det_pos[1]:.1f}), type={det.get('type', 'Unknown')}{match_info}"

            if other_tracks_info:
                prompt += f"""

[Other Active Tracks] ({len(other_tracks_info)} total)"""
                for t_info in other_tracks_info[:10]:
                    tid = t_info.get('track_id')
                    tpos = t_info.get('pos', [0, 0])
                    tmatched = "Matched" if t_info.get('matched') else "Lost"
                    prompt += f"\n- ID{tid}: ({tpos[0]:.1f}, {tpos[1]:.1f}), status={tmatched}"

            if track_matched:
                prompt += """

[Analysis Task - Match Verification]
Track matched by DeepSORT, please verify:
1. Is current position continuous with previous frame? (movement should be < 3m/frame)
2. Is matched detection reasonable?

Possible conclusions:
- "correct": Tracking correct, position continuous
- "error": Tracking error, position jumped, possible false match

Return JSON:
{"decision": "correct or error", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
            else:
                prompt += """

[Analysis Task - Lost Track Analysis]
Track lost in current frame (not matched), please analyze:
1. Did track truly disappear (exited/occluded)?
2. Could other matched track positions be continuation of this track? (check ID preemption)

If other matched track position is close to predicted position, possible ID jump.

Possible conclusions:
- "disappeared": Track truly disappeared (exited/occluded)
- "id_jump_to_X": Track occupied by ID=X, should merge
- "unknown": Cannot determine

Return JSON:
{"decision": "disappeared or id_jump_to_X or unknown", "confidence": 0.0-1.0, "jump_to_id": null or ID number, "reasoning": "brief explanation"}"""

            agent_prompts.append((track_id, prompt, track_matched))

        # Parallel LLM calls
        import concurrent.futures

        def parse_llm_response(response: str) -> Dict:
            """Parse LLM response"""
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
            return {"decision": "unknown", "confidence": 0.3, "reasoning": "Parse failed"}

        def call_llm_for_track(track_id: int, prompt: str, track_matched: bool) -> tuple:
            try:
                response = self.llm_client.chat_simple(prompt)
                result = parse_llm_response(response)
                result['track_matched'] = track_matched
                result['frame_id'] = frame_id
                return track_id, result
            except Exception as e:
                log_error(f"LLM call failed for track {track_id}: {e}")
                result = {"decision": "unknown", "confidence": 0.0, "reasoning": str(e), "track_matched": track_matched}
                return track_id, result

        # Use thread pool for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agent_prompts), 32)) as executor:
            futures = {executor.submit(call_llm_for_track, tid, prompt, matched): tid for tid, prompt, matched in agent_prompts}
            for future in concurrent.futures.as_completed(futures):
                try:
                    track_id, result = future.result()
                    results[track_id] = result
                except Exception as e:
                    log_error(f"Agent failed: {e}")

        return results

    def _llm_enhanced_process_with_lanes(self, frame: FrameDetection,
                                          detections: List[Dict],
                                          prev_lane_tracks: Dict[str, List]):
        """
        [Deprecated] LLM enhanced processing - Lane-aware version (per-frame calls)

        Changed to batch analysis after tracking: _llm_batch_analyze_trajectories()
        Restored per-frame ID analysis: _llm_per_frame_id_consistency_analysis()

        Core optimizations:
        1. Lane count conservation analysis
        2. Occluded target inference
        3. ID consistency analysis (prevent ID jump)
        """
        # [Deprecated] This method no longer used, kept for potential rollback
        pass

    def _dict_to_lane_track(self, d: Dict) -> LaneConstrainedTrack:
        """Convert dict to lane-constrained track"""
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
        """Convert TrackedObject to LaneConstrainedTrack"""
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
        """Get lane ID for detection"""
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
        """Get nearby tracks for specified track"""
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
        """Interpolate trajectory"""
        if track_id not in self._tracker.tracks:
            return

        track = self._tracker.tracks[track_id]

        # Use Kalman predicted position for interpolation
        pred_pos = track.predicted_location()
        pred_vel = track.predicted_velocity()

        # Add interpolation data
        track.positions.append(pred_pos.tolist())
        track.velocities.append(pred_vel.tolist())
        track.frame_ids.append(frame_id)

    def _detect_anomalies(self, _frame: FrameDetection, detections: List[Dict]) -> List[Dict]:
        """Detect anomalies"""
        anomalies = []

        # Simplified anomaly detection - based on detection count change
        if self._lane_history:
            for lane_id, history in self._lane_history.items():
                if len(history.count_history) >= 1:
                    prev_count = history.count_history[-1]
                    curr_count = len(detections)  # Simplified: use total detection count
                    diff = curr_count - prev_count

                    if abs(diff) >= 3:  # Large change
                        anomalies.append({
                            "type": "count_mismatch_large",
                            "lane_id": lane_id,
                            "prev_count": prev_count,
                            "curr_count": curr_count,
                            "diff": diff,
                        })

        # Update lane history
        if detections:
            lane_id = "default"
            if lane_id not in self._lane_history:
                self._lane_history[lane_id] = LaneHistory(lane_id=lane_id)
            self._lane_history[lane_id].count_history.append(len(detections))

        return anomalies

    def _llm_batch_analyze_trajectories(self, frames: List[FrameDetection]):
        """
        [New] Batch LLM analysis - Analyze complete trajectories after tracking

        Core optimizations:
        1. Don't call LLM per-frame, batch analyze after tracking
        2. Global analysis based on complete trajectories, more accurate
        3. Only analyze problematic tracks, reduce LLM calls

        Analysis content:
        1. ID jump detection - Identify and fix ID discontinuity (new)
        2. Track segment merging - Identify multiple segments of same vehicle
        3. Anomalous track marking - Mark suspicious tracks (e.g., teleport, speed anomaly)
        """
        log_info(f"\n[LLM Batch Analysis] Starting analysis of {len(self._tracker.tracks)} tracks...")

        tracks = self._tracker.get_active_tracks()
        analysis_results = []

        # 1. [New] First detect and merge ID jumping trajectories
        log_info("[ID Jump Detection] Detecting and merging ID jump trajectories...")
        self._detect_and_merge_id_jumping_trajectories(tracks, frames)

        # 2. Global lane count analysis
        self._llm_analyze_lane_count_conservation(frames)

        # 3. Analyze each track
        for track_id, track in tracks.items():
            if track.state == TrackState.DELETED:
                continue

            # Skip too short tracks
            if len(track.frame_ids) < 3:
                continue

            result = self._llm_analyze_single_track(track_id, track, frames)
            if result:
                analysis_results.append(result)

        # 4. Apply trajectory optimization based on analysis results
        self._apply_llm_analysis_results(analysis_results)

        log_info(f"[LLM Batch Analysis] Completed, analyzed {len(analysis_results)} tracks, LLM calls {self._stats.get('llm_calls', 0)}")

    def _detect_and_merge_id_jumping_trajectories(self, tracks: Dict, frames: List[FrameDetection]):
        """
        [New] Detect and merge ID jumping trajectories

        Core approach:
        1. Find track pairs that don't overlap in time but are in same lane
        2. Check position continuity (end of first track vs start of second track)
        3. Call LLM for suspicious track pairs to analyze if should merge
        """
        if not self._llm_optimizer:
            return

        # Group tracks by lane
        lane_trajectories: Dict[str, List[Tuple[int, TrackedObject]]] = {}
        for track_id, track in tracks.items():
            if track.state == TrackState.DELETED:
                continue
            if len(track.frame_ids) < 2:
                continue

            lane_id = getattr(self._tracker, '_track_lanes', {}).get(track_id, 'unknown')
            if lane_id not in lane_trajectories:
                lane_trajectories[lane_id] = []
            lane_trajectories[lane_id].append((track_id, track))

        # For each lane, detect possible ID jumps
        for lane_id, traj_list in lane_trajectories.items():
            # Sort by start frame
            traj_list.sort(key=lambda x: x[1].frame_ids[0])

            # Check adjacent track pairs
            for i in range(len(traj_list) - 1):
                id1, track1 = traj_list[i]
                id2, track2 = traj_list[i + 1]

                # Check if possible ID jump
                jump_info = self._check_possible_id_jumping(track1, track2)

                if jump_info and jump_info['possible']:
                    # Call LLM to analyze if should merge
                    llm_result = self._llm_optimizer.analyze_id_jumping_batch(
                        track1_id=id1,
                        track1=track1,
                        track2_id=id2,
                        track2=track2,
                        jump_info=jump_info,
                        map_api=self.map_api
                    )

                    self._stats['llm_calls'] += 1

                    # Merge based on LLM recommendation
                    if llm_result.get('should_merge', False):
                        log_info(f"[ID Merge] Merging tracks {id1} and {id2}: {llm_result.get('reasoning')}")
                        self._merge_trajectories(id1, id2)

    def _check_possible_id_jumping(self, track1: TrackedObject, track2: TrackedObject) -> Dict:
        """
        Check if two tracks may be ID jump

        Conditions:
        1. Time non-overlapping (track1 end < track2 start)
        2. Small time gap (< 10 frames)
        3. Close positions (< 10m)
        4. Consistent velocity direction
        """
        # Time check
        end_frame_1 = track1.frame_ids[-1]
        start_frame_2 = track2.frame_ids[0]

        if start_frame_2 <= end_frame_1:
            return {'possible': False, 'reason': 'Time overlap'}

        time_gap = start_frame_2 - end_frame_1
        if time_gap > 10:
            return {'possible': False, 'reason': 'Time gap too large'}

        # Position check
        end_pos_1 = np.array(track1.positions[-1][:2])
        start_pos_2 = np.array(track2.positions[0][:2])

        # Predict track1 end position (based on velocity)
        if len(track1.velocities) > 0:
            vel = np.array(track1.velocities[-1][:2])
            predicted_pos = end_pos_1 + vel * time_gap * 0.1
        else:
            predicted_pos = end_pos_1

        distance = np.linalg.norm(predicted_pos - start_pos_2)

        if distance > 15.0:
            return {'possible': False, 'reason': f'Distance too far: {distance:.1f}m'}

        # Velocity direction check
        if len(track1.velocities) > 0 and len(track2.velocities) > 0:
            vel1 = np.array(track1.velocities[-1][:2])
            vel2 = np.array(track2.velocities[0][:2])

            # Calculate angle between directions
            norm1, norm2 = np.linalg.norm(vel1), np.linalg.norm(vel2)
            if norm1 > 0.1 and norm2 > 0.1:
                cos_sim = np.dot(vel1, vel2) / (norm1 * norm2)
                if cos_sim < 0.5:  # Angle > 60 degrees
                    return {'possible': False, 'reason': 'Velocity direction inconsistent'}

        return {
            'possible': True,
            'time_gap': time_gap,
            'distance': distance,
            'track1_end_pos': end_pos_1.tolist(),
            'track2_start_pos': start_pos_2.tolist(),
            'predicted_pos': predicted_pos.tolist()
        }

    def _llm_analyze_lane_count_conservation(self, frames: List[FrameDetection]):
        """Batch analyze lane count conservation for each lane"""
        if not self._llm_optimizer:
            return

        # Group by lane and count vehicles per frame
        lane_frame_counts: Dict[str, Dict[int, List]] = {}  # lane_id -> frame_id -> track_ids

        for track_id, track in self._tracker.tracks.items():
            if track.state == TrackState.DELETED:
                continue
            lane_id = getattr(self._tracker, '_track_lanes', {}).get(track_id, 'unknown')
            if lane_id not in lane_frame_counts:
                lane_frame_counts[lane_id] = {}

            for frame_id in track.frame_ids:
                if frame_id not in lane_frame_counts[lane_id]:
                    lane_frame_counts[lane_id][frame_id] = []
                lane_frame_counts[lane_id][frame_id].append(track_id)

        # Analyze count changes for each lane
        for lane_id, frame_counts in lane_frame_counts.items():
            sorted_frames = sorted(frame_counts.keys())
            for i in range(1, len(sorted_frames)):
                prev_frame = sorted_frames[i - 1]
                curr_frame = sorted_frames[i]
                prev_count = len(frame_counts[prev_frame])
                curr_count = len(frame_counts[curr_frame])
                diff = curr_count - prev_count

                # Only call LLM for large changes
                if abs(diff) >= 2:
                    self._llm_optimizer.analyze_lane_count_conservation(
                        lane_id=lane_id,
                        prev_frame_id=prev_frame,
                        curr_frame_id=curr_frame,
                        prev_tracks=[],  # No detailed track info needed for batch analysis
                        curr_detections=[],
                        map_api=self.map_api
                    )
                    self._stats['llm_calls'] += 1

    def _llm_analyze_single_track(self, track_id: int, track: TrackedObject,
                                   frames: List[FrameDetection]) -> Optional[Dict]:
        """Analyze single track"""
        if not self._llm_optimizer:
            return None

        # Build track history
        track_history = []
        for i, (pos, frame_id) in enumerate(zip(track.positions[-10:], track.frame_ids[-10:])):
            track_history.append({
                'frame_id': frame_id,
                'pos': list(pos) if hasattr(pos, '__iter__') else pos,
                'confidence': getattr(track, 'confidence', 1.0)
            })

        # Detect track issues
        issues = self._detect_track_issues(track)

        if not issues:
            return None  # No issues, no LLM analysis needed

        # Call LLM analysis
        result = self._llm_optimizer.analyze_track_quality(
            track_id=track_id,
            track_history=track_history,
            issues=issues,
            map_api=self.map_api
        )

        self._stats['llm_calls'] += 1

        return {
            'track_id': track_id,
            'issues': issues,
            'llm_result': result
        }

    def _detect_track_issues(self, track: TrackedObject) -> List[Dict]:
        """Detect track issues"""
        issues = []

        if len(track.positions) < 2:
            return issues

        # 1. Detect speed anomaly
        for i in range(1, len(track.positions)):
            prev_pos = np.array(track.positions[i-1][:2])
            curr_pos = np.array(track.positions[i][:2])
            dist = np.linalg.norm(curr_pos - prev_pos)
            speed = dist / 0.1  # Assuming 0.1s frame interval

            if speed > 30.0:  # Exceeds 30m/s (108km/h)
                issues.append({
                    'type': 'speed_anomaly',
                    'frame_idx': i,
                    'speed': speed,
                    'description': f'Speed anomaly: {speed:.1f} m/s'
                })

        # 2. Detect trajectory discontinuity (jump)
        for i in range(1, len(track.frame_ids)):
            gap = track.frame_ids[i] - track.frame_ids[i-1]
            if gap > 5:  # Frame gap > 5 frames
                issues.append({
                    'type': 'trajectory_gap',
                    'frame_gap': gap,
                    'description': f'Trajectory discontinuous: missing {gap} frames'
                })

        # 3. Detect too many lost frames
        if track.time_since_update > 5:
            issues.append({
                'type': 'lost_frames',
                'lost_count': track.time_since_update,
                'description': f'Too many lost frames: {track.time_since_update}'
            })

        return issues

    def _apply_llm_analysis_results(self, results: List[Dict]):
        """Apply LLM analysis results for trajectory optimization"""
        for result in results:
            track_id = result['track_id']
            llm_result = result.get('llm_result', {})

            # Process based on LLM recommendation
            action = llm_result.get('action')

            if action == 'merge':
                # Merge trajectories (need target track ID)
                target_id = llm_result.get('merge_with')
                if target_id and target_id in self._tracker.tracks:
                    self._merge_trajectories(track_id, target_id)

            elif action == 'remove':
                # Mark for deletion
                if track_id in self._tracker.tracks:
                    self._tracker.tracks[track_id].state = TrackState.DELETED

            elif action == 'interpolate':
                # Interpolate fix
                self._interpolate_track(track_id, 0)

    def _merge_trajectories(self, track_id: int, target_id: int):
        """
        Merge two trajectories (for ID jump fix)

        Merge track_id into target_id, track_id is earlier in time
        """
        if track_id not in self._tracker.tracks or target_id not in self._tracker.tracks:
            return

        target_track = self._tracker.tracks[target_id]
        source_track = self._tracker.tracks[track_id]

        # Check time order: source should be before target
        if source_track.frame_ids[-1] > target_track.frame_ids[0]:
            pass  # Continue execution, handled below
        else:
            # Time order error, swap
            source_track, target_track = target_track, source_track

        # Calculate time gap
        source_end_frame = source_track.frame_ids[-1]
        target_start_frame = target_track.frame_ids[0]
        gap = target_start_frame - source_end_frame - 1

        # If gap exists, interpolate
        if gap > 0:
            self._interpolate_merge_gap(source_track, target_track, gap)

        # Merge data (source first, target after)
        target_track.positions[:0] = source_track.positions
        target_track.velocities[:0] = source_track.velocities
        target_track.frame_ids[:0] = source_track.frame_ids

        # Delete source trajectory
        source_track.state = TrackState.DELETED
        log_info(f"[Track Merge] ID {source_track.track_id} -> {target_track.track_id}, filled {gap} frame gap")

    def _interpolate_merge_gap(self, source_track: TrackedObject, target_track: TrackedObject, gap: int):
        """
        Interpolate gap between two trajectories
        """
        if gap <= 0:
            return

        # Get source track end state
        end_pos = np.array(source_track.positions[-1][:2])
        end_vel = np.array(source_track.velocities[-1][:2]) if source_track.velocities else np.array([0, 0])

        # Get target track start state
        start_pos = np.array(target_track.positions[0][:2])
        start_vel = np.array(target_track.velocities[0][:2]) if target_track.velocities else np.array([0, 0])

        # Linear interpolation for gap frames
        for i in range(gap, 0, -1):
            t = i / (gap + 1)  # Interpolation parameter
            interp_pos = end_pos + (start_pos - end_pos) * t
            interp_vel = end_vel + (start_vel - end_vel) * t

            # Insert at source track end
            source_track.positions.append([interp_pos[0], interp_pos[1], 0])
            source_track.velocities.append([interp_vel[0], interp_vel[1], 0])
            source_track.frame_ids.append(source_track.frame_ids[-1] + 1)

    def _build_trajectories_with_interpolation(self, renumber_ids: bool = True):
        """
        Build trajectories from tracker (with interpolation frames), apply ID jump corrections

        Args:
            renumber_ids: Whether to renumber IDs to consecutive (default True)
        """
        tracks = self._tracker.get_active_tracks()

        # Build raw trajectories first
        raw_trajectories = {}
        for track_id, tracked_obj in tracks.items():
            trajectory = Trajectory(
                track_id=track_id,
                positions=tracked_obj.positions.copy(),
                velocities=tracked_obj.velocities.copy(),
                frame_ids=tracked_obj.frame_ids.copy(),
                obj_types=[tracked_obj.obj_type] * len(tracked_obj.frame_ids),
            )
            raw_trajectories[track_id] = trajectory

        # Apply ID jump corrections
        # [Configurable] Set APPLY_ID_JUMP_CORRECTIONS=False to disable corrections
        apply_corrections = os.getenv("APPLY_ID_JUMP_CORRECTIONS", "true").lower() == "true"
        if apply_corrections:
            self._trajectories = self._apply_id_jump_corrections(raw_trajectories)
        else:
            log_info("[ID Jump Corrections] Disabled, keeping raw trajectories")
            self._trajectories = raw_trajectories

        # Renumber IDs to consecutive
        if renumber_ids:
            self._renumber_trajectory_ids()

    def _renumber_trajectory_ids(self):
        """
        Renumber trajectory IDs to consecutive (starting from 1)

        Sort by trajectory length, longest track gets smallest ID
        """
        if not self._trajectories:
            return

        # Sort tracks by length (longest first gets smallest ID)
        sorted_tracks = sorted(
            self._trajectories.items(),
            key=lambda x: (-len(x[1].frame_ids), x[0])  # By length desc, then ID asc
        )

        # Build new ID mapping
        id_mapping = {}
        for new_id, (old_id, traj) in enumerate(sorted_tracks, start=1):
            id_mapping[old_id] = new_id
            traj.track_id = new_id  # Modify track's own ID

        # Create new trajectory dictionary
        self._trajectories = {new_id: self._trajectories[old_id] for old_id, new_id in id_mapping.items()}

        log_info(f"[ID Renumbering] Completed, {len(id_mapping)} tracks -> ID range 1-{len(id_mapping)}")
        log_debug(f"  ID mapping: {id_mapping}")

    def _apply_id_jump_corrections(self, trajectories: Dict[int, Trajectory]) -> Dict[int, Trajectory]:
        """
        Apply ID jump corrections

        Merge jumped trajectories into original trajectory

        Args:
            trajectories: Original trajectory dictionary

        Returns:
            Corrected trajectory dictionary
        """
        if not self._llm_optimizer or not self._llm_optimizer.id_jump_corrections:
            return trajectories

        log_info(f"\n[ID Jump Corrections] Applying corrections from {len(self._llm_optimizer.id_jump_corrections)} frames...")

        # Build ID mapping: {jump_to_id: lost_id}, meaning jump_to_id should change to lost_id
        id_remapping = {}  # {new_id: original_id}

        for frame_id, corrections in self._llm_optimizer.id_jump_corrections.items():
            for lost_id, jump_to_id in corrections.items():
                # lost_id is lost track, jump_to_id is jumped-to track
                # We need to merge jump_to_id data into lost_id
                if jump_to_id not in id_remapping:
                    id_remapping[jump_to_id] = lost_id
                    log_debug(f"  ID mapping: {jump_to_id} -> {lost_id}")

        if not id_remapping:
            return trajectories

        # Apply corrections
        corrected_trajectories = {}
        processed_ids = set()

        for track_id, traj in trajectories.items():
            if track_id in processed_ids:
                continue

            if track_id in id_remapping:
                # This ID should merge into another ID
                original_id = id_remapping[track_id]

                if original_id in trajectories:
                    original_traj = trajectories[original_id]

                    # Calculate time gap
                    # gap > 0: gap between tracks
                    # gap = 0: tracks consecutive in time
                    # gap < 0: tracks overlap in time (typical ID jump case)
                    original_end = original_traj.frame_ids[-1] if original_traj.frame_ids else -1
                    track_start = traj.frame_ids[0] if traj.frame_ids else -1
                    gap = track_start - original_end - 1

                    # Merge condition: time gap < 5 frames (including overlap, negative gap means overlap)
                    if gap < 5 and gap > -5:
                        # Merge trajectories: add current track data to original track
                        # Handle overlapping frames with deduplication
                        if gap < 0:
                            # Time overlap: only merge non-overlapping parts
                            overlap_frames = -gap  # Overlap frame count
                            for i in range(overlap_frames, len(traj.frame_ids)):
                                fid = traj.frame_ids[i]
                                original_traj.positions.append(traj.positions[i])
                                original_traj.velocities.append(traj.velocities[i])
                                original_traj.frame_ids.append(fid)
                                original_traj.obj_types.append(traj.obj_types[i])
                            log_info(f"  Merge: ID {track_id} -> ID {original_id}, overlap={overlap_frames} frames, added={len(traj.frame_ids) - overlap_frames} frames, merged length={len(original_traj.frame_ids)}")
                        else:
                            # Time consecutive or small gap: direct merge
                            for i, fid in enumerate(traj.frame_ids):
                                original_traj.positions.append(traj.positions[i])
                                original_traj.velocities.append(traj.velocities[i])
                                original_traj.frame_ids.append(fid)
                                original_traj.obj_types.append(traj.obj_types[i])
                            log_info(f"  Merge: ID {track_id} -> ID {original_id}, gap={gap} frames, merged length={len(original_traj.frame_ids)}")

                        corrected_trajectories[original_id] = original_traj
                        processed_ids.add(track_id)
                        processed_ids.add(original_id)
                    else:
                        # Time not consecutive, don't merge, keep both tracks
                        corrected_trajectories[track_id] = traj
                        corrected_trajectories[original_id] = original_traj
                        processed_ids.add(track_id)
                        processed_ids.add(original_id)
                        log_warning(f"  Skip merge: ID {track_id} and ID {original_id} time gap too large ({gap} frames), keeping separate tracks")
                else:
                    # Original track doesn't exist, keep current track
                    corrected_trajectories[track_id] = traj
                    processed_ids.add(track_id)
                    log_warning(f"  Original ID {original_id} doesn't exist, keeping ID {track_id}")
            else:
                # This ID doesn't need correction
                corrected_trajectories[track_id] = traj
                processed_ids.add(track_id)

        log_info(f"[ID Jump Corrections] Completed, corrected track count: {len(corrected_trajectories)}")
        return corrected_trajectories

    def _get_trajectory_by_id(self, vehicle_id: int) -> Dict[str, Any]:
        """Get trajectory by vehicle ID"""
        traj = self._trajectories.get(vehicle_id)
        if not traj:
            return {"success": False, "error": f"Trajectory {vehicle_id} not found"}

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
        """Save reconstruction results"""
        result = self.get_reconstruction_result()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save as CSV table format (columns: track ID, rows: frame ID)
        # Track IDs already renumbered during build, no need to renumber again
        self._save_trajectory_csv(output_path, renumber_track_ids=False)

        return {
            "success": True,
            "message": f"Results saved to {output_path}",
            "path": output_path,
        }

    def _save_trajectory_csv(self, json_output_path: str, renumber_track_ids: bool = True):
        """
        Save trajectory data as CSV table

        Format:
        - Rows: frame ID
        - Columns: track ID
        - Cells: x,y coordinates

        Args:
            json_output_path: JSON output path
            renumber_track_ids: Whether to renumber track IDs to consecutive (default True)
        """
        try:
            import csv
            from pathlib import Path

            # Get output directory
            output_dir = Path(json_output_path).parent
            csv_path = output_dir / "trajectory_table.csv"

            # Collect all frame IDs and track IDs
            all_frame_ids = set()
            original_track_ids = set()
            trajectory_data = {}  # {(frame_id, track_id): (x, y)}

            for tid_str, traj in self._trajectories.items():
                track_id = int(tid_str)
                original_track_ids.add(track_id)
                for i, fid in enumerate(traj.frame_ids):
                    all_frame_ids.add(fid)
                    pos = traj.positions[i]
                    trajectory_data[(fid, track_id)] = (pos[0], pos[1])

            # Sort
            sorted_frames = sorted(all_frame_ids)
            sorted_original_tracks = sorted(original_track_ids)

            # Renumber track IDs to consecutive
            if renumber_track_ids:
                # Build mapping: original ID -> consecutive ID (starting from 1)
                id_mapping = {orig_id: new_id for new_id, orig_id in enumerate(sorted_original_tracks, start=1)}
                # Reverse mapping for header
                sorted_tracks = list(range(1, len(sorted_original_tracks) + 1))
                # Remap data
                remapped_data = {}
                for (fid, orig_tid), pos in trajectory_data.items():
                    remapped_data[(fid, id_mapping[orig_tid])] = pos
                trajectory_data = remapped_data
            else:
                sorted_tracks = sorted_original_tracks

            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Header: frame_id, track_1, track_2, ...
                header = ['frame_id'] + [f'track_{tid}' for tid in sorted_tracks]
                writer.writerow(header)

                # Each row: frame ID and coordinates for each track
                for fid in sorted_frames:
                    row = [fid]
                    for tid in sorted_tracks:
                        pos = trajectory_data.get((fid, tid))
                        if pos:
                            row.append(f"{pos[0]:.4f},{pos[1]:.4f}")
                        else:
                            row.append('')  # Track doesn't exist at this frame
                    writer.writerow(row)

            if renumber_track_ids:
                log_info(f"[Trajectory Table] Saved to {csv_path} ({len(sorted_frames)} frames x {len(sorted_tracks)} tracks), IDs renumbered")
                log_info(f"  ID mapping: {dict(list(id_mapping.items())[:10])}{'...' if len(id_mapping) > 10 else ''}")
            else:
                log_info(f"[Trajectory Table] Saved to {csv_path} ({len(sorted_frames)} frames x {len(sorted_tracks)} tracks)")

        except Exception as e:
            log_error(f"[Trajectory Table] Save failed: {e}")

    def _get_traffic_flow_summary(self) -> Dict[str, Any]:
        """Get traffic flow summary"""
        if not self._trajectories:
            return {"success": False, "error": "Please reconstruct traffic flow first"}

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

    # ==================== Public Methods ====================

    def get_reconstruction_result(self) -> Dict[str, Any]:
        """Get reconstruction result"""
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
        """Get trajectory positions"""
        traj = self._trajectories.get(track_id)
        if traj:
            return np.array(traj.positions)
        return None

    def get_trajectory_at_frame(self, frame_id: int) -> Dict[int, List[float]]:
        """Get all target positions at specified frame"""
        result = {}
        for tid, traj in self._trajectories.items():
            pos = traj.get_position_at_frame(frame_id)
            if pos is not None:
                result[tid] = pos
        return result


# ==================== Convenience Functions ====================

def reconstruct_traffic_flow(frames: List[Dict],
                             max_distance: float = 5.0,
                             max_velocity: float = 30.0,
                             use_llm: bool = False,
                             llm_client: Optional[LLMClient] = None) -> Dict:
    """
    Convenience function: Reconstruct traffic flow

    Args:
        frames: Frame data list, each frame contains 'frame_id' and 'objects'
        max_distance: Maximum matching distance (meters)
        max_velocity: Maximum velocity (m/s)
        use_llm: Whether to enable LLM optimization
        llm_client: LLM client (required when LLM enabled)

    Returns:
        Reconstruction result
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
