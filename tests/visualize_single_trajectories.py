"""
singleTrajectorycan - TrajectorysinglegenerationGraphvalueComparison
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.traffic_flow import reconstruct_traffic_flow
from agents.deepsort_tracker import DeepSORTTracker
from apis.map_api import MapAPI
from utils.detection_loader import DetectionLoader


def transform_point(point, matrix):
    """PointFromvehiclesCoordinateTransformationtoCoordinate"""
    homo_point = np.array([point[0], point[1], point[2], 1.0])
    transformed = matrix @ homo_point
    return transformed[:3].tolist()


def load_ground_truth(gt_dir: Path, num_frames: int = 50):
    """LoadvaluedataandTransformationCoordinate"""
    gt_files = sorted(gt_dir.glob('*.json'))[:num_frames]

    frames = []
    gt_tracks = {}

    for gt_file in gt_files:
        frame_id = int(gt_file.stem)
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)

        matrix = np.array(gt['ego2global_transformation_matrix']).reshape(4, 4)

        objects = []
        for obj in gt.get('objects', []):
            global_loc = transform_point(obj['location'], matrix)
            objects.append({
                'id': obj['id'],
                'location': global_loc,
                'type': obj['type'],
            })

            gt_id = obj['id']
            if gt_id not in gt_tracks:
                gt_tracks[gt_id] = []
            gt_tracks[gt_id].append((frame_id, global_loc))

        frames.append({
            'frame_id': frame_id,
            'objects': objects,
        })

    return frames, gt_tracks


def load_model_detections(detection_dir: str, num_frames: int = 50):
    """LoadmodelsDetectionResult"""
    loader = DetectionLoader(detection_dir, enable_tracking=False)
    model_frames = loader.load_frames(0, num_frames)

    frames = []
    for frame in model_frames:
        objects = []
        for obj in frame.objects:
            d = obj.to_dict()
            objects.append({
                'id': d['id'],
                'location': d['location'],
                'type': d['type'],
            })
        frames.append({
            'frame_id': frame.frame_id,
            'objects': objects,
        })

    return frames


def load_map(map_path: str = None) -> MapAPI:
    """LoadMap"""
    if map_path is None:
        map_path = Path(__file__).parent.parent / 'data' / 'vector_map.json'

    if Path(map_path).exists():
        return MapAPI(map_file=str(map_path))
    return None


def find_best_matching_gt_track(pred_positions, gt_tracks, max_distance=10.0):
    """
    toPredictTrajectoryMatchingvalueTrajectory

    PointNearbyDistanceforMatching
    """
    if not pred_positions or not gt_tracks:
        return None, float('inf')

    pred_start = np.array(pred_positions[0][:2])

    best_gt_id = None
    best_distance = float('inf')

    for gt_id, gt_positions in gt_tracks.items():
        if len(gt_positions) < 3:
            continue

        # computingPredictPointvaluePointDistance
        gt_start = np.array(gt_positions[0][1][:2])
        distance = np.linalg.norm(pred_start - gt_start)

        if distance < best_distance:
            best_distance = distance
            best_gt_id = gt_id

    if best_distance > max_distance:
        return None, best_distance

    return best_gt_id, best_distance


def visualize_single_trajectory(
    track_id,
    pred_traj,
    gt_track_id,
    gt_positions,
    output_dir: Path,
    map_api: MapAPI = None
):
    """
    cansingleTrajectoryvalueComparison

    Args:
        track_id: PredictTrajectory ID
        pred_traj: PredictTrajectorydata (include positions, frame_ids etc.)
        gt_track_id: MatchingvalueTrajectory ID
        gt_positions: valueTrajectorylocationlist [(frame_id, position), ...]
        output_dir: OutputDirectory
        map_api: Map API (can，lanes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # notificationPredictTrajectorydata
    pred_positions = pred_traj.get('positions', [])
    pred_frames = pred_traj.get('frame_ids', [])
    pred_type = pred_traj.get('type', 'Unknown')
    pred_lane = pred_traj.get('dominant_lane', '')

    if len(pred_positions) < 2:
        return None

    # ColorSolution
    pred_color = '#FF4444'  #  - Predict
    gt_color = '#44AA44'   #  - value

    # ========== Graph 1: TrajectoryComparison ==========
    ax1 = axes[0]

    # PredictTrajectory
    pred_xs = [p[0] for p in pred_positions]
    pred_ys = [p[1] for p in pred_positions]
    ax1.plot(pred_xs, pred_ys, color=pred_color, linewidth=3, label='Predicted', alpha=0.8)
    ax1.scatter(pred_xs[0], pred_ys[0], c=pred_color, s=150, marker='o',
                zorder=5, edgecolors='white', linewidth=2, label='Pred Start')
    ax1.scatter(pred_xs[-1], pred_ys[-1], c=pred_color, s=150, marker='x',
                zorder=5, linewidth=3, label='Pred End')

    # valueTrajectory
    if gt_positions:
        gt_xs = [p[1][0] for p in gt_positions]
        gt_ys = [p[1][1] for p in gt_positions]
        ax1.plot(gt_xs, gt_ys, color=gt_color, linewidth=3, linestyle='--',
                 label=f'GT (ID:{gt_track_id})', alpha=0.8)
        ax1.scatter(gt_xs[0], gt_ys[0], c=gt_color, s=150, marker='o',
                    zorder=5, edgecolors='white', linewidth=2, label='GT Start')
        ax1.scatter(gt_xs[-1], gt_ys[-1], c=gt_color, s=150, marker='x',
                    zorder=5, linewidth=3, label='GT End')

    # lanes (IfhaveMap)
    if map_api:
        _draw_lane_lines(ax1, map_api, alpha=0.3)

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Trajectory Comparison\nTrack ID: {track_id}', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ========== Graph 2: X CoordinateComparison ==========
    ax2 = axes[1]

    if gt_positions:
        # on ID
        gt_frame_dict = {fp[0]: fp[1] for fp in gt_positions}

        pred_frame_xs = pred_frames
        pred_x_vals = [p[0] for p in pred_positions]

        gt_frame_xs = [fp[0] for fp in gt_positions]
        gt_x_vals = [fp[1][0] for fp in gt_positions]

        ax2.plot(pred_frame_xs, pred_x_vals, color=pred_color, linewidth=2,
                 marker='o', markersize=4, label='Predicted X')
        ax2.plot(gt_frame_xs, gt_x_vals, color=gt_color, linewidth=2,
                 linestyle='--', marker='s', markersize=4, label='GT X')

        ax2.set_xlabel('Frame ID', fontsize=12)
        ax2.set_ylabel('X (m)', fontsize=12)
    else:
        ax2.plot(pred_frames, pred_x_vals, color=pred_color, linewidth=2,
                 marker='o', markersize=4, label='Predicted X')
        ax2.set_xlabel('Frame ID', fontsize=12)
        ax2.set_ylabel('X (m)', fontsize=12)

    ax2.set_title(f'X Coordinate Over Time\nTrack Length: {len(pred_positions)} frames', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ========== Graph 3: Y CoordinateComparison ==========
    ax3 = axes[2]

    if gt_positions:
        pred_y_vals = [p[1] for p in pred_positions]
        gt_y_vals = [fp[1][1] for fp in gt_positions]

        ax3.plot(pred_frame_xs, pred_y_vals, color=pred_color, linewidth=2,
                 marker='o', markersize=4, label='Predicted Y')
        ax3.plot(gt_frame_xs, gt_y_vals, color=gt_color, linewidth=2,
                 linestyle='--', marker='s', markersize=4, label='GT Y')

        ax3.set_xlabel('Frame ID', fontsize=12)
        ax3.set_ylabel('Y (m)', fontsize=12)
    else:
        pred_y_vals = [p[1] for p in pred_positions]
        ax3.plot(pred_frames, pred_y_vals, color=pred_color, linewidth=2,
                 marker='o', markersize=4, label='Predicted Y')
        ax3.set_xlabel('Frame ID', fontsize=12)
        ax3.set_ylabel('Y (m)', fontsize=12)

    ax3.set_title(f'Y Coordinate Over Time\nType: {pred_type}', fontsize=12)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # generationFile
    gt_suffix = f"_vs_GT{gt_track_id}" if gt_track_id else "_no_GT_match"
    filename = f"track_{track_id:04d}{gt_suffix}.png"
    output_path = output_dir / "single_trajectories" / filename

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _draw_lane_lines(ax, map_api, alpha=0.5):
    """underaxislanes"""
    try:
        # Gethavelanes
        lane_lines = map_api.map.lane_lines

        for lane_id, lane_data in lane_lines.items():
            coords = lane_data.get('coordinates', [])
            if len(coords) < 2:
                continue

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            lane_type = lane_data.get('type', 'unknown')
            if 'solid' in lane_type:
                color = '#888888'
                linewidth = 2
            elif 'dashed' in lane_type:
                color = '#AAAAAA'
                linewidth = 1.5
            else:
                color = '#CCCCCC'
                linewidth = 1

            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    except Exception:
        pass  # Iffail，past


def visualize_all_trajectories(
    gt_tracks,
    result_with_map,
    output_dir: Path,
    map_api: MapAPI = None,
    min_length: int = 10
):
    """
    haveTrajectorygenerationsinglecanComparisonGraph

    Args:
        gt_tracks: valueTrajectoryDictionary {gt_id: [(frame_id, position), ...]}
        result_with_map: centroidResult (Map)
        output_dir: OutputDirectory
        map_api: Map API
        min_length: MinimumTrajectorylength
    """
    trajectories = result_with_map.get('trajectories', {})

    # CreateOutputDirectory
    single_dir = output_dir / "single_trajectories"
    single_dir.mkdir(exist_ok=True)

    # info
    stats = {
        'total': len(trajectories),
        'with_gt_match': 0,
        'without_gt_match': 0,
        'long_trajectories': 0,
    }

    print(f"\nGenerating single trajectory visualizations...")
    print(f"Total trajectories: {len(trajectories)}")

    for track_id, traj in trajectories.items():
        if traj['length'] < min_length:
            continue

        if traj['length'] >= 20:
            stats['long_trajectories'] += 1

        # MatchingvalueTrajectory
        gt_match, match_distance = find_best_matching_gt_track(
            traj['positions'], gt_tracks, max_distance=15.0
        )

        if gt_match:
            stats['with_gt_match'] += 1
            gt_positions = gt_tracks[gt_match]
        else:
            stats['without_gt_match'] += 1
            gt_positions = None

        # generationcan
        output_path = visualize_single_trajectory(
            track_id=track_id,
            pred_traj=traj,
            gt_track_id=gt_match,
            gt_positions=gt_positions,
            output_dir=output_dir,
            map_api=map_api
        )

        if output_path:
            status = f"GT:{gt_match}" if gt_match else "No GT match"
            print(f"  Track {track_id:4d} | Len: {traj['length']:3d} | {status} | {output_path.name}")

    # generation
    summary_path = output_dir / "single_trajectories" / "visualization_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'trajectories': {
                tid: {
                    'length': traj['length'],
                    'dominant_lane': traj.get('dominant_lane', ''),
                    'type': traj.get('type', 'Unknown'),
                    'has_gt_match': find_best_matching_gt_track(traj['positions'], gt_tracks)[0] is not None
                }
                for tid, traj in trajectories.items()
                if traj['length'] >= min_length
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Visualization Summary")
    print(f"{'='*60}")
    print(f"  Total trajectories:        {stats['total']}")
    print(f"  Trajectories >= {min_length} frames: {stats['with_gt_match'] + stats['without_gt_match']}")
    print(f"  With GT match:             {stats['with_gt_match']}")
    print(f"  Without GT match:          {stats['without_gt_match']}")
    print(f"  Long trajectories (>=20):  {stats['long_trajectories']}")
    print(f"\nOutput directory: {single_dir.absolute()}")
    print(f"Summary file: {summary_path}")


def main():
    print("=" * 70)
    print("Single Trajectory Visualization")
    print("=" * 70)

    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    num_frames = 297
    gt_dir = Path('data/00/annotations/result_all_V1')

    print(f"\nLoading data ({num_frames} frames)...")

    # Load ground truth
    print("  Loading ground truth...")
    gt_frames, gt_tracks = load_ground_truth(gt_dir, num_frames)
    print(f"    GT frames: {len(gt_frames)}, tracks: {len(gt_tracks)}")

    # Load model detections
    print("  Loading model detections...")
    model_frames = load_model_detections('data/json_results', num_frames)
    print(f"    Model frames: {len(model_frames)}")

    # Load map
    print("  Loading map...")
    map_api = load_map()
    if map_api:
        summary = map_api.get_map_summary()
        print(f"    Map loaded: {summary.get('total_lanes', 0)} lanes")
    else:
        print("    Map not found")

    # Run reconstruction with map
    print("\n" + "-" * 70)
    print("Running reconstruction (with map)...")
    print("-" * 70)

    result_with_map = reconstruct_traffic_flow(
        model_frames,
        map_api=map_api,
        max_distance=5.0,
        max_velocity=30.0,
        use_map=True,
    )

    # Visualize all trajectories
    visualize_all_trajectories(
        gt_tracks,
        result_with_map,
        output_dir,
        map_api=map_api,
        min_length=10
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
