"""
canScript

generationdoubleGraphComparison：
1. DetectionResult + Map
2. TrackingResult + Map
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.detection_loader import DetectionLoader
from agents.deepsort_tracker import DeepSORTTracker
from apis.map_api import MapAPI
from models.map_data import Map Loader


def get_vehicle_color(obj_type: str) -> str:
    """RootvehicletypeReturnColor"""
    colors = {
        'Vehicle': 'blue',
        'Bus': 'orange',
        'Truck': 'red',
        'Pedestrian': 'green',
        'Cyclist': 'purple',
        'Unknown': 'gray',
    }
    return colors.get(obj_type, 'gray')


def get_vehicle_size(obj_type: str) -> tuple:
    """RootvehicletypeReturn"""
    sizes = {
        'Vehicle': (4.5, 2.0),
        'Bus': (10.0, 2.5),
        'Truck': (8.0, 2.5),
        'Pedestrian': (0.5, 0.5),
        'Cyclist': (1.8, 0.6),
        'Unknown': (4.0, 1.8),
    }
    return sizes.get(obj_type, (4.0, 1.8))


def get_lane_color(line_color: str) -> str:
    """RootlanesColorReturn matplotlib Color"""
    colors = {
        'yellow': 'gold',
        'white': 'gray',
        'unknown': 'lightgray',
    }
    return colors.get(line_color.lower(), 'lightgray')


def get_lane_style(lane_type: str) -> tuple:
    """RootlanestypeReturn"""
    styles = {
        'solid': ('-', 2),
        'dashed': ('--', 2),
        'double_solid': ('-', 3),
        'double_dashed': ('--', 3),
        'curb': ('-', 4),
        'fence': ('-', 4),
        'no_lane': (':', 1),
    }
    return styles.get(lane_type.lower(), ('-', 2))


def draw_map(ax, map_api: Optional[MapAPI], x_range: tuple = None, y_range: tuple = None):
    """
    Map（lanesBoundary，notCenter）

    Args:
        ax: matplotlib axis
        map_api: Map API
        x_range: X axisRange (min, max)
        y_range: Y axisRange (min, max)
    """
    if map_api is None:
        return

    all_x = []
    all_y = []

    # havelanes（Boundary）
    for lane_id, lane in map_api.map.lane_lines.items():
        coords = lane.coordinates
        if not coords:
            continue

        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        all_x.extend(x_vals)
        all_y.extend(y_vals)

        color = get_lane_color(lane.color)
        linestyle, linewidth = get_lane_style(lane.type)

        ax.plot(x_vals, y_vals, linestyle, color=color, linewidth=linewidth, alpha=0.8)

    # SetCoordinateRange
    if x_range and y_range:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    elif all_x and all_y:
        margin = 20
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


def draw_objects(ax, objects, map_api: Optional[MapAPI], show_id: bool = True,
                 title: str = "", x_range: tuple = None, y_range: tuple = None,
                 prev_frame_objects: Optional[List[Dict]] = None):
    """
    underaxisGoalandMap

    Args:
        ax: matplotlib axis
        objects: Goallist，include location, size, heading, type, id etc.
        map_api: Map API
        show_id: iswhetherShow ID
        title: Graph
        x_range: X axisRange
        y_range: Y axisRange
        prev_frame_objects: beforeaGoallist（computingDirection）
    """
    ax.clear()
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # Map
    draw_map(ax, map_api, x_range, y_range)

    if not objects:
        ax.text(0, 0, 'No objects', ha='center', va='center', fontsize=14)
        return

    # havelocationSetCoordinateRange
    all_x = []
    all_y = []

    # beforealocationDictionary（computingDirection）
    prev_positions = {}
    if prev_frame_objects:
        for obj in prev_frame_objects:
            pos = obj.get('location') or obj.get('position')
            if pos is not None:
                oid = obj.get('id')
                if oid is not None:
                    prev_positions[oid] = np.array(pos[:2])

    for obj in objects:
        pos = obj.get('location') or obj.get('position')
        if pos is None:
            continue

        x, y = pos[0], pos[1]
        all_x.append(x)
        all_y.append(y)

        # Getvehicle
        size = obj.get('size')
        if size is None:
            size = get_vehicle_size(obj.get('type', 'Unknown'))

        obj_type = obj.get('type', 'Unknown')
        color = get_vehicle_color(obj_type)
        oid = obj.get('id')

        # UseVelocityDirectioncomputing heading
        # Detectiondatain heading cannotcan，Use velocity computingDirection
        vx, vy = 0, 0
        vel = obj.get('velocity')
        if vel:
            # SupportedDictionaryandlistTwotype
            if isinstance(vel, dict):
                vx = vel.get('vx', 0)
                vy = vel.get('vy', 0)
            elif isinstance(vel, (list, tuple)) and len(vel) >= 2:
                vx, vy = vel[0], vel[1]

        if abs(vx) > 0.1 or abs(vy) > 0.1:
            heading = np.arctan2(vy, vx)
        else:
            # haveVelocityinfoUse heading
            heading = obj.get('heading', 0.0)

        # representationsvehicle
        length, width = size[0], size[1]

        # Create，RotationCenterunderCenter
        rect = Rectangle(
            (x - length/2, y - width/2),
            length, width,
            angle=np.degrees(heading),
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

        # CenterPoint
        ax.plot(x, y, 'o', color=color, markersize=6)

        # Heading（UselengthDirection 1.5 ）
        end_x = x + np.cos(heading) * length * 1.5
        end_y = y + np.sin(heading) * length * 1.5
        ax.plot([x, end_x], [y, end_y], '-', color=color, linewidth=2)

        # Show ID
        if show_id and oid is not None:
            ax.text(x, y + 1, f"#{oid}",
                   ha='center', va='bottom', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # SetCoordinateRange
    if all_x and all_y and not x_range:
        margin = 10
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


def run_tracker_on_all_frames(detection_dir: str, max_distance: float = 5.0):
    """
    RunTrackingerandReturnhaveTrackingResult

    Returns:
        Dict[frame_id, List[tracked_objects]]
    """
    loader = DetectionLoader(detection_dir, enable_tracking=False)
    frame_count = loader.get_frame_count()
    frames = loader.load_frames(0, frame_count)

    tracker = DeepSORTTracker(
        map_api=None,
        max_distance=max_distance,
        max_velocity=30.0,
        frame_interval=0.1,
        min_hits=2,
        max_misses=30,
        use_map=False,
        max_iou_distance=max_distance,
    )

    all_tracks = {}

    for frame in frames:
        frame_id = frame.frame_id
        detections = []

        for obj in frame.objects:
            d = obj.to_dict()
            pos = d.get('location')
            if pos is not None:
                detections.append({
                    'location': pos,
                    'velocity': d.get('velocity', [0, 0, 0]),
                    'type': d.get('type', 'Unknown'),
                    'heading': d.get('heading', 0.0),
                    'speed': d.get('speed', 0.0),
                })

        # UpdateTrackinger
        tracks = tracker.update(detections, frame_id)

        # SaveCurrentTrackingResult
        frame_tracks = []
        for track_id, track_obj in tracks.items():
            if track_obj.last_position is not None:
                frame_tracks.append({
                    'id': track_id,
                    'location': track_obj.last_position,
                    'velocity': track_obj.velocities[-1] if track_obj.velocities else [0, 0, 0],
                    'type': track_obj.obj_type,
                    'heading': 0.0,
                    'size': get_vehicle_size(track_obj.obj_type),
                })

        all_tracks[frame_id] = frame_tracks

    return all_tracks, frames


def visualize_frame(frame_id: int, det_objects: List[Dict], track_objects: List[Dict],
                   map_api: Optional[MapAPI], output_dir: Path,
                   x_range: tuple = None, y_range: tuple = None):
    """
    cansingleTwoGraph（Detection + Map，Tracking + Map）

    Args:
        frame_id:  ID
        det_objects: DetectionResultlist
        track_objects: TrackingResultlist
        map_api: Map API
        output_dir: OutputDirectory
        x_range: X axisRange
        y_range: Y axisRange
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. DetectionResult + Map
    draw_objects(axes[0], det_objects, map_api, show_id=False,
                 title=f'Frame {frame_id} - Detections',
                 x_range=x_range, y_range=y_range)

    # 2. TrackingResult + Map
    draw_objects(axes[1], track_objects, map_api, show_id=True,
                 title=f'Frame {frame_id} - Tracking',
                 x_range=x_range, y_range=y_range)

    plt.tight_layout()

    # Save
    output_path = output_dir / f'frame_{frame_id:05d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def create_frame_range_visualization(all_frames: List, all_tracks: Dict,
                                     map_api: Optional[MapAPI],
                                     output_dir: Path,
                                     start_frame: int = 0,
                                     end_frame: Optional[int] = None):
    """
    CreateRangecan

    Args:
        all_frames: havedata（include GT andDetection）
        all_tracks: TrackingResultDictionary
        map_api: Map API
        output_dir: OutputDirectory
        start_frame: 
        end_frame: 
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(all_frames)

    print(f"Generating visualizations for frames {start_frame} to {end_frame-1}...")

    for i, frame in enumerate(all_frames[start_frame:end_frame]):
        frame_id = frame.frame_id

        # directrixDetectiondata
        det_objects = []
        for obj in frame.objects:
            d = obj.to_dict()
            det_objects.append({
                'location': d['location'],
                'size': d.get('size', get_vehicle_size(d['type'])),
                'heading': d.get('heading', 0.0),
                'type': d['type'],
                'velocity': d.get('velocity', [0, 0, 0]),
            })

        # GetTrackingResult
        track_objects = all_tracks.get(frame_id, [])

        # can
        output_path = visualize_frame(
            frame_id, det_objects, track_objects, map_api, output_dir
        )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{end_frame - start_frame} frames...")

    print(f"Saved {end_frame - start_frame} frames to {output_dir}")


def main():
    """function"""
    import argparse

    parser = argparse.ArgumentParser(description='Per-frame visualization with map')
    parser.add_argument('--data_dir', type=str, default='data/json_results',
                       help='Detection data directory')
    parser.add_argument('--map_file', type=str, default='data/vector_map.json',
                       help='Vector map file path')
    parser.add_argument('--output_dir', type=str, default='test_output/per_frame_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Start frame ID')
    parser.add_argument('--num_frames', type=int, default=297,
                       help='Number of frames to visualize')
    parser.add_argument('--max_distance', type=float, default=5.0,
                       help='Max matching distance for tracking')

    args = parser.parse_args()

    print("=" * 70)
    print("Per-Frame Visualization (2 subplots: Detections+Map / Tracking+Map)")
    print("=" * 70)

    # LoadMap
    print(f"\nLoading map from {args.map_file}...")
    try:
        map_api = MapAPI(map_file=args.map_file)
        print(f"  Map loaded: {map_api.map.get_lane_count()} lanes, {map_api.map.get_centerline_count()} centerlines")
    except FileNotFoundError:
        print(f"  Warning: Map file not found, proceeding without map")
        map_api = None

    # LoaddataandRunTrackinger
    print(f"\nLoading data from {args.data_dir}...")
    all_tracks, all_frames = run_tracker_on_all_frames(args.data_dir, args.max_distance)
    print(f"  Loaded {len(all_frames)} frames")
    print(f"  Tracking complete: {len(all_tracks)} frames with tracks")

    # Createcan
    output_dir = Path(args.output_dir)
    end_frame = min(args.start_frame + args.num_frames, len(all_frames))

    create_frame_range_visualization(
        all_frames, all_tracks, map_api, output_dir,
        start_frame=args.start_frame,
        end_frame=end_frame
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
