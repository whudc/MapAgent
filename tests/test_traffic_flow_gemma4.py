#!/usr/bin/env python3
"""
Traffic Flow Reconstruction Test Script - Three-Column Comparison Visualization

Generates comparison images with three columns:
- Left: Ground Truth (Annotations)
- Center: Detection Results (from TrafficFlowAgent internal data)
- Right: Tracking Results (from TrafficFlowAgent output)

Usage:
    python tests/test_traffic_flow_gemma4.py --start 0 --end 100
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio
import tempfile
import shutil

from agents.traffic_flow import TrafficFlowAgent
from agents.base import AgentContext
from apis.map_api import MapAPI
from core.llm_client import LLMClient, LLMConfig, LLMProvider


def load_annotation_ground_truth(annotation_dir: Path, frame_id: int) -> Tuple[List[Dict], np.ndarray]:
    """
    Load ground truth annotations and transformation matrix

    Args:
        annotation_dir: Directory path (data/00/annotations/result_all_V1)
        frame_id: Frame ID

    Returns:
        (list of annotation objects, ego2global transformation matrix)
    """
    annotation_file = annotation_dir / f"{frame_id:06d}.json"
    if not annotation_file.exists():
        print(f"File not found: {annotation_file}")
        return [], np.eye(4)

    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = data.get('objects', [])

    # Get ego2global transformation matrix
    ego2global = np.array(data.get('ego2global_transformation_matrix', np.eye(4)))

    # Transform coordinates to global frame
    result = []
    for obj in objects:
        # Transform from ego frame to global frame
        ego_pos = np.array(obj.get('location', [0, 0, 0]) + [1])  # Homogeneous coordinates
        global_pos = ego2global @ ego_pos
        location = global_pos[:3].tolist()

        result.append({
            'id': obj.get('id', -1),
            'type': obj.get('type', 'Unknown'),
            'location': location,
            'size': obj.get('size', [4, 2, 1.5]),
            'rotation': obj.get('rotation', [0, 0, 0]),
            'velocity': obj.get('velocity', [0, 0, 0]),
        })

    return result, ego2global


def get_detection_objects_from_frame(frames: List[Dict], frame_id: int) -> List[Dict]:
    """
    Extract detection objects from frame data returned by TrafficFlowAgent

    This uses the same data that TrafficFlowAgent uses internally.

    Args:
        frames: List of frame data from result['frames']
        frame_id: Frame ID to extract

    Returns:
        List of detection objects for the specified frame
    """
    for frame in frames:
        if frame.get('frame_id') == frame_id:
            vehicles = frame.get('vehicles', [])
            result = []
            for v in vehicles:
                result.append({
                    'id': v.get('vehicle_id', -1),
                    'type': v.get('vehicle_type', 'Unknown'),
                    'location': v.get('position', [0, 0, 0]),
                    'heading': v.get('heading', 0),
                    'size': [4, 2, 1.5],  # Default vehicle size
                })
            return result
    return []


def get_tracking_results_for_frame(trajectories: List[Dict], frame_id: int) -> List[Dict]:
    """
    Get tracking results for a specific frame from trajectory data

    Args:
        trajectories: List of trajectories from result['trajectories']
        frame_id: Frame ID

    Returns:
        List of tracked objects
    """
    result = []

    for traj in trajectories:
        states = traj.get('states', [])
        for state in states:
            if state.get('frame_id') == frame_id:
                pos = state.get('position', [0, 0, 0])
                result.append({
                    'id': traj.get('vehicle_id', -1),
                    'type': traj.get('vehicle_type', 'Unknown'),
                    'location': pos,
                    'heading': state.get('heading', 0),
                })
                break

    return result


def get_color_for_type(obj_type: str) -> Tuple[float, float, float]:
    """Return color based on object type"""
    colors = {
        'Car': (0.2, 0.6, 1.0),        # Blue
        'Vehicle': (0.2, 0.6, 1.0),
        'Bus': (1.0, 0.6, 0.2),        # Orange
        'Truck': (1.0, 0.3, 0.3),      # Red
        'Pedestrian': (0.3, 0.8, 0.3), # Green
        'Cyclist': (0.7, 0.3, 0.9),    # Purple
        'Motorcycle': (0.7, 0.3, 0.9),
        'Unknown': (0.5, 0.5, 0.5),    # Gray
    }
    return colors.get(obj_type, (0.5, 0.5, 0.5))


def draw_object(ax, obj: Dict, is_tracking: bool = False):
    """
    Draw a single object

    Args:
        ax: matplotlib axes
        obj: Object information
        is_tracking: Whether this is a tracking result (tracking results show ID labels)
    """
    location = obj.get('location', [0, 0, 0])
    x, y = location[0], -location[1]  # Y-axis flip

    obj_type = obj.get('type', 'Unknown')
    color = get_color_for_type(obj_type)

    obj_id = obj.get('id', -1)

    if is_tracking:
        # Tracking result: point + ID label
        ax.scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidths=1)
        ax.annotate(f'ID:{obj_id}', (x, y), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    else:
        # Detection/Annotation: draw bounding box
        size = obj.get('size', [4, 2, 1.5])
        length, width = size[0], size[1]

        # Get rotation angle (heading or rotation)
        heading = obj.get('heading', None)
        if heading is None:
            rotation = obj.get('rotation', [0, 0, 0])
            heading = rotation[2] if len(rotation) >= 3 else 0

        # Create rectangle
        rect = patches.Rectangle(
            (x - length/2, y - width/2),
            length, width,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.3
        )

        # Rotate rectangle
        transform = Affine2D().rotate_around(x, y, heading) + ax.transData
        rect.set_transform(transform)

        ax.add_patch(rect)


def draw_map_lanes(ax, map_api: Optional[MapAPI], alpha: float = 0.3):
    """Draw map lane lines"""
    if map_api is None:
        return

    lane_colors = {
        'solid': '#FFD700',
        'dashed': '#90EE90',
        'double_solid': '#FF6B6B',
        'double_dashed': '#87CEEB',
        'bilateral': '#DDA0DD',
        'curb': '#808080',
        'fence': '#8B4513',
        'no_lane': '#CCCCCC',
    }

    for lane_id, lane in map_api.map.lane_lines.items():
        coords = lane.coordinates
        if len(coords) >= 2:
            xs = [c[0] for c in coords]
            ys = [-c[1] for c in coords]  # Y-axis flip
            color = lane_colors.get(lane.type, '#CCCCCC')
            ax.plot(xs, ys, color=color, linewidth=1, alpha=alpha)


def generate_frame_comparison_image(
    frame_id: int,
    gt_objects: List[Dict],
    det_objects: List[Dict],
    track_objects: List[Dict],
    map_api: Optional[MapAPI],
    output_path: Path
):
    """
    Generate three-column comparison image for a single frame

    Args:
        frame_id: Frame ID
        gt_objects: List of ground truth objects
        det_objects: List of detection results
        track_objects: List of tracking results
        map_api: Map API
        output_path: Output image path
    """
    # Create three-column figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Frame {frame_id:06d} Comparison', fontsize=14, color='white')

    # Set coordinate range
    all_x, all_y = [], []
    for obj in gt_objects + det_objects + track_objects:
        loc = obj.get('location', [0, 0, 0])
        all_x.append(loc[0])
        all_y.append(-loc[1])

    if all_x and all_y:
        x_min, x_max = min(all_x) - 20, max(all_x) + 20
        y_min, y_max = min(all_y) - 20, max(all_y) + 20
    else:
        x_min, x_max = 100, 200
        y_min, y_max = -100, -50

    titles = ['Ground Truth (Annotation)', 'Detection Results', 'Tracking Results']
    data_list = [gt_objects, det_objects, track_objects]
    is_tracking_list = [False, False, True]

    for ax, title, objects, is_tracking in zip(axes, titles, data_list, is_tracking_list):
        # Set background
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        # Draw map lane lines
        draw_map_lanes(ax, map_api)

        # Draw objects
        for obj in objects:
            draw_object(ax, obj, is_tracking=is_tracking)

        # Set title and labels
        ax.set_title(title, fontsize=12, color='white', pad=10)
        ax.set_xlabel('X (m)', fontsize=10, color='gray')
        ax.set_ylabel('Y (m)', fontsize=10, color='gray')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')

        # Display object count
        ax.text(0.02, 0.98, f'Count: {len(objects)}',
               transform=ax.transAxes, fontsize=10, color='cyan',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    plt.tight_layout()

    # Set overall background
    fig.patch.set_facecolor('#0f0f1a')

    # Save image
    plt.savefig(output_path, dpi=150, facecolor='#0f0f1a', edgecolor='none')
    plt.close(fig)

    return output_path


def generate_trajectory_gif(
    frames_dir: Path,
    trajectories: List[Dict],
    map_api: Optional[MapAPI],
    start_frame: int,
    end_frame: int,
    output_path: Path,
    fps: int = 10
):
    """
    Generate GIF animation from trajectory data

    Args:
        frames_dir: Frame image directory
        trajectories: List of reconstructed trajectories
        map_api: Map API
        start_frame: Start frame ID
        end_frame: End frame ID
        output_path: Output GIF path
        fps: Frames per second
    """
    print(f"\nGenerating GIF animation ({start_frame}-{end_frame})...")

    # Create figure for animation
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    # Get coordinate range from trajectory data
    all_x, all_y = [], []
    for traj in trajectories:
        for state in traj.get('states', []):
            pos = state.get('position', [0, 0, 0])
            all_x.append(pos[0])
            all_y.append(-pos[1])

    if all_x and all_y:
        x_margin = (max(all_x) - min(all_x)) * 0.2
        y_margin = (max(all_y) - min(all_y)) * 0.2
        x_min, x_max = min(all_x) - x_margin, max(all_x) + x_margin
        y_min, y_max = min(all_y) - y_margin, max(all_y) + y_margin
    else:
        x_min, x_max, y_min, y_max = -50, 150, -100, 50

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray')

    # Draw map lane lines
    draw_map_lanes(ax, map_api, alpha=0.5)

    # Build trajectory history for each frame
    traj_history = {}  # track_id -> list of (frame_id, x, y, heading)
    for traj in trajectories:
        track_id = traj.get('vehicle_id', -1)
        traj_history[track_id] = []
        for state in traj.get('states', []):
            fid = state.get('frame_id', 0)
            pos = state.get('position', [0, 0, 0])
            heading = state.get('heading', 0.0)
            traj_history[track_id].append((fid, pos[0], -pos[1], heading))

    # Get coordinate range
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Animation update function
    def update(frame_id):
        ax.clear()
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_title(f'Frame {frame_id:06d} - Traffic Flow Reconstruction',
                    fontsize=14, color='white', fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=10, color='gray')
        ax.set_ylabel('Y (m)', fontsize=10, color='gray')
        ax.tick_params(colors='gray')

        # Draw map lane lines
        draw_map_lanes(ax, map_api, alpha=0.3)

        # Draw trajectory trails (historical positions)
        for track_id, history in traj_history.items():
            past_positions = [(x, y) for fid, x, y, heading in history if fid <= frame_id]
            if len(past_positions) > 1:
                trail_x = [p[0] for p in past_positions[-20:]]  # Last 20 frames
                trail_y = [p[1] for p in past_positions[-20:]]
                ax.plot(trail_x, trail_y, alpha=0.3, linewidth=1)

        # Draw current positions (rectangle + heading)
        current_ids = []

        for track_id, history in traj_history.items():
            for fid, x, y, heading in history:
                if fid == frame_id:
                    current_ids.append(track_id)
                    # Set color based on trajectory length
                    traj_len = len(history)
                    if traj_len > 50:
                        color = '#3B82F6'  # Blue - long trajectory
                    elif traj_len > 20:
                        color = '#10B981'  # Green - medium trajectory
                    else:
                        color = '#F59E0B'  # Orange - short trajectory

                    # Draw rectangle with heading (vehicle size: 4m x 2m)
                    length, width = 4.0, 2.0
                    rect = patches.Rectangle(
                        (x - length/2, y - width/2),
                        length, width,
                        linewidth=1.5,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.7
                    )
                    transform = Affine2D().rotate_around(x, y, heading) + ax.transData
                    rect.set_transform(transform)
                    ax.add_patch(rect)
                    break

        # Add ID labels
        for track_id, history in traj_history.items():
            for fid, x, y, heading in history:
                if fid == frame_id:
                    traj_len = len(history)
                    ax.annotate(f'ID:{track_id}', (x, y), textcoords="offset points",
                               xytext=(5, 5), fontsize=8, color='white',
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='blue' if traj_len > 20 else 'orange',
                                       alpha=0.7))
                    break

        # Frame info
        ax.text(0.02, 0.98, f'Frame: {frame_id}/{end_frame} | '
                           f'Vehicles: {len(current_ids)}',
               transform=ax.transAxes, fontsize=11, color='cyan',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    # Generate frame sequence
    frame_ids = list(range(start_frame, end_frame + 1))
    total_frames = len(frame_ids)

    print(f"  Generating {total_frames} frames...")

    # Create animation
    anim = FuncAnimation(fig, update, frames=frame_ids,
                        interval=1000/fps, blit=False)

    # Save as GIF
    print(f"  Saving to {output_path}...")
    # Keep anim reference to prevent garbage collection
    anim.save(str(output_path), writer=PillowWriter(fps=fps))

    plt.close(fig)
    print(f"  GIF saved: {output_path}")

    return output_path


def generate_html_index(frames_dir: Path, start_frame: int, end_frame: int, output_path: Path):
    """
    Generate HTML index page with comparison images

    Args:
        frames_dir: Image directory
        start_frame: Start frame
        end_frame: End frame
        output_path: Output HTML path
    """
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Traffic Flow Comparison - Frames {start_frame}-{end_frame}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f1a;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #3b82f6;
        }}
        .nav {{
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 20px;
            background: #1a1a2e;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .nav button {{
            padding: 10px 20px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}
        .nav button:hover {{
            background: #2563eb;
        }}
        .frame-info {{
            text-align: center;
            font-size: 18px;
            margin-bottom: 10px;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            background: #1a1a2e;
            border-radius: 8px;
            padding: 10px;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        .frame-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            padding: 20px;
        }}
        .frame-thumb {{
            width: 120px;
            cursor: pointer;
            border: 2px solid transparent;
            border-radius: 4px;
            transition: border-color 0.3s;
        }}
        .frame-thumb:hover {{
            border-color: #3b82f6;
        }}
        .frame-thumb.active {{
            border-color: #3b82f6;
        }}
    </style>
</head>
<body>
    <h1>Traffic Flow Comparison Visualization</h1>

    <div class="nav">
        <button onclick="prevFrame()">Prev</button>
        <button onclick="nextFrame()">Next</button>
        <button onclick="playAnimation()">Play</button>
        <button onclick="pauseAnimation()">Pause</button>
        <span style="margin-left: 20px;">Frame: <span id="currentFrame">{start_frame}</span> / {end_frame}</span>
    </div>

    <div class="frame-info" id="frameInfo">Frame {start_frame:06d}</div>

    <div class="image-container">
        <img id="frameImage" src="frames/frame_{start_frame:06d}.png" alt="Frame {start_frame}">
    </div>

    <div class="frame-list" id="frameList">
'''

    # Add thumbnail list
    for fid in range(start_frame, end_frame + 1):
        html_content += f'''        <img class="frame-thumb" data-frame="{fid}"
             src="frames/frame_{fid:06d}.png"
             onclick="showFrame({fid})"
             alt="Frame {fid}">
'''

    html_content += '''    </div>

    <script>
        let currentFrame = ''' + str(start_frame) + ''';
        let startFrame = ''' + str(start_frame) + ''';
        let endFrame = ''' + str(end_frame) + ''';
        let isPlaying = false;
        let animationId = null;

        function showFrame(frameId) {
            currentFrame = frameId;
            document.getElementById('frameImage').src = 'frames/frame_' + frameId.toString().padStart(6, '0') + '.png';
            document.getElementById('currentFrame').textContent = frameId;
            document.getElementById('frameInfo').textContent = 'Frame ' + frameId.toString().padStart(6, '0');

            // Update thumbnail highlight
            document.querySelectorAll('.frame-thumb').forEach(t => {
                t.classList.remove('active');
                if (parseInt(t.dataset.frame) === frameId) {
                    t.classList.add('active');
                }
            });
        }

        function prevFrame() {
            if (currentFrame > startFrame) {
                showFrame(currentFrame - 1);
            }
        }

        function nextFrame() {
            if (currentFrame < endFrame) {
                showFrame(currentFrame + 1);
            }
        }

        function playAnimation() {
            if (isPlaying) return;
            isPlaying = true;

            function animate() {
                if (!isPlaying) return;
                nextFrame();
                if (currentFrame >= endFrame) {
                    currentFrame = startFrame;
                }
                animationId = setTimeout(animate, 500);
            }
            animate();
        }

        function pauseAnimation() {
            isPlaying = false;
            if (animationId) {
                clearTimeout(animationId);
                animationId = null;
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') prevFrame();
            if (e.key === 'ArrowRight') nextFrame();
            if (e.key === ' ') {
                e.preventDefault();
                if (isPlaying) pauseAnimation();
                else playAnimation();
            }
        });

        // Initialize
        showFrame(startFrame);
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  HTML index saved: {output_path}")


def create_gemma4_client(port: int = 8001) -> LLMClient:
    """Create Gemma4 local model client"""
    config = LLMConfig(
        provider=LLMProvider.GEMMA4_LOCAL,
        model="Gemma4",
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",
        max_tokens=4096,
        temperature=0.3,
    )
    return LLMClient(config)


def check_local_model_service(port: int = 8001) -> bool:
    """Check if local model service is running"""
    import requests
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Traffic Flow Reconstruction Test - Three-Column Comparison Visualization")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to process")
    parser.add_argument("--start", type=int, default=0, help="Start frame ID")
    parser.add_argument("--end", type=int, default=10, help="End frame ID")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM optimization")
    parser.add_argument("--port", type=int, default=8001, help="Local model service port")
    parser.add_argument("--annotation-dir", type=str,
                       default="data/00/annotations/result_all_V1",
                       help="Ground truth annotation directory")
    parser.add_argument("--detection-dir", type=str,
                       default="data/json_results",
                       help="Detection results directory")
    parser.add_argument("--map-file", type=str,
                       default="data/vector_map.json",
                       help="Map file path")
    parser.add_argument("--output-dir", type=str,
                       default="test_output_gif",
                       help="Output directory")
    parser.add_argument("--gif-fps", type=int, default=10,
                       help="GIF frames per second")
    parser.add_argument("--no-gif", action="store_true",
                       help="Disable GIF export")
    args = parser.parse_args()

    print("=" * 70)
    print("Traffic Flow Reconstruction Test - Three-Column Comparison Visualization")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Set parameters
    use_llm = not args.no_llm
    start_frame = args.start
    end_frame = args.end

    annotation_dir = Path(args.annotation_dir)
    detection_dir = Path(args.detection_dir)

    print(f"\nConfiguration:")
    print(f"  Annotation directory: {annotation_dir}")
    print(f"  Detection directory: {detection_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Frame range: {start_frame} - {end_frame}")
    print(f"  LLM optimization: {'Enabled (Gemma4)' if use_llm else 'Disabled'}")

    # Check local model service
    if use_llm:
        print(f"\nChecking Gemma4 service (port {args.port})...")
        if check_local_model_service(args.port):
            print("  Gemma4 service is running")
        else:
            print("  Gemma4 service is not running!")
            print("  Will use pure DeepSORT mode...")
            use_llm = False

    # Load map
    print("\nLoading map...")
    try:
        map_api = MapAPI(map_file=args.map_file)
        print(f"  Map loaded successfully: {map_api.map.get_lane_count()} lanes")
    except FileNotFoundError:
        print(f"  Map file not found: {args.map_file}")
        map_api = None

    # Create LLM client
    llm_client = None
    if use_llm:
        try:
            llm_client = create_gemma4_client(args.port)
            print(f"  Gemma4 client created successfully")
        except Exception as e:
            print(f"  Failed to create Gemma4 client: {e}")
            use_llm = False

    # Create TrafficFlowAgent
    print("\nCreating TrafficFlowAgent...")
    context = AgentContext(map_api=map_api, llm_client=llm_client)
    agent = TrafficFlowAgent(context, use_llm=use_llm)
    print(f"  Agent name: {agent.name}")
    print(f"  Mode: {'LLM Hybrid Optimization' if use_llm else 'Pure DeepSORT'}")

    # Execute traffic flow reconstruction
    # Note: TrafficFlowAgent internally creates DetectionLoader and manages all tracking
    print("\nExecuting traffic flow reconstruction...")
    start_time = time.time()

    result = agent.process(
        query="Reconstruct traffic flow",
        detection_path=str(detection_dir),
        start_frame=start_frame,
        end_frame=end_frame,
        output_path=str(output_dir / "reconstruction_result.json")
    )

    elapsed_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("Reconstruction Results")
    print("=" * 70)

    if result.get('success'):
        # Get data from result (same as main code uses)
        trajectories = result.get('trajectories', [])
        frames = result.get('frames', [])  # Frame data from TrafficFlowAgent internal
        total_frames = result.get('total_frames', 0)
        statistics = result.get('statistics', {})

        print(f"  Reconstruction completed!")
        print(f"  Total frames: {total_frames}")
        print(f"  Total trajectories: {len(trajectories)}")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")

        # Generate comparison images
        print("\nGenerating comparison images...")

        for frame_id in range(start_frame, end_frame + 1):
            # Load ground truth from annotation files
            gt_objects, ego2global = load_annotation_ground_truth(annotation_dir, frame_id)

            # Get detection results from TrafficFlowAgent internal frame data
            # This is consistent with main code - uses the same data
            det_objects = get_detection_objects_from_frame(frames, frame_id)

            # Get tracking results from trajectories
            track_objects = get_tracking_results_for_frame(trajectories, frame_id)

            # Generate comparison image
            img_path = frames_dir / f"frame_{frame_id:06d}.png"

            generate_frame_comparison_image(
                frame_id=frame_id,
                gt_objects=gt_objects,
                det_objects=det_objects,
                track_objects=track_objects,
                map_api=map_api,
                output_path=img_path
            )

            print(f"  Frame {frame_id:06d}: GT={len(gt_objects)}, Det={len(det_objects)}, Track={len(track_objects)}")

        # Generate HTML index
        print("\nGenerating HTML index page...")
        html_path = output_dir / "index.html"
        generate_html_index(frames_dir, start_frame, end_frame, html_path)

        # Generate GIF animation
        if not args.no_gif:
            print("\nGenerating GIF animation...")
            gif_path = output_dir / "trajectories.gif"
            generate_trajectory_gif(
                frames_dir=frames_dir,
                trajectories=trajectories,
                map_api=map_api,
                start_frame=start_frame,
                end_frame=end_frame,
                output_path=gif_path,
                fps=args.gif_fps
            )

        # Save data and summary
        summary = {
            'success': True,
            'total_frames': total_frames,
            'total_trajectories': len(trajectories),
            'elapsed_time': elapsed_time,
            'use_llm': use_llm,
            'gif_exported': not args.no_gif,
            'gif_fps': args.gif_fps,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'statistics': statistics,
        }

        with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    else:
        print(f"  Reconstruction failed: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print(f"Output directory: {output_dir}")
    print(f"\nView results:")
    print(f"  Open {output_dir}/index.html to view comparison images")
    print(f"  Image directory: {frames_dir}")
    if not args.no_gif:
        print(f"  GIF animation: {output_dir}/trajectories.gif")
    print("=" * 70)


if __name__ == "__main__":
    main()
