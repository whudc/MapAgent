#!/usr/bin/env python3
"""
trafficcentroidTestingScript - triangleGraphComparisoncan

generationComparisonGraph，includetriangleGraph：
- left：Detectionvalue（annotations）
- center：DetectionResult（detections）
- right：TrackingResult（tracking）

UseMethod:
    python tests/test_traffic_flow_gemma4.py --start 0 --end 100
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# SetProjectPath
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter
import imageio
from pathlib import Path
import tempfile
import shutil

from agents.traffic_flow import TrafficFlowAgent
from agents.base import AgentContext
from apis.map_api import MapAPI
from core.llm_client import LLMClient, LLMConfig, LLMProvider


def load_annotation_ground_truth(annotation_dir: Path, frame_id: int) -> Tuple[List[Dict], np.ndarray]:
    """
    LoadvalueandTransformationtoCoordinate

    Args:
        annotation_dir: Directory (data/00/annotations/result_all_V1)
        frame_id:  ID

    Returns:
        (onlist, ego2globalTransformMatrix)
    """
    annotation_file = annotation_dir / f"{frame_id:06d}.json"
    if not annotation_file.exists():
        print(f"Filenotunder: {annotation_file}")
        return [], np.eye(4)

    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = data.get('objects', [])

    # Get ego2global TransformMatrix
    ego2global = np.array(data.get('ego2global_transformation_matrix', np.eye(4)))

    # Transformationa（TransformationtoCoordinate）
    result = []
    for obj in objects:
        #  ego CoordinateTransformationto global Coordinate
        ego_pos = np.array(obj.get('location', [0, 0, 0]) + [1])  # timeCoordinate
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


def load_detection_results(detection_dir: Path, frame_id: int) -> List[Dict]:
    """
    LoadDetectionResult

    Args:
        detection_dir: DetectionResultDirectory (data/json_results)
        frame_id:  ID

    Returns:
        DetectionResultlist
    """
    detection_file = detection_dir / f"00_{frame_id:06d}.json"
    if not detection_file.exists():
        print(f"DetectionFilenotunder: {detection_file}")
        return []

    with open(detection_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get('detections', [])

    # Transformationa
    result = []
    for det in detections:
        pos = det.get('position', {})
        vel = det.get('velocity', {})
        size = det.get('size', {})

        result.append({
            'id': det.get('tracking_id', det.get('id', -1)),
            'type': det.get('class', 'Unknown'),
            'location': [pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)],
            'size': [size.get('length', 4), size.get('width', 2), size.get('height', 1.5)],
            'heading': det.get('heading', 0),
            'velocity': [vel.get('vx', 0), vel.get('vy', 0), 0],
            'score': det.get('score', 1.0),
        })

    return result


def get_tracking_results_for_frame(trajectories: List[Dict], frame_id: int) -> List[Dict]:
    """
    FromTrajectorydatacenterGetTrackingResult

    Args:
        trajectories: Trajectorylist
        frame_id:  ID

    Returns:
        Trackingonlist
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
                })
                break

    return result


def get_color_for_type(obj_type: str) -> Tuple[float, float, float]:
    """RootontypeReturnColor"""
    colors = {
        'Car': (0.2, 0.6, 1.0),        # 
        'Vehicle': (0.2, 0.6, 1.0),
        'Bus': (1.0, 0.6, 0.2),        # 
        'Truck': (1.0, 0.3, 0.3),      # 
        'Pedestrian': (0.3, 0.8, 0.3), # 
        'Cyclist': (0.7, 0.3, 0.9),    # 
        'Motorcycle': (0.7, 0.3, 0.9),
        'Unknown': (0.5, 0.5, 0.5),    # 
    }
    return colors.get(obj_type, (0.5, 0.5, 0.5))


def draw_object(ax, obj: Dict, is_tracking: bool = False):
    """
    underGraphaon

    Args:
        ax: matplotlib axes
        obj: oninfo
        is_tracking: iswhetherisTrackingResult（TrackingResult，Other）
    """
    location = obj.get('location', [0, 0, 0])
    x, y = location[0], -location[1]  # Y 

    obj_type = obj.get('type', 'Unknown')
    color = get_color_for_type(obj_type)

    obj_id = obj.get('id', -1)

    if is_tracking:
        # TrackingResult：Point + ID 
        ax.scatter(x, y, c=[color], s=100, marker='o', edgecolors='white', linewidths=1)
        ax.annotate(f'ID:{obj_id}', (x, y), textcoords="offset points",
                   xytext=(5, 5), fontsize=8, color='white',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    else:
        # Detection/：，notShow ID
        size = obj.get('size', [4, 2, 1.5])
        length, width = size[0], size[1]

        # GetRotationAngle（heading  rotation）
        heading = obj.get('heading', None)
        if heading is None:
            rotation = obj.get('rotation', [0, 0, 0])
            heading = rotation[2] if len(rotation) >= 3 else 0

        # Create
        rect = patches.Rectangle(
            (x - length/2, y - width/2),
            length, width,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.3
        )

        # Rotation
        transform = Affine2D().rotate_around(x, y, heading) + ax.transData
        rect.set_transform(transform)

        ax.add_patch(rect)


def draw_map_lanes(ax, map_api: Optional[MapAPI], alpha: float = 0.3):
    """Maplanes"""
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
            ys = [-c[1] for c in coords]  # Y 
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
    generationsingletriangleGraphComparisonGraph

    Args:
        frame_id:  ID
        gt_objects: valueonlist
        det_objects: DetectionResultonlist
        track_objects: TrackingResultonlist
        map_api: Map API
        output_path: OutputGraphPath
    """
    # CreatetriangleGraph
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Frame {frame_id:06d} Comparison', fontsize=14, color='white')

    # SetaCoordinateRange
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
        # Set
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        # Maplanes
        draw_map_lanes(ax, map_api)

        # on
        for obj in objects:
            draw_object(ax, obj, is_tracking=is_tracking)

        # SetandLabel
        ax.set_title(title, fontsize=12, color='white', pad=10)
        ax.set_xlabel('X (m)', fontsize=10, color='gray')
        ax.set_ylabel('Y (m)', fontsize=10, color='gray')
        ax.tick_params(colors='gray')
        ax.grid(True, alpha=0.2, color='gray')

        # Showoncount
        ax.text(0.02, 0.98, f'Count: {len(objects)}',
               transform=ax.transAxes, fontsize=10, color='cyan',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    plt.tight_layout()

    # Set
    fig.patch.set_facecolor('#0f0f1a')

    # SaveGraph
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
        frames_dir: Directory containing frame images
        trajectories: Trajectory list from reconstruction
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
    
    # Get coordinate range from trajectories
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
    
    # Draw map lanes
    draw_map_lanes(ax, map_api, alpha=0.5)
    
    # Create artist objects for animation
    scatter = ax.scatter([], [], c=[], s=100, marker='o', 
                         edgecolors='white', linewidths=1, alpha=0.8)
    id_annotations = []
    
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
        
        # Draw map lanes
        draw_map_lanes(ax, map_api, alpha=0.3)
        
        # Draw trajectory trails (past positions)
        for track_id, history in traj_history.items():
            past_positions = [(x, y) for fid, x, y, heading in history if fid <= frame_id]
            if len(past_positions) > 1:
                trail_x = [p[0] for p in past_positions[-20:]]  # Last 20 frames
                trail_y = [p[1] for p in past_positions[-20:]]
                ax.plot(trail_x, trail_y, alpha=0.3, linewidth=1)
        
        # Draw current positions with rectangles and heading
        current_ids = []
        current_headings = []

        for track_id, history in traj_history.items():
            for fid, x, y, heading in history:
                if fid == frame_id:
                    current_ids.append(track_id)
                    current_headings.append(heading)
                    # Get color based on trajectory length
                    traj_len = len(history)
                    if traj_len > 50:
                        color = '#3B82F6'  # Blue - long track
                    elif traj_len > 20:
                        color = '#10B981'  # Green - medium track
                    else:
                        color = '#F59E0B'  # Orange - short track

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

        # Add ID annotations
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
    
    # Generate frames
    frame_ids = list(range(start_frame, end_frame + 1))
    total_frames = len(frame_ids)
    
    print(f"  Generating {total_frames} frames...")
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frame_ids, 
                        interval=1000/fps, blit=False)
    
    # Save as GIF
    print(f"  Saving to {output_path}...")
    # Keep reference to anim to prevent garbage collection
    anim.save(str(output_path), writer=PillowWriter(fps=fps))
    
    plt.close(fig)
    print(f"  ✓ GIF saved: {output_path}")
    
    return output_path

def generate_html_index(frames_dir: Path, start_frame: int, end_frame: int, output_path: Path):
    """
    generation HTML Indexpage，haveComparisonGraph

    Args:
        frames_dir: GraphDirectory
        start_frame: 
        end_frame: 
        output_path: Output HTML Path
    """
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
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
    <h1>🚗 Traffic Flow Comparison Visualization</h1>

    <div class="nav">
        <button onclick="prevFrame()">◀ Prev</button>
        <button onclick="nextFrame()">Next ▶</button>
        <button onclick="playAnimation()">▶ Play</button>
        <button onclick="pauseAnimation()">⏸ Pause</button>
        <span style="margin-left: 20px;">Frame: <span id="currentFrame">{start_frame}</span> / {end_frame}</span>
    </div>

    <div class="frame-info" id="frameInfo">Frame {start_frame:06d}</div>

    <div class="image-container">
        <img id="frameImage" src="frames/frame_{start_frame:06d}.png" alt="Frame {start_frame}">
    </div>

    <div class="frame-list" id="frameList">
'''

    # AddGraphinterface
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

            // UpdateGraphheight
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

        // keyNavigation
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

    print(f"  ✓ HTML IndexalreadySave: {output_path}")


def create_gemma4_client(port: int = 8001) -> LLMClient:
    """Create Gemma4 Localmodels"""
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
    """CheckLocalmodelsiswhetherRun"""
    import requests
    try:
        response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="trafficcentroidTesting - triangleGraphComparisoncan")
    parser.add_argument("--frames", type=int, default=None, help="Process")
    parser.add_argument("--start", type=int, default=0, help=" ID")
    parser.add_argument("--end", type=int, default=10, help=" ID")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM Optimization")
    parser.add_argument("--port", type=int, default=8001, help="LocalmodelsService port")
    parser.add_argument("--annotation-dir", type=str,
                       default="data/00/annotations/result_all_V1",
                       help="valueDirectory")
    parser.add_argument("--detection-dir", type=str,
                       default="data/json_results",
                       help="DetectionResultDirectory")
    parser.add_argument("--map-file", type=str,
                       default="data/vector_map.json",
                       help="MapFile")
    parser.add_argument("--output-dir", type=str,
                       default="test_output_gif",
                       help="OutputDirectory")
    parser.add_argument("--gif-fps", type=int, default=10, 
                       help="GIF frames per second")
    parser.add_argument("--no-gif", action="store_true",
                       help="Disable GIF export")
    args = parser.parse_args()

    print("=" * 70)
    print("trafficcentroidTesting - triangleGraphComparisoncan")
    print("=" * 70)

    # CreateOutputDirectory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # SetParameter
    use_llm = not args.no_llm
    start_frame = args.start
    end_frame = args.end

    annotation_dir = Path(args.annotation_dir)
    detection_dir = Path(args.detection_dir)

    print(f"\nConfiguration:")
    print(f"  Directory: {annotation_dir}")
    print(f"  DetectionDirectory: {detection_dir}")
    print(f"  OutputDirectory: {output_dir}")
    print(f"  Range: {start_frame} - {end_frame}")
    print(f"  LLM Optimization: {'Enable (Gemma4)' if use_llm else 'Disable'}")

    # CheckLocalmodels
    if use_llm:
        print(f"\nCheck Gemma4  (port {args.port})...")
        if check_local_model_service(args.port):
            print("  ✓ Gemma4 Runcenter")
        else:
            print("  ✗ Gemma4 notRun!")
            print("  Use DeepSORT ...")
            use_llm = False

    # LoadMap
    print("\nLoadMap...")
    try:
        map_api = MapAPI(map_file=args.map_file)
        print(f"  ✓ MapLoadto: {map_api.map.get_lane_count()} lanes")
    except FileNotFoundError:
        print(f"  ✗ MapFilenotto: {args.map_file}")
        map_api = None

    # Create LLM 
    llm_client = None
    if use_llm:
        try:
            llm_client = create_gemma4_client(args.port)
            print(f"  ✓ Gemma4 Createto")
        except Exception as e:
            print(f"  ✗ Create Gemma4 fail: {e}")
            use_llm = False

    # Create Agent
    print("\nCreate TrafficFlowAgent...")
    context = AgentContext(map_api=map_api, llm_client=llm_client)
    agent = TrafficFlowAgent(context, use_llm=use_llm)
    print(f"  Agent: {agent.name}")
    print(f"  : {'LLM iterationOptimization' if use_llm else ' DeepSORT'}")

    # Executecentroid
    print("\ntrafficcentroid...")
    start_time = time.time()

    result = agent.process(
        query="DetectionResultcentroidtraffic",
        detection_path=detection_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        output_path=str(output_dir / "reconstruction_result.json")
    )

    elapsed_time = time.time() - start_time

    # PrintResult
    print("\n" + "=" * 70)
    print("centroidResult")
    print("=" * 70)

    if result.get('success'):
        trajectories = result.get('trajectories', [])
        total_frames = result.get('total_frames', 0)
        statistics = result.get('statistics', {})

        print(f"  ✓ centroidto!")
        print(f"  : {total_frames}")
        print(f"  Trajectory: {len(trajectories)}")
        print(f"  : {elapsed_time:.2f} seconds")

        # generationComparisonGraph
        print("\ngenerationComparisonGraph...")

        for frame_id in range(start_frame, end_frame + 1):
            # Loadthree types ofdata
            gt_objects, ego2global = load_annotation_ground_truth(annotation_dir, frame_id)
            det_objects = load_detection_results(detection_dir, frame_id)
            track_objects = get_tracking_results_for_frame(trajectories, frame_id)

            # generationComparisonGraph
            img_path = frames_dir / f"frame_{frame_id:06d}.png"

            generate_frame_comparison_image(
                frame_id=frame_id,
                gt_objects=gt_objects,
                det_objects=det_objects,
                track_objects=track_objects,
                map_api=map_api,
                output_path=img_path
            )

            print(f"  ✓ Frame {frame_id:06d}: GT={len(gt_objects)}, Det={len(det_objects)}, Track={len(track_objects)}")

        # generation HTML Index
        print("\ngeneration HTML ...")
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

        # Savedatawill
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
        print(f"  ✗ centroidfail: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 70)
    print("Testingto!")
    print(f"OutputDirectory: {output_dir}")
    print(f"\nViewResult:")
    print(f"  Open {output_dir}/index.html haveComparison")
    print(f"  GraphDirectory: {frames_dir}")
    if not args.no_gif:
        print(f"  GIF Animation: {output_dir}/trajectories.gif")
    print("=" * 70)


if __name__ == "__main__":
    main()