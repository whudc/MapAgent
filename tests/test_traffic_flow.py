"""
交通流重建测试

测试基于位置跟踪的轨迹重建
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.traffic_flow import TrafficFlowAgent, reconstruct_traffic_flow
from agents.position_tracker import PositionTracker
from utils.detection_loader import DetectionLoader


def transform_point(point, matrix):
    """将点从自车坐标系转换到世界坐标系"""
    homo_point = np.array([point[0], point[1], point[2], 1.0])
    transformed = matrix @ homo_point
    return transformed[:3].tolist()


def load_ground_truth(gt_dir: Path, num_frames: int = 50):
    """加载真值数据并转换坐标系"""
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
    """加载模型检测结果"""
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


def visualize_results(gt_tracks, result, output_dir: Path):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Ground truth
    ax1 = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, len(gt_tracks)))

    for idx, (gt_id, positions) in enumerate(gt_tracks.items()):
        if len(positions) < 5:
            continue
        xs = [p[1][0] for p in positions]
        ys = [p[1][1] for p in positions]
        ax1.plot(xs, ys, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
        ax1.scatter(xs[0], ys[0], c=[colors[idx % len(colors)]], s=80, marker='o', zorder=5)
        ax1.scatter(xs[-1], ys[-1], c=[colors[idx % len(colors)]], s=80, marker='x', zorder=5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Ground Truth ({len(gt_tracks)} tracks)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Reconstructed
    ax2 = axes[1]
    trajectories = result.get('trajectories', {})
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

    for idx, (track_id, traj) in enumerate(trajectories.items()):
        positions = traj['positions']
        if len(positions) < 5:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax2.plot(xs, ys, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
        ax2.scatter(xs[0], ys[0], c=[colors[idx % len(colors)]], s=80, marker='o', zorder=5)
        ax2.scatter(xs[-1], ys[-1], c=[colors[idx % len(colors)]], s=80, marker='x', zorder=5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Reconstructed ({len(trajectories)} tracks)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Traffic Flow Reconstruction Test")
    print("=" * 70)

    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    num_frames = 50
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

    # Run reconstruction
    print("\n" + "-" * 70)
    print("Running reconstruction...")
    print("-" * 70)

    result = reconstruct_traffic_flow(
        model_frames,
        max_distance=5.0,
        max_velocity=30.0,
    )

    # Evaluate
    trajectories = result.get('trajectories', {})
    lengths = [t['length'] for t in trajectories.values()]
    gt_lengths = [len(v) for v in gt_tracks.values()]

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Reconstructed tracks: {len(trajectories)}")
    print(f"Ground truth tracks: {len(gt_tracks)}")
    print(f"Avg track length: {np.mean(lengths):.1f} frames (GT: {np.mean(gt_lengths):.1f})")
    print(f"Max track length: {max(lengths) if lengths else 0} frames")
    print(f"Tracks >= 10 frames: {sum(1 for l in lengths if l >= 10)}")
    print(f"Tracks >= 20 frames: {sum(1 for l in lengths if l >= 20)} (GT: {sum(1 for l in gt_lengths if l >= 20)})")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(gt_tracks, result, output_dir)

    # Save results
    result_path = output_dir / 'reconstruction_result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved: {result_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()