"""
交通流重建测试

测试纯 DeepSORT 跟踪器的轨迹重建效果
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
from agents.deepsort_tracker import DeepSORTTracker
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


def find_best_matching_gt_track(pred_positions, gt_tracks, max_distance=15.0):
    """找到与预测轨迹最匹配的真值轨迹（基于起点距离）"""
    if not pred_positions or not gt_tracks:
        return None, float('inf')

    pred_start = np.array(pred_positions[0][:2])
    best_gt_id = None
    best_distance = float('inf')

    for gt_id, gt_pos_list in gt_tracks.items():
        if len(gt_pos_list) < 3:
            continue
        gt_start = np.array(gt_pos_list[0][1][:2])
        distance = np.linalg.norm(pred_start - gt_start)
        if distance < best_distance:
            best_distance = distance
            best_gt_id = gt_id

    if best_distance > max_distance:
        return None, best_distance
    return best_gt_id, best_distance


def visualize_single_trajectory(
    track_id, pred_traj, gt_track_id, gt_positions,
    single_dir: Path, track_color='#FF4444', gt_color='#44AA44'
):
    """可视化单条轨迹与真值对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    pred_positions = pred_traj.get('positions', [])
    pred_frames = pred_traj.get('frame_ids', [])
    pred_type = pred_traj.get('type', 'Unknown')
    pred_length = pred_traj.get('length', 0)

    if len(pred_positions) < 2:
        plt.close()
        return None

    # ========== 子图 1: 轨迹对比 (XY 平面) ==========
    ax1 = axes[0]

    # 预测轨迹
    pred_xs = [p[0] for p in pred_positions]
    pred_ys = [p[1] for p in pred_positions]
    ax1.plot(pred_xs, pred_ys, color=track_color, linewidth=3, label='Predicted', alpha=0.8)
    ax1.scatter(pred_xs[0], pred_ys[0], c=track_color, s=150, marker='o',
                zorder=5, edgecolors='white', linewidth=2, label='Pred Start')
    ax1.scatter(pred_xs[-1], pred_ys[-1], c=track_color, s=150, marker='x',
                zorder=5, linewidth=3, label='Pred End')

    # 真值轨迹
    if gt_positions:
        gt_xs = [p[1][0] for p in gt_positions]
        gt_ys = [p[1][1] for p in gt_positions]
        ax1.plot(gt_xs, gt_ys, color=gt_color, linewidth=3, linestyle='--',
                 label=f'GT (ID:{gt_track_id})', alpha=0.8)
        ax1.scatter(gt_xs[0], gt_ys[0], c=gt_color, s=150, marker='o',
                    zorder=5, edgecolors='white', linewidth=2, label='GT Start')
        ax1.scatter(gt_xs[-1], gt_ys[-1], c=gt_color, s=150, marker='x',
                    zorder=5, linewidth=3, label='GT End')

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(f'Track ID: {track_id} | Length: {pred_length} frames\nType: {pred_type}', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ========== 子图 2: 坐标随时间变化 ==========
    ax2 = axes[1]

    pred_x_vals = [p[0] for p in pred_positions]
    pred_y_vals = [p[1] for p in pred_positions]

    ax2.plot(pred_frames, pred_x_vals, color=track_color, linewidth=2,
             marker='o', markersize=5, label='Pred X', alpha=0.7)
    ax2.plot(pred_frames, pred_y_vals, color=track_color, linewidth=2,
             marker='s', markersize=5, label='Pred Y', alpha=0.5)

    if gt_positions:
        gt_frame_dict = {fp[0]: fp[1] for fp in gt_positions}
        matched_frames = [fp[0] for fp in gt_positions]
        gt_x_vals = [fp[1][0] for fp in gt_positions]
        gt_y_vals = [fp[1][1] for fp in gt_positions]

        ax2.plot(matched_frames, gt_x_vals, color=gt_color, linewidth=2,
                 linestyle='--', marker='o', markersize=5, label='GT X', alpha=0.7)
        ax2.plot(matched_frames, gt_y_vals, color=gt_color, linewidth=2,
                 linestyle='--', marker='s', markersize=5, label='GT Y', alpha=0.5)

    ax2.set_xlabel('Frame ID', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title(f'Position Over Time', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    gt_suffix = f"_vs_GT{gt_track_id}" if gt_track_id else "_no_GT"
    filename = f"track_{track_id:04d}{gt_suffix}.png"
    output_path = single_dir / filename

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def visualize_comparison(gt_tracks, result_no_map, result_with_map, output_dir: Path):
    """可视化对比三种结果"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

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

    # max_distance=3.0
    ax2 = axes[1]
    trajectories = result_no_map.get('trajectories', {})
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
    ax2.set_title(f'max_distance=3.0 ({len(trajectories)} tracks)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # max_distance=5.0
    ax3 = axes[2]
    trajectories = result_with_map.get('trajectories', {})
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

    for idx, (track_id, traj) in enumerate(trajectories.items()):
        positions = traj['positions']
        if len(positions) < 5:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        ax3.plot(xs, ys, color=colors[idx % len(colors)], linewidth=2, alpha=0.8)
        ax3.scatter(xs[0], ys[0], c=[colors[idx % len(colors)]], s=80, marker='o', zorder=5)
        ax3.scatter(xs[-1], ys[-1], c=[colors[idx % len(colors)]], s=80, marker='x', zorder=5)

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'max_distance=5.0 ({len(trajectories)} tracks)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def visualize_all_single_trajectories(gt_tracks, result_with_map, output_dir: Path, min_length: int = 10):
    """为所有轨迹生成单独的可视化对比图"""
    trajectories = result_with_map.get('trajectories', {})

    single_dir = output_dir / "single_trajectories"
    single_dir.mkdir(exist_ok=True)

    stats = {'total': len(trajectories), 'with_gt': 0, 'without_gt': 0, 'long': 0}

    print(f"\nGenerating single trajectory visualizations...")

    for track_id, traj in trajectories.items():
        if traj['length'] < min_length:
            continue

        if traj['length'] >= 20:
            stats['long'] += 1

        # 寻找匹配的真值轨迹
        gt_match, match_dist = find_best_matching_gt_track(traj['positions'], gt_tracks, max_distance=15.0)

        if gt_match:
            stats['with_gt'] += 1
            gt_positions = gt_tracks[gt_match]
        else:
            stats['without_gt'] += 1
            gt_positions = None

        # 生成可视化
        output_path = visualize_single_trajectory(
            track_id=track_id,
            pred_traj=traj,
            gt_track_id=gt_match,
            gt_positions=gt_positions,
            single_dir=single_dir
        )

        if output_path:
            status = f"GT:{gt_match}" if gt_match else "No GT"
            print(f"  Track {track_id:4d} | Len: {traj['length']:3d} | {status}")

    # 保存汇总统计
    summary_path = output_dir / "single_trajectories" / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'trajectories': {
                tid: {
                    'length': traj['length'],
                    'type': traj.get('type', 'Unknown'),
                    'has_gt': find_best_matching_gt_track(traj['positions'], gt_tracks)[0] is not None
                }
                for tid, traj in trajectories.items()
                if traj['length'] >= min_length
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Single Trajectory Summary:")
    print(f"  Total: {stats['total']} | >= {min_length} frames: {stats['with_gt']+stats['without_gt']} | With GT: {stats['with_gt']} | No GT: {stats['without_gt']} | Long (>=20): {stats['long']}")
    print(f"  Output: {single_dir}")


def main():
    print("=" * 70)
    print("Traffic Flow Reconstruction Test (Pure DeepSORT)")
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

    # ========== 跟踪参数调优 ==========
    # 减小 max_distance：让匹配更严格，减少轨迹合并
    # 减小 max_misses：让轨迹更快过期，增加轨迹数量
    TRACK_MAX_DISTANCE = 3.0      # 从 5.0 降低到 3.0
    TRACK_MAX_MISSES = 10         # 从 30 降低到 10
    TRACK_MIN_HITS = 3            # 从 2 增加到 3（更严格确认）
    # ================================

    # Run reconstruction (pure DeepSORT, max_distance=3.0)
    print("\n" + "-" * 70)
    print("Running reconstruction (max_distance=3.0)...")
    print("-" * 70)

    result_no_map = reconstruct_traffic_flow(
        model_frames,
        max_distance=TRACK_MAX_DISTANCE,
        max_velocity=30.0,
    )

    # Run reconstruction with different parameters
    print("\n" + "-" * 70)
    print("Running reconstruction (max_distance=5.0)...")
    print("-" * 70)

    result_with_map = reconstruct_traffic_flow(
        model_frames,
        max_distance=5.0,
        max_velocity=30.0,
    )

    # Evaluate
    gt_lengths = [len(v) for v in gt_tracks.values()]

    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)

    # Without map
    traj_no_map = result_no_map.get('trajectories', {})
    lengths_no_map = [t['length'] for t in traj_no_map.values()]

    print("\n[Without Map]")
    print(f"  Tracks: {len(traj_no_map)}")
    print(f"  Avg length: {np.mean(lengths_no_map):.1f} frames")
    print(f"  Max length: {max(lengths_no_map) if lengths_no_map else 0} frames")
    print(f"  Tracks >= 10 frames: {sum(1 for l in lengths_no_map if l >= 10)}")
    print(f"  Tracks >= 20 frames: {sum(1 for l in lengths_no_map if l >= 20)}")

    # With different parameters
    traj_with_map = result_with_map.get('trajectories', {})
    lengths_with_map = [t['length'] for t in traj_with_map.values()]

    print("\n[With max_distance=5.0]")
    print(f"  Tracks: {len(traj_with_map)}")
    print(f"  Avg length: {np.mean(lengths_with_map):.1f} frames")
    print(f"  Max length: {max(lengths_with_map) if lengths_with_map else 0} frames")
    print(f"  Tracks >= 10 frames: {sum(1 for l in lengths_with_map if l >= 10)}")
    print(f"  Tracks >= 20 frames: {sum(1 for l in lengths_with_map if l >= 20)}")

    # Ground truth
    print("\n[Ground Truth]")
    print(f"  Tracks: {len(gt_tracks)}")
    print(f"  Avg length: {np.mean(gt_lengths):.1f} frames")
    print(f"  Tracks >= 20 frames: {sum(1 for l in gt_lengths if l >= 20)}")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_comparison(gt_tracks, result_no_map, result_with_map, output_dir)
    visualize_all_single_trajectories(gt_tracks, result_with_map, output_dir, min_length=10)

    # Save results
    result_path = output_dir / 'reconstruction_result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'max_distance_3.0': result_no_map,
            'max_distance_5.0': result_with_map,
            'ground_truth_summary': {
                'tracks': len(gt_tracks),
                'avg_length': np.mean(gt_lengths),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {result_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()