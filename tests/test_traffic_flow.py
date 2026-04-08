"""
交通流重建测试

测试 TrafficFlowAgent 的轨迹重建效果（与 UI 使用方式一致）
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.traffic_flow import TrafficFlowAgent, reconstruct_traffic_flow
from agents.base import AgentContext
from utils.detection_loader import DetectionLoader
from apis.map_api import MapAPI


def get_vehicle_color(obj_type: str) -> str:
    """根据车辆类型返回颜色"""
    colors = {
        'Vehicle': 'blue',
        'Bus': 'orange',
        'Truck': 'red',
        'Pedestrian': 'green',
        'Cyclist': 'purple',
        'Unknown': 'gray',
    }
    return colors.get(obj_type, 'gray')


def get_lane_color(line_color: str) -> str:
    """根据车道线颜色返回 matplotlib 颜色"""
    colors = {
        'yellow': 'gold',
        'white': 'gray',
        'unknown': 'lightgray',
    }
    return colors.get(line_color.lower(), 'lightgray')


def get_lane_style(lane_type: str) -> tuple:
    """根据车道线类型返回线型"""
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
    """绘制地图（仅车道边界）"""
    if map_api is None:
        return

    all_x = []
    all_y = []

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

    if x_range and y_range:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    elif all_x and all_y:
        margin = 20
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


def draw_trajectories(ax, trajectories: Dict, map_api: Optional[MapAPI],
                     title: str = "", min_length: int = 5,
                     x_range: tuple = None, y_range: tuple = None):
    """
    绘制轨迹和地图

    Args:
        ax: matplotlib 轴
        trajectories: 轨迹字典 {track_id: trajectory_data}
        map_api: 地图 API
        title: 标题
        min_length: 最小轨迹长度
        x_range: X 轴范围
        y_range: Y 轴范围
    """
    ax.clear()
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)

    # 先绘制地图
    draw_map(ax, map_api, x_range, y_range)

    if not trajectories:
        ax.text(0, 0, 'No trajectories', ha='center', va='center', fontsize=14)
        return

    all_x = []
    all_y = []
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

    for idx, (track_id, traj) in enumerate(trajectories.items()):
        positions = traj.get('positions', [])
        if len(positions) < min_length:
            continue

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        all_x.extend(xs)
        all_y.extend(ys)

        color = colors[idx % len(colors)]

        # 绘制轨迹线
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8, label=f'#{track_id}')

        # 绘制起点和终点
        ax.scatter(xs[0], ys[0], c=[color], s=80, marker='o', zorder=5,
                  edgecolors='white', linewidth=1.5, label='_nolegend_')
        ax.scatter(xs[-1], ys[-1], c=[color], s=80, marker='x', zorder=5,
                  linewidth=2, label='_nolegend_')

    # 添加图例说明
    if trajectories:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=8, label='Start'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                  markersize=8, label='End'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 设置坐标范围
    if all_x and all_y and not x_range:
        margin = 15
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


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


def visualize_results(result: dict, map_api: Optional[MapAPI], output_dir: Path, min_length: int = 10):
    """可视化所有轨迹（结合地图）"""
    trajectories = result.get('trajectories', {})

    # 汇总图（带地图）
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    draw_trajectories(ax, trajectories, map_api,
                     title=f'Traffic Flow Reconstruction ({len(trajectories)} tracks)',
                     min_length=min_length)
    plt.tight_layout()
    output_path = output_dir / 'trajectories_overview_with_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved overview with map: {output_path}")
    plt.close()

    # 单条轨迹图（带地图）
    single_dir = output_dir / "single_trajectories"
    single_dir.mkdir(exist_ok=True)

    print(f"\nGenerating single trajectory visualizations with map...")
    count = 0
    for track_id, traj in trajectories.items():
        if traj['length'] < min_length:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        single_traj = {track_id: traj}
        draw_trajectories(ax, single_traj, map_api,
                         title=f'Track ID: {track_id} | Length: {traj["length"]} frames | Type: {traj.get("type", "Unknown")}',
                         min_length=1)
        plt.tight_layout()

        output_path = single_dir / f"track_{track_id:04d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        count += 1
        print(f"  Track {track_id:4d} | Len: {traj['length']:3d} | Type: {traj.get('type', 'Unknown')}")

    # 保存汇总统计
    stats = {
        'total': len(trajectories),
        f'>= {min_length} frames': count,
        'long (>=20)': sum(1 for t in trajectories.values() if t['length'] >= 20),
    }
    summary_path = single_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'trajectories': {
                tid: {
                    'length': traj['length'],
                    'type': traj.get('type', 'Unknown'),
                }
                for tid, traj in trajectories.items()
                if traj['length'] >= min_length
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Trajectory Summary:")
    print(f"  Total: {stats['total']} | >= {min_length} frames: {stats[f'>= {min_length} frames']} | Long (>=20): {stats['long (>=20)']}")
    print(f"  Output: {single_dir}")


def main():
    print("=" * 70)
    print("Traffic Flow Reconstruction Test (TrafficFlowAgent)")
    print("=" * 70)

    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    num_frames = 297
    detection_dir = 'data/json_results'
    map_file = 'data/vector_map.json'

    print(f"\nConfig:")
    print(f"  Detection dir: {detection_dir}")
    print(f"  Map file: {map_file}")
    print(f"  Num frames: {num_frames}")

    # Load map
    print("\nLoading map...")
    try:
        map_api = MapAPI(map_file=map_file)
        print(f"  Map loaded: {map_api.map.get_lane_count()} lanes, {map_api.map.get_centerline_count()} centerlines")
    except FileNotFoundError:
        print(f"  Warning: Map file not found ({map_file}), proceeding without map")
        map_api = None

    # 使用 Agent 方式重建（与 UI 一致）
    print("\n" + "=" * 70)
    print("Method 1: TrafficFlowAgent.process() (与 UI 一致)")
    print("=" * 70)

    # 创建 Agent 上下文
    context = AgentContext(map_api=map_api, llm_client=None)

    # 创建 Agent（use_llm=False，纯 DeepSORT 模式）
    agent = TrafficFlowAgent(context, use_llm=False)
    print(f"  Agent created: {agent.name}")

    # 调用 process 方法（与 UI 调用方式一致）
    print(f"\n  Running reconstruction via process()...")
    result_agent = agent.process(
        query="基于检测结果重建交通流",
        detection_path=detection_dir,
        start_frame=None,
        end_frame=None,
        output_path=str(output_dir / 'agent_result.json')
    )

    if result_agent.get('success'):
        print(f"  Success!")
        print(f"    Total frames: {result_agent.get('total_frames', 0)}")
        print(f"    Total vehicles: {result_agent.get('total_vehicles', 0)}")
        trajectories_agent = result_agent.get('trajectories', [])
    else:
        print(f"  Failed: {result_agent.get('error', 'Unknown error')}")
        trajectories_agent = []

    # 对比：使用便捷函数方式
    print("\n" + "=" * 70)
    print("Method 2: reconstruct_traffic_flow() (便捷函数)")
    print("=" * 70)

    # Load model detections for convenience function
    model_frames = load_model_detections(detection_dir, num_frames)
    print(f"  Loaded {len(model_frames)} frames")

    result_func = reconstruct_traffic_flow(
        model_frames,
        max_distance=5.0,
        max_velocity=30.0,
    )

    traj_func = result_func.get('trajectories', {})
    lengths_func = [t['length'] for t in traj_func.values()]

    print(f"  Tracks: {len(traj_func)}")
    print(f"  Avg length: {np.mean(lengths_func):.1f} frames")
    print(f"  Max length: {max(lengths_func) if lengths_func else 0} frames")

    # 打印 Agent 结果统计
    print("\n" + "=" * 70)
    print("Agent Results Summary")
    print("=" * 70)

    if trajectories_agent:
        lengths_agent = [t.get('length', len(t.get('states', []))) for t in trajectories_agent]
        print(f"  Total trajectories: {len(trajectories_agent)}")
        print(f"  Avg length: {np.mean(lengths_agent):.1f} frames")
        print(f"  Max length: {max(lengths_agent) if lengths_agent else 0} frames")
        print(f"  Tracks >= 10 frames: {sum(1 for l in lengths_agent if l >= 10)}")
        print(f"  Tracks >= 20 frames: {sum(1 for l in lengths_agent if l >= 20)}")

    # Visualize (使用便捷函数的结果，因为格式兼容)
    print("\nGenerating visualizations...")
    visualize_results(result_func, map_api, output_dir, min_length=10)

    # Save results
    result_path = output_dir / 'reconstruction_result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'agent_method': {
                'total_frames': result_agent.get('total_frames', 0),
                'total_vehicles': result_agent.get('total_vehicles', 0),
            },
            'convenience_function': result_func,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {result_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()