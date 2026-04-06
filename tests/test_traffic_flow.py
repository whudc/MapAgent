"""
交通流重建测试脚本

功能：
1. 基于规则推理重建交通流（包含车辆跟踪）
2. 基于LLM推理优化交通流
3. 保存两种重建结果
4. 可视化重建结果对比
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TABLEAU_COLORS

from apis.map_api import MapAPI
from agents.base import AgentContext
from agents.traffic_flow import TrafficFlowAgent
from utils.detection_loader import DetectionLoader, VehicleTracker
from utils.result_saver import TrafficFlowSaver, format_summary


# 车辆类型颜色映射
VEHICLE_COLORS = {
    'Car': '#3B82F6',
    'Truck': '#EF4444',
    'Bus': '#F59E0B',
    'Suv': '#8B5CF6',
    'Non_motor_rider': '#10B981',
    'Motorcycle': '#06B6D4',
    'Pedestrian': '#9CA3AF',
    'Unknown': '#6B7280'
}


class TrafficFlowReconstructionTester:
    """交通流重建测试器"""

    def __init__(self, map_file: str, detection_path: str, output_dir: str = "test_output"):
        """
        初始化

        Args:
            map_file: 地图文件路径
            detection_path: 检测结果目录
            output_dir: 输出目录
        """
        self.map_file = map_file
        self.detection_path = detection_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载地图
        print(f"加载地图: {map_file}")
        self.map_api = MapAPI(map_file=map_file)

        # 加载检测结果
        print(f"加载检测结果: {detection_path}")
        self.loader = DetectionLoader(detection_path)
        print(f"  格式: {self.loader.get_data_format()}")
        print(f"  帧数: {self.loader.get_frame_count()}")

        # 结果存储
        self.rule_based_result: Optional[Dict] = None
        self.llm_optimized_result: Optional[Dict] = None

    def test_tracking(self, start_frame: int = None, end_frame: int = None):
        """
        测试车辆跟踪功能

        Args:
            start_frame: 起始帧
            end_frame: 结束帧
        """
        print("\n" + "="*60)
        print("车辆跟踪测试")
        print("="*60)

        # 创建加载器并运行跟踪
        loader = DetectionLoader(self.detection_path, enable_tracking=True)
        loader.run_tracking(start_frame, end_frame)

        # 获取跟踪统计
        stats = loader.get_tracker_statistics()
        print(f"\n跟踪统计:")
        print(f"  总轨迹数: {stats['total_tracks']}")

        # 分析轨迹长度分布
        lengths = list(stats['track_lengths'].values())
        if lengths:
            print(f"  轨迹长度分布:")
            print(f"    - 最短: {min(lengths)} 帧")
            print(f"    - 最长: {max(lengths)} 帧")
            print(f"    - 平均: {sum(lengths)/len(lengths):.1f} 帧")

        return loader

    def run_rule_based_reconstruction(self, start_frame: int = None, end_frame: int = None) -> Dict:
        """
        运行基于规则的重建

        Args:
            start_frame: 起始帧
            end_frame: 结束帧

        Returns:
            重建结果
        """
        print("\n" + "="*60)
        print("基于规则的交通流重建")
        print("="*60)

        context = AgentContext(map_api=self.map_api, llm_client=None)
        agent = TrafficFlowAgent(context)

        result = agent.process(
            query="基于规则重建交通流",
            detection_path=self.detection_path,
            start_frame=start_frame,
            end_frame=end_frame,
            use_llm=False  # 不使用LLM
        )

        self.rule_based_result = result
        print(f"\n规则重建结果:")
        print(f"  总帧数: {result.get('total_frames', 0)}")
        print(f"  总车辆数: {result.get('total_vehicles', 0)}")
        print(f"  摘要: {result.get('summary', '')}")

        return result

    def run_llm_optimized_reconstruction(self, start_frame: int = None, end_frame: int = None,
                                          llm_provider: str = "deepseek") -> Dict:
        """
        运行LLM优化的重建

        Args:
            start_frame: 起始帧
            end_frame: 结束帧
            llm_provider: LLM提供商

        Returns:
            重建结果
        """
        print("\n" + "="*60)
        print(f"LLM优化的交通流重建 (provider: {llm_provider})")
        print("="*60)

        # 创建LLM客户端
        from core.llm_client import LLMClient, LLMConfig
        from config import settings

        llm_client = None

        try:
            # 优先使用 settings 中配置的 API key
            api_key = settings.llm_api_key if settings.llm_api_key else None

            if not api_key:
                # 尝试从环境变量获取
                import os
                if llm_provider == "deepseek":
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                elif llm_provider == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                elif llm_provider == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                print("警告: 未配置有效的 API Key")
                print("将使用规则推理替代 LLM 优化")
            else:
                if llm_provider == "deepseek":
                    config = LLMConfig.for_deepseek(api_key=api_key)
                else:
                    config = LLMConfig.from_env()

                llm_client = LLMClient(config)
                print(f"LLM客户端已初始化: {config.model}")
        except Exception as e:
            print(f"警告: 无法创建LLM客户端: {e}")
            print("将使用规则推理替代")

        context = AgentContext(map_api=self.map_api, llm_client=llm_client)
        agent = TrafficFlowAgent(context)

        result = agent.process(
            query="基于LLM优化重建交通流",
            detection_path=self.detection_path,
            start_frame=start_frame,
            end_frame=end_frame,
            use_llm=True  # 使用LLM
        )

        self.llm_optimized_result = result
        print(f"\nLLM优化结果:")
        print(f"  总帧数: {result.get('total_frames', 0)}")
        print(f"  总车辆数: {result.get('total_vehicles', 0)}")
        print(f"  摘要: {result.get('summary', '')}")

        return result

    def save_results(self) -> Dict[str, str]:
        """
        保存重建结果

        Returns:
            保存的文件路径
        """
        print("\n" + "="*60)
        print("保存重建结果")
        print("="*60)

        saver = TrafficFlowSaver(str(self.output_dir))
        saved_files = {}

        # 保存规则推理结果
        if self.rule_based_result:
            files = saver.save_all(self.rule_based_result, "rule_based")
            saved_files['rule_based'] = files
            print(f"\n规则推理结果已保存:")
            for name, path in files.items():
                print(f"  {name}: {path}")

        # 保存LLM优化结果
        if self.llm_optimized_result:
            files = saver.save_all(self.llm_optimized_result, "llm_optimized")
            saved_files['llm_optimized'] = files
            print(f"\nLLM优化结果已保存:")
            for name, path in files.items():
                print(f"  {name}: {path}")

        return saved_files

    def visualize_single_frame(self, frame_idx: int = 0, show_trajectories: bool = True):
        """
        可视化单帧

        Args:
            frame_idx: 帧索引
            show_trajectories: 是否显示轨迹
        """
        if not self.rule_based_result:
            print("请先运行重建")
            return

        frames = self.rule_based_result.get('frames', [])
        if frame_idx >= len(frames):
            print(f"帧索引 {frame_idx} 超出范围")
            return

        frame = frames[frame_idx]
        vehicles = frame.get('vehicles', [])

        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制车辆
        for v in vehicles:
            pos = v.get('position', [0, 0, 0])
            v_type = v.get('vehicle_type', 'Unknown')
            v_id = v.get('vehicle_id', 0)
            heading = v.get('heading', 0)

            color = VEHICLE_COLORS.get(v_type, VEHICLE_COLORS['Unknown'])

            # 绘制车辆点
            ax.scatter(pos[0], pos[1], c=color, s=100, marker='s', zorder=5)

            # 绘制航向箭头
            dx = np.cos(heading) * 3
            dy = np.sin(heading) * 3
            ax.arrow(pos[0], pos[1], dx, dy, head_width=1.5, head_length=1,
                    fc=color, ec=color, alpha=0.7, zorder=4)

            # 显示ID
            ax.annotate(str(v_id), (pos[0], pos[1] + 3), fontsize=8, ha='center', color=color)

        # 绘制轨迹
        if show_trajectories:
            trajectories = self.rule_based_result.get('trajectories', [])
            for traj in trajectories:
                states = traj.get('states', [])
                if len(states) >= 2:
                    x_coords = [s['position'][0] for s in states]
                    y_coords = [s['position'][1] for s in states]
                    v_type = traj.get('vehicle_type', 'Unknown')
                    color = VEHICLE_COLORS.get(v_type, VEHICLE_COLORS['Unknown'])

                    ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1.5, linestyle='--')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'帧 {frame.get("frame_id", frame_idx)} - 车辆数: {len(vehicles)}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 图例
        legend_elements = [plt.scatter([], [], c=color, s=100, marker='s', label=vtype)
                          for vtype, color in VEHICLE_COLORS.items()
                          if any(v.get('vehicle_type') == vtype for v in vehicles)]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # 保存图像
        output_path = self.output_dir / f'frame_{frame_idx}.png'
        plt.savefig(output_path, dpi=150)
        print(f"帧图像已保存: {output_path}")

        plt.show()

    def visualize_animation(self, max_frames: int = 100, interval: int = 100):
        """
        生成动画可视化

        Args:
            max_frames: 最大帧数
            interval: 帧间隔(ms)
        """
        if not self.rule_based_result:
            print("请先运行重建")
            return

        frames = self.rule_based_result.get('frames', [])[:max_frames]
        if not frames:
            print("没有帧数据")
            return

        print(f"\n生成动画 ({len(frames)} 帧)...")

        fig, ax = plt.subplots(figsize=(14, 10))

        # 计算所有帧的边界
        all_x, all_y = [], []
        for frame in frames:
            for v in frame.get('vehicles', []):
                pos = v.get('position', [0, 0, 0])
                all_x.append(pos[0])
                all_y.append(pos[1])

        margin = 20
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin

        def update(frame_idx):
            ax.clear()
            frame = frames[frame_idx]
            vehicles = frame.get('vehicles', [])

            # 绘制车辆
            for v in vehicles:
                pos = v.get('position', [0, 0, 0])
                v_type = v.get('vehicle_type', 'Unknown')
                v_id = v.get('vehicle_id', 0)
                heading = v.get('heading', 0)

                color = VEHICLE_COLORS.get(v_type, VEHICLE_COLORS['Unknown'])

                # 车辆点
                ax.scatter(pos[0], pos[1], c=color, s=150, marker='s', zorder=5, edgecolors='white', linewidths=1)

                # 航向箭头
                dx = np.cos(heading) * 4
                dy = np.sin(heading) * 4
                ax.arrow(pos[0], pos[1], dx, dy, head_width=2, head_length=1.5,
                        fc=color, ec='white', alpha=0.8, zorder=4)

                # ID标签
                ax.annotate(str(v_id), (pos[0], pos[1] + 5), fontsize=9, ha='center',
                           color='black', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title(f'交通流重建 - 帧 {frame.get("frame_id", frame_idx)} / {len(frames)} | 车辆数: {len(vehicles)}',
                        fontsize=14)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # 时间进度条
            progress = frame_idx / len(frames)
            ax.axhline(y=y_min - 5, xmin=0.1, xmax=0.1 + 0.8 * progress, color='green', linewidth=3)

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=True)

        # 保存动画
        output_path = self.output_dir / 'traffic_flow_animation.gif'
        anim.save(str(output_path), writer='pillow', fps=10)
        print(f"动画已保存: {output_path}")

        plt.show()

    def compare_results(self):
        """
        对比规则推理和LLM优化结果
        """
        if not self.rule_based_result or not self.llm_optimized_result:
            print("请先运行两种重建")
            return

        print("\n" + "="*60)
        print("结果对比")
        print("="*60)

        # 基本统计对比
        print("\n基本统计:")
        print(f"{'指标':<20} {'规则推理':<15} {'LLM优化':<15} {'差异':<15}")
        print("-" * 65)

        metrics = ['total_frames', 'total_vehicles', 'duration_seconds']
        for metric in metrics:
            rule_val = self.rule_based_result.get(metric, 0)
            llm_val = self.llm_optimized_result.get(metric, 0)
            diff = llm_val - rule_val
            print(f"{metric:<20} {rule_val:<15} {llm_val:<15} {diff:+<15}")

        # 车辆类型对比
        print("\n车辆类型分布:")
        rule_types = {}
        llm_types = {}

        for traj in self.rule_based_result.get('trajectories', []):
            vtype = traj.get('vehicle_type', 'Unknown')
            rule_types[vtype] = rule_types.get(vtype, 0) + 1

        for traj in self.llm_optimized_result.get('trajectories', []):
            vtype = traj.get('vehicle_type', 'Unknown')
            llm_types[vtype] = llm_types.get(vtype, 0) + 1

        all_types = set(rule_types.keys()) | set(llm_types.keys())
        print(f"{'类型':<20} {'规则推理':<15} {'LLM优化':<15}")
        print("-" * 50)
        for vtype in sorted(all_types):
            print(f"{vtype:<20} {rule_types.get(vtype, 0):<15} {llm_types.get(vtype, 0):<15}")

        # 行为统计对比
        print("\n行为统计:")
        rule_behaviors = {}
        llm_behaviors = {}

        for traj in self.rule_based_result.get('trajectories', []):
            for b in traj.get('behaviors', []):
                rule_behaviors[b] = rule_behaviors.get(b, 0) + 1

        for traj in self.llm_optimized_result.get('trajectories', []):
            for b in traj.get('behaviors', []):
                llm_behaviors[b] = llm_behaviors.get(b, 0) + 1

        all_behaviors = set(rule_behaviors.keys()) | set(llm_behaviors.keys())
        print(f"{'行为':<20} {'规则推理':<15} {'LLM优化':<15}")
        print("-" * 50)
        for behavior in sorted(all_behaviors):
            print(f"{behavior:<20} {rule_behaviors.get(behavior, 0):<15} {llm_behaviors.get(behavior, 0):<15}")

        # 可视化对比图
        self._plot_comparison()

    def _plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for idx, (result, title) in enumerate([
            (self.rule_based_result, "规则推理"),
            (self.llm_optimized_result, "LLM优化")
        ]):
            ax = axes[idx]
            trajectories = result.get('trajectories', [])

            for traj in trajectories:
                states = traj.get('states', [])
                if len(states) >= 2:
                    x_coords = [s['position'][0] for s in states]
                    y_coords = [s['position'][1] for s in states]
                    v_type = traj.get('vehicle_type', 'Unknown')
                    color = VEHICLE_COLORS.get(v_type, VEHICLE_COLORS['Unknown'])

                    ax.plot(x_coords, y_coords, color=color, alpha=0.6, linewidth=1.5)

                    # 标记起点和终点
                    ax.scatter(x_coords[0], y_coords[0], c=color, s=50, marker='o', zorder=5)
                    ax.scatter(x_coords[-1], y_coords[-1], c=color, s=50, marker='x', zorder=5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{title}\n车辆数: {len(trajectories)}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / 'comparison.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n对比图已保存: {output_path}")

        plt.show()

    def generate_report(self) -> str:
        """
        生成测试报告

        Returns:
            报告文本
        """
        report = []
        report.append("=" * 70)
        report.append("交通流重建测试报告")
        report.append("=" * 70)
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"检测结果目录: {self.detection_path}")
        report.append(f"地图文件: {self.map_file}")
        report.append(f"数据格式: {self.loader.get_data_format()}")
        report.append(f"总帧数: {self.loader.get_frame_count()}")

        if self.rule_based_result:
            report.append("\n" + "-" * 70)
            report.append("规则推理结果")
            report.append("-" * 70)
            report.append(format_summary(self.rule_based_result))

        if self.llm_optimized_result:
            report.append("\n" + "-" * 70)
            report.append("LLM优化结果")
            report.append("-" * 70)
            report.append(format_summary(self.llm_optimized_result))

        report.append("\n" + "=" * 70)

        report_text = "\n".join(report)

        # 保存报告
        report_path = self.output_dir / 'test_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n测试报告已保存: {report_path}")

        return report_text


def main():
    parser = argparse.ArgumentParser(description="交通流重建测试脚本")

    parser.add_argument('--map', type=str, default='data/vector_map.json',
                        help='地图文件路径')
    parser.add_argument('--detection', type=str, default='data/json_results',
                        help='检测结果目录')
    parser.add_argument('--output', type=str, default='test_output',
                        help='输出目录')
    parser.add_argument('--start-frame', type=int, default=None,
                        help='起始帧')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='结束帧')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='动画最大帧数')
    parser.add_argument('--llm-provider', type=str, default='deepseek',
                        choices=['deepseek', 'anthropic', 'openai'],
                        help='LLM提供商')
    parser.add_argument('--skip-llm', action='store_true',
                        help='跳过LLM优化')
    parser.add_argument('--animation', action='store_true',
                        help='生成动画')
    parser.add_argument('--compare', action='store_true',
                        help='对比结果')
    parser.add_argument('--test-tracking', action='store_true',
                        help='测试车辆跟踪功能')

    args = parser.parse_args()

    # 创建测试器
    tester = TrafficFlowReconstructionTester(
        map_file=args.map,
        detection_path=args.detection,
        output_dir=args.output
    )

    # 测试跟踪
    if args.test_tracking:
        tester.test_tracking(
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )

    # 运行规则推理
    tester.run_rule_based_reconstruction(
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )

    # 运行LLM优化
    if not args.skip_llm:
        tester.run_llm_optimized_reconstruction(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            llm_provider=args.llm_provider
        )

    # 保存结果
    tester.save_results()

    # 对比结果
    if args.compare and not args.skip_llm:
        tester.compare_results()

    # 可视化单帧
    tester.visualize_single_frame(frame_idx=0, show_trajectories=True)

    # 生成动画
    if args.animation:
        tester.visualize_animation(max_frames=args.max_frames)

    # 生成报告
    report = tester.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()