"""
交通流重建结果保存器

支持将重建结果保存为多种格式
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class TrafficFlowSaver:
    """
    交通流重建结果保存器

    支持格式：
    - JSON: 完整的轨迹数据
    - CSV: 帧统计摘要
    """

    def __init__(self, output_dir: str = "."):
        """
        初始化

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, result: Dict, filename: str = "reconstruction_result.json") -> str:
        """
        保存为JSON格式

        Args:
            result: 重建结果
            filename: 文件名

        Returns:
            保存的文件路径
        """
        output_path = self.output_dir / filename

        # 添加元数据
        output_data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0",
                "format": "traffic_flow_reconstruction"
            },
            "result": result
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return str(output_path)

    def save_csv(self, result: Dict, filename: str = "frame_summary.csv") -> str:
        """
        保存帧摘要为CSV格式

        Args:
            result: 重建结果
            filename: 文件名

        Returns:
            保存的文件路径
        """
        output_path = self.output_dir / filename

        frames = result.get('frames', [])

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow([
                'frame_id', 'timestamp', 'vehicle_count',
                'ego_x', 'ego_y', 'ego_z'
            ])

            # 写入数据
            for frame in frames:
                ego_pos = frame.get('ego_position') or [None, None, None]
                writer.writerow([
                    frame.get('frame_id', 0),
                    frame.get('timestamp', ''),
                    frame.get('vehicle_count', 0),
                    ego_pos[0] if ego_pos else None,
                    ego_pos[1] if ego_pos else None,
                    ego_pos[2] if ego_pos else None
                ])

        return str(output_path)

    def save_trajectory_csv(self, result: Dict, filename: str = "trajectories.csv") -> str:
        """
        保存轨迹数据为CSV格式

        Args:
            result: 重建结果
            filename: 文件名

        Returns:
            保存的文件路径
        """
        output_path = self.output_dir / filename

        trajectories = result.get('trajectories', [])

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow([
                'vehicle_id', 'vehicle_type', 'frame_id', 'timestamp',
                'x', 'y', 'z', 'heading', 'speed', 'matched_lane', 'behavior'
            ])

            # 写入数据
            for traj in trajectories:
                vid = traj.get('vehicle_id', 0)
                vtype = traj.get('vehicle_type', 'Unknown')

                for state in traj.get('states', []):
                    pos = state.get('position', [0, 0, 0])
                    writer.writerow([
                        vid,
                        vtype,
                        state.get('frame_id', 0),
                        state.get('timestamp', ''),
                        pos[0], pos[1], pos[2],
                        state.get('heading', ''),
                        state.get('speed', ''),
                        state.get('matched_lane', ''),
                        state.get('behavior', '')
                    ])

        return str(output_path)

    def save_all(self, result: Dict, base_name: str = "reconstruction") -> Dict[str, str]:
        """
        保存所有格式

        Args:
            result: 重建结果
            base_name: 基础文件名

        Returns:
            保存的文件路径字典
        """
        return {
            "json": self.save_json(result, f"{base_name}.json"),
            "frame_csv": self.save_csv(result, f"{base_name}_frames.csv"),
            "trajectory_csv": self.save_trajectory_csv(result, f"{base_name}_trajectories.csv")
        }

    def load_json(self, filename: str = "reconstruction_result.json") -> Optional[Dict]:
        """
        加载JSON格式的结果

        Args:
            filename: 文件名

        Returns:
            加载的结果
        """
        input_path = self.output_dir / filename

        if not input_path.exists():
            return None

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 返回实际结果
        return data.get('result', data)


def format_summary(result: Dict) -> str:
    """
    格式化结果摘要

    Args:
        result: 重建结果

    Returns:
        格式化的摘要字符串
    """
    lines = []

    lines.append("=" * 50)
    lines.append("交通流重建结果摘要")
    lines.append("=" * 50)

    # 基本信息
    lines.append(f"\n总帧数: {result.get('total_frames', 0)}")
    lines.append(f"总车辆数: {result.get('total_vehicles', 0)}")
    lines.append(f"时长: {result.get('duration_seconds', 0):.1f} 秒")

    # 车辆类型统计
    trajectories = result.get('trajectories', [])
    type_counts = {}
    for traj in trajectories:
        vtype = traj.get('vehicle_type', 'Unknown')
        type_counts[vtype] = type_counts.get(vtype, 0) + 1

    lines.append("\n车辆类型分布:")
    for vtype, count in type_counts.items():
        lines.append(f"  - {vtype}: {count} 辆")

    # 行为统计
    behavior_counts = {}
    for traj in trajectories:
        for behavior in traj.get('behaviors', []):
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

    if behavior_counts:
        lines.append("\n行为统计:")
        for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {behavior}: {count} 次")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)