"""
DeepSORT 跟踪器测试

测试纯 DeepSORT 跟踪器的效果
"""

import sys
import json
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.deepsort_tracker import DeepSORTTracker
from utils.detection_loader import DetectionLoader


def load_detections(detection_dir: str, num_frames: int = 50):
    """加载检测数据"""
    loader = DetectionLoader(detection_dir, enable_tracking=False)
    frames = loader.load_frames(0, num_frames)

    detection_frames = []
    for frame in frames:
        objects = []
        for obj in frame.objects:
            d = obj.to_dict()
            objects.append({
                'id': d['id'],
                'location': d['location'],
                'velocity': d.get('velocity', [0, 0, 0]),
                'type': d['type'],
                'heading': d.get('heading', 0.0),
                'speed': d.get('speed', 0.0),
            })
        detection_frames.append({
            'frame_id': frame.frame_id,
            'objects': objects,
        })

    return detection_frames


def run_tracker(tracker, frames):
    """运行跟踪器并返回结果"""
    for frame_data in frames:
        frame_id = frame_data.get('frame_id', 0)
        objects = frame_data.get('objects', [])

        detections = []
        for obj in objects:
            pos = obj.get('location')
            if pos is not None:
                detections.append({
                    'location': pos,
                    'velocity': obj.get('velocity', [0, 0, 0]),
                    'type': obj.get('type', 'Unknown'),
                    'heading': obj.get('heading', 0.0),
                    'speed': obj.get('speed', 0.0),
                })

        tracker.update(detections, frame_id)

    return tracker.get_active_tracks(), tracker.get_statistics()


def test_deepsort_tracker(frames):
    """测试 DeepSORT 跟踪器"""

    print("\n" + "=" * 70)
    print("DeepSORT Tracker Test")
    print("=" * 70)

    # DeepSORT (纯位置跟踪)
    print("\n[DeepSORT - Pure Position Tracking]")
    tracker = DeepSORTTracker(
        map_api=None,
        max_distance=5.0,
        max_velocity=30.0,
        frame_interval=0.1,
        min_hits=2,
        max_misses=30,
        use_map=False,
        max_iou_distance=5.0,
    )
    tracks, stats = run_tracker(tracker, frames)

    print(f"  Active tracks: {len(tracks)}")
    print(f"  Total matches: {stats.get('matches', 0)}")
    print(f"  New tracks: {stats.get('new_tracks', 0)}")
    print(f"  Deleted tracks: {stats.get('deleted_tracks', 0)}")
    print(f"  Avg track length: {stats.get('avg_track_length', 0):.1f}")
    print(f"  Max track length: {stats.get('max_track_length', 0)}")

    # 计算轨迹长度分布
    lengths = [t.age for t in tracks.values()]
    print("\n[Track Length Distribution]")
    print(f"  >= 5 frames: {sum(1 for l in lengths if l >= 5)}")
    print(f"  >= 10 frames: {sum(1 for l in lengths if l >= 10)}")
    print(f"  >= 20 frames: {sum(1 for l in lengths if l >= 20)}")
    print(f"  >= 50 frames: {sum(1 for l in lengths if l >= 50)}")

    return {
        'tracks': len(tracks),
        'stats': stats,
        'lengths': lengths,
    }


def main():
    print("=" * 70)
    print("DeepSORT Tracker Test")
    print("=" * 70)

    num_frames = 297
    detection_dir = 'data/json_results'

    print(f"\nLoading data ({num_frames} frames)...")
    frames = load_detections(detection_dir, num_frames)
    print(f"  Loaded {len(frames)} frames")

    # 测试 DeepSORT 跟踪器
    results = test_deepsort_tracker(frames)

    # 保存结果
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'deepsort_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tracks': results['tracks'],
            'stats': results['stats'],
            'length_distribution': {
                '>=5': sum(1 for l in results['lengths'] if l >= 5),
                '>=10': sum(1 for l in results['lengths'] if l >= 10),
                '>=20': sum(1 for l in results['lengths'] if l >= 20),
                '>=50': sum(1 for l in results['lengths'] if l >= 50),
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
