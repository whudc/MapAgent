#!/usr/bin/env python3
"""
MapAgent 静态矢量地图生成器

遍历所有帧数据，合并重复数据，生成一个完整的静态矢量地图 JSON 文件。

数据源:
- result_4dline_V1: 车道线数据
- result_traffic_sign_V1: 交通标志数据
- result_traffic_light_V1: 交通灯数据

输出: vector_map.json (单个合并后的静态地图)
"""

import json
import os
import glob
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ==================== 数据模型 ====================

class LaneType(str, Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    DOUBLE_DASHED = "double_dashed"
    LEFT_DASHED_RIGHT_SOLID = "left_dashed_right_solid"
    BILATERAL = "bilateral"
    CURB = "curb"
    FENCE = "fence"
    NO_LANE = "no_lane"
    DIVERSION_BOUNDARY = "diversion_boundary"
    UNKNOWN = "unknown"


class LaneColor(str, Enum):
    YELLOW = "yellow"
    WHITE = "white"
    UNKNOWN = "unknown"


@dataclass
class LaneLine:
    """车道线"""
    id: str
    type: str
    color: str
    coordinates: List[List[float]]
    length: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "color": self.color,
            "coordinates": self.coordinates,
            "length": round(self.length, 2)
        }


@dataclass
class Centerline:
    """车道中心线"""
    id: str
    coordinates: List[List[float]]
    left_boundary_id: Optional[str] = None
    right_boundary_id: Optional[str] = None
    predecessor_ids: Set[str] = field(default_factory=set)
    successor_ids: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "coordinates": self.coordinates,
            "left_boundary_id": self.left_boundary_id,
            "right_boundary_id": self.right_boundary_id,
            "predecessor_ids": list(self.predecessor_ids),
            "successor_ids": list(self.successor_ids)
        }


@dataclass
class RoadMark:
    """道路标记"""
    id: str
    type: str
    coordinates: List[List[float]]
    semantic: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "coordinates": self.coordinates,
            "semantic": self.semantic
        }


@dataclass
class TrafficSign:
    """交通标志"""
    id: str
    category: str
    function: Dict
    camera: str
    bbox: List[float]
    position_3d: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category,
            "function": self.function,
            "camera": self.camera,
            "bbox": self.bbox,
            "position_3d": self.position_3d
        }


@dataclass
class Intersection:
    """路口"""
    id: str
    center: List[float]
    lanes: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "center": self.center,
            "lanes": list(self.lanes)
        }


# ==================== 类型映射 ====================

LANE_TYPE_MAP = {
    "SOLID_LANE": LaneType.SOLID,
    "DASHED_LANE": LaneType.DASHED,
    "DOUBLE_SOLID": LaneType.DOUBLE_SOLID,
    "DOUBLE_DASHED": LaneType.DOUBLE_DASHED,
    "LEFT_DASHED_RIGHT_SOLID": LaneType.LEFT_DASHED_RIGHT_SOLID,
    "BILATERAL": LaneType.BILATERAL,
    "CURB": LaneType.CURB,
    "FENCE": LaneType.FENCE,
    "NO_LANE": LaneType.NO_LANE,
    "DIVERSION_BOUNDAR": LaneType.DIVERSION_BOUNDARY,
    "SOLID_LINE": LaneType.SOLID,
}

LANE_COLOR_MAP = {
    "YELLOW": LaneColor.YELLOW,
    "WHITE": LaneColor.WHITE,
}


# ==================== 工具函数 ====================

def calculate_length(coords: List[List[float]]) -> float:
    """计算线段长度"""
    if len(coords) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(coords)):
        p1, p2 = coords[i-1], coords[i]
        if HAS_NUMPY:
            length += np.linalg.norm(np.array(p2) - np.array(p1))
        else:
            dist = sum((p2[j] - p1[j])**2 for j in range(min(len(p1), len(p2))))
            length += dist ** 0.5
    return length


def coords_hash(coords: List[List[float]], precision: int = 2) -> str:
    """生成坐标的哈希值用于去重"""
    if not coords:
        return ""
    # 取首尾和中间点生成哈希
    key_points = []
    if len(coords) >= 1:
        key_points.append([round(c, precision) for c in coords[0]])
    if len(coords) >= 2:
        key_points.append([round(c, precision) for c in coords[-1]])
    if len(coords) >= 3:
        mid = len(coords) // 2
        key_points.append([round(c, precision) for c in coords[mid]])

    return hashlib.md5(json.dumps(key_points).encode()).hexdigest()[:16]


def coords_similarity(coords1: List[List[float]], coords2: List[List[float]], threshold: float = 5.0) -> bool:
    """判断两条线是否相似（用于去重）"""
    if not coords1 or not coords2:
        return False

    # 比较首尾点距离
    if HAS_NUMPY:
        start_dist = np.linalg.norm(np.array(coords1[0]) - np.array(coords2[0]))
        end_dist = np.linalg.norm(np.array(coords1[-1]) - np.array(coords2[-1]))
    else:
        start_dist = sum((coords1[0][j] - coords2[0][j])**2 for j in range(3)) ** 0.5
        end_dist = sum((coords1[-1][j] - coords2[-1][j])**2 for j in range(3)) ** 0.5

    return start_dist < threshold and end_dist < threshold


def mean_position(coords: List[List[float]]) -> List[float]:
    """计算坐标的平均位置"""
    if not coords:
        return [0.0, 0.0, 0.0]
    if HAS_NUMPY:
        return list(np.mean(coords, axis=0))
    else:
        n = len(coords)
        dim = len(coords[0]) if coords else 3
        return [sum(c[i] for c in coords) / n for i in range(dim)]


# ==================== 数据合并器 ====================

class VectorMapMerger:
    """矢量地图合并器 - 合并所有帧的数据"""

    def __init__(self):
        self.lane_lines: Dict[str, LaneLine] = {}
        self.centerlines: Dict[str, Centerline] = {}
        self.road_marks: Dict[str, RoadMark] = {}
        self.traffic_signs: Dict[str, TrafficSign] = {}
        self.traffic_lights: Dict[str, Any] = {}
        self.intersections: Dict[str, Intersection] = {}

        # 用于去重的哈希表
        self.lane_hash_map: Dict[str, str] = {}  # hash -> lane_id
        self.centerline_hash_map: Dict[str, str] = {}  # hash -> centerline_id

        # 全局统计
        self.frames_processed = 0
        self.total_objects_seen = 0

        # 坐标变换缓存（每帧的 ego2global 矩阵可能不同）
        self._transform_cache: Dict[int, List[List[float]]] = {}

    def _transform_point(self, point: List[float], transform: List[List[float]]) -> List[float]:
        """
        使用 4x4 变换矩阵将点从局部坐标系转换到全局坐标系

        Args:
            point: 局部坐标系中的点 [x, y, z]
            transform: 4x4 齐次变换矩阵

        Returns:
            全局坐标系中的点 [x, y, z]
        """
        if not transform or len(transform) < 4:
            return point

        x, y, z = point[0], point[1], point[2]

        # 应用 4x4 变换矩阵
        global_x = transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3]
        global_y = transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3]
        global_z = transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3]

        return [global_x, global_y, global_z]

    def _apply_frame_transform(self, coords: List[List[float]],
                                frame_idx: int,
                                lane_data: Dict) -> List[List[float]]:
        """
        对一帧的所有坐标应用 ego2global 变换

        Args:
            coords: 局部坐标系中的坐标列表
            frame_idx: 帧索引
            lane_data: 原始车道线数据（包含 ego2global_transformation_matrix）

        Returns:
            全局坐标系中的坐标列表
        """
        if not coords:
            return coords

        # 获取或缓存变换矩阵
        if frame_idx not in self._transform_cache:
            transform = lane_data.get('ego2global_transformation_matrix')
            if transform:
                self._transform_cache[frame_idx] = transform
            else:
                return coords  # 没有变换矩阵，返回原坐标

        transform = self._transform_cache[frame_idx]
        if not transform:
            return coords

        # 对所有点应用变换
        return [self._transform_point(p, transform) for p in coords]

    def process_frame(self, lane_data: Dict, sign_data: Dict = None, frame_idx: int = 0):
        """处理单帧数据并合并"""
        self.frames_processed += 1

        # 获取 ego2global 变换矩阵
        ego_transform = lane_data.get('ego2global_transformation_matrix')

        # 解析车道线
        lanes = lane_data.get('lanelines_annotation', {}).get('lane', [])
        for lane in lanes:
            lane_id = str(lane.get('id', ''))
            raw_type = lane.get('type', 'UNKNOWN')
            raw_color = lane.get('color', 'UNKNOWN')
            geo_3d = lane.get('geo_3d', [])

            if not geo_3d:
                continue

            # 应用 ego2global 坐标变换
            if ego_transform:
                geo_3d = self._apply_frame_transform(geo_3d, frame_idx, lane_data)

            lane_type = LANE_TYPE_MAP.get(raw_type, LaneType.UNKNOWN).value
            lane_color = LANE_COLOR_MAP.get(raw_color, LaneColor.UNKNOWN).value

            # 生成哈希判断是否已存在
            h = coords_hash(geo_3d)

            if h in self.lane_hash_map:
                # 已存在相似车道线，跳过
                existing_id = self.lane_hash_map[h]
                # 但可以验证是否真的是同一条
                if coords_similarity(self.lane_lines[existing_id].coordinates, geo_3d):
                    continue

            # 添加新车道线
            self.lane_hash_map[h] = lane_id
            self.lane_lines[lane_id] = LaneLine(
                id=lane_id,
                type=lane_type,
                color=lane_color,
                coordinates=geo_3d,
                length=calculate_length(geo_3d)
            )

        # 解析中心线和拓扑关系
        associations = lane_data.get('lanelines_annotation', {}).get('associations', [])

        # 构建车道 ID 到变换后坐标的映射
        lane_coords_map = {}
        for l in lanes:
            lid = str(l['id'])
            coords = l.get('geo_3d', [])
            if ego_transform:
                coords = self._apply_frame_transform(coords, frame_idx, lane_data)
            lane_coords_map[lid] = coords

        for assoc in associations:
            centerline_id = str(assoc.get('centerline_id', ''))

            # 获取坐标（已变换）
            coords = lane_coords_map.get(centerline_id, [])

            # 获取拓扑关系
            left_id = assoc.get('centerline_left_ID')
            right_id = assoc.get('centerline_right_ID')
            predecessors = [str(p) for p in assoc.get('id_centerline_predecessor', [])]
            successors = [str(s) for s in assoc.get('id_centerline_successor', [])]

            # 合并或创建中心线
            if centerline_id in self.centerlines:
                # 已存在，合并拓扑关系
                cl = self.centerlines[centerline_id]
                cl.predecessor_ids.update(predecessors)
                cl.successor_ids.update(successors)
                if left_id and not cl.left_boundary_id:
                    cl.left_boundary_id = str(left_id)
                if right_id and not cl.right_boundary_id:
                    cl.right_boundary_id = str(right_id)
            else:
                # 创建新的中心线
                self.centerlines[centerline_id] = Centerline(
                    id=centerline_id,
                    coordinates=coords,
                    left_boundary_id=str(left_id) if left_id else None,
                    right_boundary_id=str(right_id) if right_id else None,
                    predecessor_ids=set(predecessors),
                    successor_ids=set(successors)
                )

        # 解析道路标记
        marks = lane_data.get('lanelines_annotation', {}).get('road_mark', [])
        for mark in marks:
            mark_id = str(mark.get('id', ''))
            mark_type = mark.get('type', 'AREA')
            keypoints = mark.get('geo_3d', {}).get('geo_keypoints_list', [])
            semantic = mark.get('semantic', {})

            if not keypoints:
                continue

            # 道路标记用ID去重
            if mark_id not in self.road_marks:
                self.road_marks[mark_id] = RoadMark(
                    id=mark_id,
                    type=mark_type,
                    coordinates=keypoints,
                    semantic=semantic
                )

        # 解析交通标志
        if sign_data:
            signs = sign_data.get('traffic_signs', [])
            for sign in signs:
                sign_id = str(sign.get('id', ''))
                outline = sign.get('traffic_sign_outline', {})
                function = outline.get('function', {})
                camera = outline.get('camera', '')
                bbox = outline.get('bbox', [])

                # 判断标志类别
                category = 'unknown'
                if function:
                    if 'lane_direction_sign' in function:
                        category = 'lane_direction'
                    elif 'location_sign' in function:
                        category = 'location'
                    elif 'unclear' in function:
                        category = 'unclear'

                # 交通标志按ID+camera去重
                sign_key = f"{sign_id}_{camera}"
                if sign_key not in self.traffic_signs:
                    self.traffic_signs[sign_key] = TrafficSign(
                        id=sign_key,
                        category=category,
                        function=function,
                        camera=camera,
                        bbox=bbox
                    )

        self.total_objects_seen += len(lanes) + len(associations)

    def build_intersections(self):
        """根据拓扑关系构建路口"""
        # 找汇聚点
        junction_points = defaultdict(set)

        for cl_id, cl in self.centerlines.items():
            # 多个前驱的点
            for pred_id in cl.predecessor_ids:
                junction_points[f"j_{pred_id}"].add(cl_id)

            # 多个后继的点
            for succ_id in cl.successor_ids:
                junction_points[f"j_{succ_id}"].add(cl_id)

        # 创建路口
        int_id = 0
        for key, lane_ids in junction_points.items():
            if len(lane_ids) >= 3:
                # 计算路口中心
                all_coords = []
                for lid in lane_ids:
                    if lid in self.centerlines:
                        all_coords.extend(self.centerlines[lid].coordinates)

                center = mean_position(all_coords) if all_coords else [0.0, 0.0, 0.0]

                self.intersections[f"intersection_{int_id}"] = Intersection(
                    id=f"intersection_{int_id}",
                    center=center,
                    lanes=lane_ids
                )
                int_id += 1

    def generate_output(self) -> Dict:
        """生成最终的JSON输出"""
        self.build_intersections()

        return {
            "version": "1.0",
            "type": "static_map",
            "frames_processed": self.frames_processed,
            "lane_lines": {k: v.to_dict() for k, v in self.lane_lines.items()},
            "centerlines": {k: v.to_dict() for k, v in self.centerlines.items()},
            "road_marks": {k: v.to_dict() for k, v in self.road_marks.items()},
            "traffic_signs": {k: v.to_dict() for k, v in self.traffic_signs.items()},
            "traffic_lights": self.traffic_lights,
            "intersections": {k: v.to_dict() for k, v in self.intersections.items()},
            "statistics": {
                "total_lane_lines": len(self.lane_lines),
                "total_centerlines": len(self.centerlines),
                "total_road_marks": len(self.road_marks),
                "total_traffic_signs": len(self.traffic_signs),
                "total_traffic_lights": len(self.traffic_lights),
                "total_intersections": len(self.intersections),
                "frames_processed": self.frames_processed
            }
        }


# ==================== 主处理流程 ====================

def generate_static_map(data_dir: str, output_file: str):
    """生成静态矢量地图"""

    lane_dir = os.path.join(data_dir, 'result_4dline_V1')
    sign_dir = os.path.join(data_dir, 'result_traffic_sign_V1')

    # 获取所有帧文件
    lane_files = sorted(glob.glob(os.path.join(lane_dir, '*.json')))

    if not lane_files:
        print(f"Error: No lane files found in {lane_dir}")
        return False

    print(f"Found {len(lane_files)} frames to process")

    merger = VectorMapMerger()

    # 处理所有帧
    for i, lane_file in enumerate(lane_files):
        frame_name = os.path.basename(lane_file)
        sign_file = os.path.join(sign_dir, frame_name)

        # 加载车道线数据
        lane_data = json.load(open(lane_file))

        # 加载交通标志数据（如果存在）
        sign_data = None
        if os.path.exists(sign_file):
            sign_data = json.load(open(sign_file))

        # 处理并合并
        merger.process_frame(lane_data, sign_data, frame_idx=i)

        # 进度显示
        if (i + 1) % 50 == 0 or i == len(lane_files) - 1:
            print(f"  Processed {i+1}/{len(lane_files)} frames")

    # 生成输出
    result = merger.generate_output()

    # 保存
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated static vector map: {output_file}")
    print(f"Statistics:")
    print(f"  - Lane lines: {len(merger.lane_lines)}")
    print(f"  - Centerlines: {len(merger.centerlines)}")
    print(f"  - Road marks: {len(merger.road_marks)}")
    print(f"  - Traffic signs: {len(merger.traffic_signs)}")
    print(f"  - Intersections: {len(merger.intersections)}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate static vector map')
    parser.add_argument('--data-dir', type=str, default='./data/00/annotations',
                        help='Data directory containing result_* folders')
    parser.add_argument('--output', type=str, default='./data/static_vector_map.json',
                        help='Output JSON file path')

    args = parser.parse_args()

    generate_static_map(args.data_dir, args.output)


if __name__ == '__main__':
    main()