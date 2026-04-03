"""
地理计算工具

提供坐标计算、距离计算、空间查询等功能
"""

import math
from typing import Tuple, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def calculate_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """
    计算两点之间的欧氏距离

    Args:
        p1: 点1坐标
        p2: 点2坐标

    Returns:
        距离
    """
    if HAS_NUMPY:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))
    else:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calculate_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    计算两点之间的2D距离

    Args:
        p1: 点1坐标 (x, y)
        p2: 点2坐标 (x, y)

    Returns:
        距离
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_line_distance(point: Tuple[float, float, float],
                           line_start: Tuple[float, float, float],
                           line_end: Tuple[float, float, float]) -> float:
    """
    计算点到线段的最短距离

    Args:
        point: 点坐标
        line_start: 线段起点
        line_end: 线段终点

    Returns:
        最短距离
    """
    if HAS_NUMPY:
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)

        # 向量计算
        ab = b - a
        ap = p - a

        # 投影参数
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-10:
            return float(np.linalg.norm(p - a))

        t = np.dot(ap, ab) / ab_sq
        t = max(0, min(1, t))

        # 最近点
        nearest = a + t * ab
        return float(np.linalg.norm(p - nearest))
    else:
        # 纯 Python 实现
        ab = [line_end[i] - line_start[i] for i in range(min(3, len(line_start)))]
        ap = [point[i] - line_start[i] for i in range(min(3, len(point)))]

        ab_sq = sum(x * x for x in ab)
        if ab_sq < 1e-10:
            return calculate_distance(point, line_start)

        t = sum(a * b for a, b in zip(ap, ab)) / ab_sq
        t = max(0, min(1, t))

        nearest = [line_start[i] + t * ab[i] for i in range(len(ab))]
        return calculate_distance(point, tuple(nearest))


def point_to_polyline_distance(point: Tuple[float, float, float],
                                polyline: List[Tuple[float, float, float]]) -> Tuple[float, int]:
    """
    计算点到折线的最短距离

    Args:
        point: 点坐标
        polyline: 折线点列表

    Returns:
        (最短距离, 最近线段索引)
    """
    if len(polyline) < 2:
        return float('inf'), -1

    min_dist = float('inf')
    min_idx = 0

    for i in range(len(polyline) - 1):
        dist = point_to_line_distance(point, polyline[i], polyline[i + 1])
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    return min_dist, min_idx


def find_nearest_point(target: Tuple[float, float, float],
                        points: List[Tuple[float, float, float]]) -> Tuple[int, float]:
    """
    在点集中找到距离目标最近的点

    Args:
        target: 目标点
        points: 点列表

    Returns:
        (最近点索引, 距离)
    """
    if not points:
        return -1, float('inf')

    min_dist = float('inf')
    min_idx = 0

    for i, p in enumerate(points):
        dist = calculate_distance(target, p)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    return min_idx, min_dist


def calculate_heading(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    计算从 p1 到 p2 的航向角

    Args:
        p1: 起点 (x, y)
        p2: 终点 (x, y)

    Returns:
        航向角（度），0度为东，逆时针为正
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def get_polyline_heading(polyline: List[Tuple[float, float, float]],
                         segment_index: int = 0) -> float:
    """
    获取折线在指定位置的航向角

    Args:
        polyline: 折线点列表
        segment_index: 线段索引

    Returns:
        航向角（度）
    """
    if len(polyline) < 2:
        return 0.0

    idx = min(segment_index, len(polyline) - 2)
    p1 = polyline[idx]
    p2 = polyline[idx + 1]

    return calculate_heading((p1[0], p1[1]), (p2[0], p2[1]))


def project_point_on_polyline(point: Tuple[float, float, float],
                               polyline: List[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], int, float]:
    """
    将点投影到折线上

    Args:
        point: 点坐标
        polyline: 折线点列表

    Returns:
        (投影点坐标, 所在线段索引, 线段内位置比例 0-1)
    """
    if len(polyline) < 2:
        return point, 0, 0.0

    min_dist = float('inf')
    best_proj = point
    best_idx = 0
    best_t = 0.0

    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]

        # 计算投影
        if HAS_NUMPY:
            a = np.array(p1)
            b = np.array(p2)
            p = np.array(point)
            ab = b - a
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-10:
                proj = a
                t = 0.0
            else:
                t = np.dot(p - a, ab) / ab_sq
                t = max(0, min(1, t))
                proj = a + t * ab
        else:
            dim = min(3, len(p1))
            ab = [p2[j] - p1[j] for j in range(dim)]
            ap = [point[j] - p1[j] for j in range(dim)]
            ab_sq = sum(x * x for x in ab)

            if ab_sq < 1e-10:
                proj = p1
                t = 0.0
            else:
                t = sum(a * b for a, b in zip(ap, ab)) / ab_sq
                t = max(0, min(1, t))
                proj = tuple(p1[j] + t * ab[j] for j in range(dim))

        # 计算距离
        dist = calculate_distance(point, tuple(proj))

        if dist < min_dist:
            min_dist = dist
            best_proj = tuple(proj) if HAS_NUMPY else tuple(proj)
            best_idx = i
            best_t = t

    return best_proj, best_idx, best_t


def line_segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    判断两条2D线段是否相交

    Args:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点

    Returns:
        是否相交
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # 使用叉积判断
    if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
        return True

    return False


def polylines_intersect(polyline1: List[Tuple[float, float, float]],
                        polyline2: List[Tuple[float, float, float]]) -> bool:
    """
    判断两条折线是否相交

    Args:
        polyline1: 第一条折线
        polyline2: 第二条折线

    Returns:
        是否相交
    """
    if len(polyline1) < 2 or len(polyline2) < 2:
        return False

    for i in range(len(polyline1) - 1):
        for j in range(len(polyline2) - 1):
            p1 = (polyline1[i][0], polyline1[i][1])
            p2 = (polyline1[i + 1][0], polyline1[i + 1][1])
            p3 = (polyline2[j][0], polyline2[j][1])
            p4 = (polyline2[j + 1][0], polyline2[j + 1][1])

            if line_segments_intersect(p1, p2, p3, p4):
                return True

    return False


def polyline_bounding_box(polyline: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """
    计算折线的包围盒

    Args:
        polyline: 折线点列表

    Returns:
        (min_x, min_y, max_x, max_y)
    """
    if not polyline:
        return (0.0, 0.0, 0.0, 0.0)

    xs = [p[0] for p in polyline]
    ys = [p[1] for p in polyline]

    return (min(xs), min(ys), max(xs), max(ys))


def point_in_bbox(point: Tuple[float, float, float],
                  bbox: Tuple[float, float, float, float]) -> bool:
    """
    判断点是否在包围盒内

    Args:
        point: 点坐标
        bbox: 包围盒 (min_x, min_y, max_x, max_y)

    Returns:
        是否在包围盒内
    """
    x, y = point[0], point[1]
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def bboxes_intersect(bbox1: Tuple[float, float, float, float],
                     bbox2: Tuple[float, float, float, float]) -> bool:
    """
    判断两个包围盒是否相交

    Args:
        bbox1: 第一个包围盒
        bbox2: 第二个包围盒

    Returns:
        是否相交
    """
    return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])


def is_point_in_polygon(point: Tuple[float, float],
                        polygon: List[Tuple[float, float]]) -> bool:
    """
    判断点是否在多边形内

    Args:
        point: 点坐标
        polygon: 多边形顶点列表

    Returns:
        是否在多边形内
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x, y = point

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


def interpolate_line(line: List[Tuple[float, float, float]],
                     interval: float) -> List[Tuple[float, float, float]]:
    """
    按指定间隔在线上插值点

    Args:
        line: 线段点列表
        interval: 插值间隔

    Returns:
        插值后的点列表
    """
    if len(line) < 2:
        return line

    result = [line[0]]
    accumulated = 0.0

    for i in range(len(line) - 1):
        p1, p2 = line[i], line[i + 1]
        segment_length = calculate_distance(p1, p2)

        while accumulated + segment_length >= interval:
            # 计算插值点
            ratio = (interval - accumulated) / segment_length
            new_point = tuple(
                p1[j] + ratio * (p2[j] - p1[j]) for j in range(min(3, len(p1)))
            )
            result.append(new_point)
            segment_length -= (interval - accumulated)
            accumulated = 0.0
            p1 = new_point

        accumulated += segment_length

    result.append(line[-1])
    return result


def smooth_polyline(polyline: List[Tuple[float, float, float]],
                    window: int = 3) -> List[Tuple[float, float, float]]:
    """
    平滑折线（移动平均）

    Args:
        polyline: 折线点列表
        window: 窗口大小

    Returns:
        平滑后的折线
    """
    if len(polyline) < window:
        return polyline

    result = [polyline[0]]  # 保持起点

    half_window = window // 2
    for i in range(1, len(polyline) - 1):
        start = max(0, i - half_window)
        end = min(len(polyline), i + half_window + 1)

        avg = tuple(
            sum(polyline[j][k] for j in range(start, end)) / (end - start)
            for k in range(min(3, len(polyline[0])))
        )
        result.append(avg)

    result.append(polyline[-1])  # 保持终点
    return result