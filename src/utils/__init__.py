"""工具函数"""

import sys
from pathlib import Path

# 确保能找到模块
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.geo import (
    calculate_distance,
    calculate_distance_2d,
    point_to_line_distance,
    find_nearest_point,
)

__all__ = [
    "calculate_distance",
    "calculate_distance_2d",
    "point_to_line_distance",
    "find_nearest_point",
]