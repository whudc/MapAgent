"""
Map API 功能测试

验证 Phase 2 实现的所有功能
"""

import sys
from pathlib import Path

# 添加 src 到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.map_data import MapLoader
from apis.map_api import MapAPI


def test_basic_queries(api: MapAPI):
    """测试基础查询"""
    print("\n" + "=" * 60)
    print("基础查询测试")
    print("=" * 60)

    # 获取地图概要
    summary = api.get_map_summary()
    print(f"\n地图概要:")
    print(f"  - 车道数: {summary['total_lanes']}")
    print(f"  - 中心线数: {summary['total_centerlines']}")
    print(f"  - 路口数: {summary['total_intersections']}")

    # 获取车道信息
    lane_ids = api.get_all_lane_ids()[:3]
    print(f"\n车道信息查询:")
    for lane_id in lane_ids:
        info = api.get_lane_info(lane_id)
        if info:
            print(f"  车道 {lane_id}:")
            print(f"    - 类型: {info['type']}, 颜色: {info['color']}")
            print(f"    - 长度: {info['length']:.1f}m")
            print(f"    - 坐标点数: {len(info['coordinates'])}")

    # 获取路口信息
    int_ids = api.get_all_intersection_ids()[:2]
    print(f"\n路口信息查询:")
    for int_id in int_ids:
        info = api.get_intersection_info(int_id)
        if info:
            print(f"  路口 {int_id}:")
            print(f"    - 中心: [{info['center'][0]:.1f}, {info['center'][1]:.1f}]")
            print(f"    - 关联车道数: {info['lane_count']}")


def test_topology_queries(api: MapAPI):
    """测试拓扑查询"""
    print("\n" + "=" * 60)
    print("拓扑查询测试")
    print("=" * 60)

    # 获取连接车道
    lane_ids = api.get_all_lane_ids()[:5]
    print(f"\n连接车道查询:")
    for lane_id in lane_ids:
        connected = api.get_connected_lanes(lane_id)
        if connected:
            print(f"  车道 {lane_id}:")
            print(f"    - 中心线ID: {connected['centerline_id']}")
            print(f"    - 前驱数: {len(connected['predecessors'])}")
            print(f"    - 后继数: {len(connected['successors'])}")
            break

    # 获取中心线信息
    cl_ids = api.get_all_centerline_ids()[:3]
    print(f"\n中心线信息查询:")
    for cl_id in cl_ids:
        info = api.get_centerline_info(cl_id)
        if info and (info['left_boundary_id'] or info['right_boundary_id']):
            print(f"  中心线 {cl_id}:")
            print(f"    - 左边界: {info['left_boundary_id']}")
            print(f"    - 右边界: {info['right_boundary_id']}")
            print(f"    - 前驱: {info['predecessor_ids'][:3]}")
            print(f"    - 后继: {info['successor_ids'][:3]}")
            break


def test_spatial_queries(api: MapAPI):
    """测试空间查询"""
    print("\n" + "=" * 60)
    print("空间查询测试")
    print("=" * 60)

    # 获取一个测试位置
    lane_ids = api.get_all_lane_ids()
    if not lane_ids:
        print("没有车道数据")
        return

    lane_info = api.get_lane_info(lane_ids[0])
    if not lane_info or not lane_info['coordinates']:
        print("没有坐标数据")
        return

    # 使用车道中点作为测试位置
    coords = lane_info['coordinates']
    mid_idx = len(coords) // 2
    position = tuple(coords[mid_idx])

    print(f"\n测试位置: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

    # 查找最近车道
    nearest = api.find_nearest_lane(position, max_distance=100)
    if nearest:
        print(f"\n最近车道:")
        print(f"  - 车道ID: {nearest['lane_id']}")
        print(f"  - 类型: {nearest['type']}")
        print(f"  - 距离: {nearest['distance']}m")

    # 查找最近中心线
    nearest_cl = api.find_nearest_centerline(position, max_distance=100)
    if nearest_cl:
        print(f"\n最近中心线:")
        print(f"  - 中心线ID: {nearest_cl['centerline_id']}")
        print(f"  - 距离: {nearest_cl['distance']}m")

    # 查找区域内车道
    lanes = api.find_lanes_in_area(position, radius=50)
    print(f"\n区域内车道 (半径50m): {len(lanes)}条")

    # 查找区域内路口
    intersections = api.find_intersections_in_area(position, radius=100)
    print(f"区域内路口 (半径100m): {len(intersections)}个")

    # 区域统计
    stats = api.get_area_statistics(position, radius=100)
    print(f"\n区域统计 (半径100m):")
    print(f"  - 车道数: {stats['lane_count']}")
    print(f"  - 中心线数: {stats['centerline_count']}")
    print(f"  - 路口数: {stats['intersection_count']}")
    print(f"  - 总车道长度: {stats['total_lane_length']}m")


def test_vehicle_matching(api: MapAPI):
    """测试车辆匹配"""
    print("\n" + "=" * 60)
    print("车辆匹配测试")
    print("=" * 60)

    # 获取一个测试位置
    lane_ids = api.get_all_lane_ids()
    if not lane_ids:
        return

    lane_info = api.get_lane_info(lane_ids[0])
    if not lane_info or not lane_info['coordinates']:
        return

    coords = lane_info['coordinates']
    # 在车道上找一个点，稍微偏移
    test_pos = (coords[0][0] + 2, coords[0][1] + 2, coords[0][2])

    print(f"\n测试位置: ({test_pos[0]:.1f}, {test_pos[1]:.1f})")

    # 匹配车辆到车道
    match = api.match_vehicle_to_lane(test_pos, heading=45.0, max_distance=20)
    if match:
        print(f"\n匹配结果:")
        print(f"  - 中心线ID: {match['centerline_id']}")
        print(f"  - 距离: {match['distance']}m")
        print(f"  - 车道航向: {match['heading']}°")
        print(f"  - 行驶方向: {'正向' if match['is_forward'] else '逆向'}")
    else:
        print("未匹配到车道")


def test_path_finding(api: MapAPI):
    """测试路径查找"""
    print("\n" + "=" * 60)
    print("路径查找测试")
    print("=" * 60)

    # 找两个有连接关系的车道
    cl_ids = api.get_all_centerline_ids()
    if len(cl_ids) < 2:
        print("中心线数据不足")
        return

    # 找有后继的中心线
    start_cl = None
    end_cl = None

    for cl_id in cl_ids:
        info = api.get_centerline_info(cl_id)
        if info and info['successor_ids']:
            start_cl = cl_id
            end_cl = info['successor_ids'][0]
            break

    if not start_cl:
        print("没有找到可连接的路径")
        return

    print(f"\n尝试从中心线 {start_cl} 到 {end_cl}")

    # 需要车道ID
    start_info = api.get_centerline_info(start_cl)
    start_lane = start_info['right_boundary_id'] or start_info['left_boundary_id'] or start_cl

    end_info = api.get_centerline_info(end_cl)
    end_lane = end_info['right_boundary_id'] or end_info['left_boundary_id'] or end_cl

    path = api.find_path_between_lanes(start_lane, end_lane)
    if path:
        print(f"找到路径: {' -> '.join(path)}")
    else:
        print("未找到路径")


def main():
    print("=" * 60)
    print("MapAgent Phase 2 测试")
    print("=" * 60)

    # 加载地图
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    print(f"\n加载地图: {map_path}")
    api = MapAPI(map_file=str(map_path))

    # 运行测试
    test_basic_queries(api)
    test_topology_queries(api)
    test_spatial_queries(api)
    test_vehicle_matching(api)
    test_path_finding(api)

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()