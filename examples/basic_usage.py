"""
基础使用示例

演示如何加载地图和使用 MapAPI / MasterAgent
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.map_data import MapLoader
from apis.map_api import MapAPI


def demo_map_api():
    """演示 MapAPI 基础用法"""
    print("=" * 50)
    print("MapAPI 基础使用示例")
    print("=" * 50)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    print(f"\n加载地图: {map_path}")

    # 使用 MapLoader 加载
    map_data = MapLoader.load_from_json(str(map_path))

    print(f"\n地图信息:")
    print(f"  - 版本: {map_data.version}")
    print(f"  - 处理帧数: {map_data.frames_processed}")
    print(f"  - 车道线数量: {map_data.get_lane_count()}")
    print(f"  - 中心线数量: {map_data.get_centerline_count()}")
    print(f"  - 路口数量: {map_data.get_intersection_count()}")

    # 使用 MapAPI
    print("\n" + "=" * 50)
    print("MapAPI 查询示例")
    print("=" * 50)

    api = MapAPI(map_file=str(map_path))

    # 获取地图概要
    summary = api.get_map_summary()
    print(f"\n地图概要:")
    print(f"  - 车道类型分布: {summary['lane_type_distribution']}")
    print(f"  - 车道颜色分布: {summary['lane_color_distribution']}")

    # 查询第一个车道
    lane_ids = list(map_data.lane_lines.keys())[:3]
    print(f"\n查询车道信息:")
    for lane_id in lane_ids:
        info = api.get_lane_info(lane_id)
        if info:
            print(f"  - 车道 {lane_id}: 类型={info['type']}, 颜色={info['color']}, 长度={info['length']:.1f}m")

    # 空间查询
    if map_data.lane_lines:
        first_lane = list(map_data.lane_lines.values())[0]
        if first_lane.coordinates:
            position = tuple(first_lane.coordinates[0])
            print(f"\n空间查询 (位置: {position}):")

            # 查找最近车道
            nearest = api.find_nearest_lane(position)
            if nearest:
                print(f"  - 最近车道: {nearest['lane_id']}, 距离: {nearest['distance']:.2f}m")

            # 区域统计
            stats = api.get_area_statistics(position, radius=50)
            print(f"  - 区域内车道数: {stats['lane_count']}")
            print(f"  - 区域内路口数: {stats['intersection_count']}")

    # 查询路口
    if map_data.intersections:
        print(f"\n路口信息:")
        for int_id, intersection in list(map_data.intersections.items())[:3]:
            print(f"  - {int_id}: 中心={intersection.center[:2]}, 关联车道数={len(intersection.lanes)}")


def demo_llm_agent():
    """演示 LLM Agent 用法"""
    print("\n" + "=" * 50)
    print("LLM Agent 示例 (需要 API Key)")
    print("=" * 50)

    from agents.master import create_master_agent

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"

    # 创建 Agent (从 config/settings.py 读取配置)
    agent = create_master_agent(map_file=str(map_path))

    # 检查 LLM 是否可用
    if not agent.llm_client:
        print("\n警告: LLM 客户端未配置")
        print("请设置 API Key:")
        print("  export DEEPSEEK_API_KEY=your-key")
        print("  或在 config/settings.py 中配置")
        print("\n可用工具列表:")
        for tool in agent.get_available_tools():
            print(f"  - {tool}")
        return

    # 测试对话
    print("\n开始对话:")
    queries = [
        "这个地图里有什么？",
        "地图有多少个路口？",
    ]

    for query in queries:
        print(f"\n用户: {query}")
        try:
            response = agent.chat(query)
            print(f"助手: {response}")
        except Exception as e:
            print(f"错误: {e}")

    # 清空历史
    agent.clear_history()


def demo_tool_execution():
    """演示直接执行工具（不需要 LLM）"""
    print("\n" + "=" * 50)
    print("工具执行示例（不需要 API Key）")
    print("=" * 50)

    from agents.master import create_master_agent

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    agent = create_master_agent(map_file=str(map_path))

    # 直接执行工具
    print("\n工具执行结果:")

    # 1. 地图概要
    result = agent._execute_tool("get_map_summary", {})
    print(f"\nget_map_summary:")
    print(f"  总车道: {result['total_lanes']}")
    print(f"  总路口: {result['total_intersections']}")

    # 2. 区域统计
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates'][0]
        result = agent._execute_tool("get_area_statistics", {
            "x": coords[0],
            "y": coords[1],
            "radius": 100
        })
        print(f"\nget_area_statistics (位置 {coords[:2]}):")
        print(f"  车道数: {result['lane_count']}")
        print(f"  车道类型: {result['lane_types']}")

        # 3. 最近车道
        result = agent._execute_tool("find_nearest_lane", {
            "x": coords[0],
            "y": coords[1]
        })
        print(f"\nfind_nearest_lane:")
        print(f"  最近车道: {result['lane_id']}")
        print(f"  距离: {result['distance']}m")


def main():
    # API 基础示例
    demo_map_api()

    # 工具执行示例（不需要 LLM）
    demo_tool_execution()

    # LLM Agent 示例（需要 API Key）
    demo_llm_agent()  # 取消注释来测试 LLM

    print("\n" + "=" * 50)
    print("示例完成")
    print("=" * 50)


if __name__ == "__main__":
    main()