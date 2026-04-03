"""
Phase 3 子 Agent 测试

测试场景理解、行为分析、路径规划 Agent
"""

import sys
from pathlib import Path

# 添加 src 到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from apis.map_api import MapAPI
from agents.base import AgentContext
from agents.scene import SceneAgent
from agents.behavior import BehaviorAgent
from agents.path import PathAgent


def test_scene_agent():
    """测试场景理解 Agent"""
    print("\n" + "=" * 60)
    print("场景理解 Agent 测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = SceneAgent(context)

    # 测试工具定义
    print(f"\n可用工具:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # 测试1: 分析指定位置
    print("\n测试1: 分析指定位置场景")
    lane_ids = map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            mid = len(coords) // 2
            position = (coords[mid][0], coords[mid][1], coords[mid][2] if len(coords[mid]) > 2 else 0)

            result = agent.process(
                query="这个位置有什么车道?",
                location=position,
                radius=50
            )
            print(f"  结果: {result.get('summary', 'N/A')}")

    # 测试2: 查询路口
    print("\n测试2: 查询路口信息")
    int_ids = map_api.get_all_intersection_ids()
    if int_ids:
        result = agent.process(
            query="这个路口的结构?",
            intersection_id=int_ids[0]
        )
        print(f"  结果: {result.get('summary', 'N/A')}")

    # 测试3: 直接调用工具
    print("\n测试3: 调用工具")
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            result = agent.execute_tool(
                "get_lane_count_by_type",
                x=coords[0][0],
                y=coords[0][1],
                radius=100
            )
            print(f"  区域内车道类型: {result.get('lane_types', {})}")


def test_behavior_agent():
    """测试行为分析 Agent"""
    print("\n" + "=" * 60)
    print("行为分析 Agent 测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = BehaviorAgent(context)

    # 测试工具定义
    print(f"\n可用工具:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # 测试1: 匹配车辆到车道
    print("\n测试1: 匹配车辆到车道")
    lane_ids = map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            z_val = coords[0][2] if len(coords[0]) > 2 else 0
            position = (coords[0][0] + 2, coords[0][1] + 2, z_val)

            result = agent.process(
                query="这辆车在哪条车道?",
                location=position,
                heading=45.0,
                speed=10.0
            )
            print(f"  预测行为: {result.get('predicted_action', 'N/A')}")
            print(f"  置信度: {result.get('confidence', 0):.0%}")
            print(f"  依据: {result.get('reasoning', 'N/A')}")
            print(f"  风险等级: {result.get('risk_level', 'N/A')}")

    # 测试2: 变道可能性
    print("\n测试2: 变道可能性分析")
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            result = agent.execute_tool(
                "get_lane_change_possibility",
                x=coords[len(coords)//2][0],
                y=coords[len(coords)//2][1],
                heading=0
            )
            print(f"  可向左变道: {result.get('can_change_left', False)}")
            print(f"  可向右变道: {result.get('can_change_right', False)}")

    # 测试3: 碰撞风险分析
    print("\n测试3: 碰撞风险分析")
    result = agent.execute_tool(
        "analyze_collision_risk",
        vehicle1_x=0, vehicle1_y=0, vehicle1_heading=0,
        vehicle2_x=20, vehicle2_y=5, vehicle2_heading=10
    )
    print(f"  距离: {result.get('distance', 0)}m")
    print(f"  碰撞风险: {result.get('collision_risk', 'N/A')}")


def test_path_agent():
    """测试路径规划 Agent"""
    print("\n" + "=" * 60)
    print("路径规划 Agent 测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = PathAgent(context)

    # 测试工具定义
    print(f"\n可用工具:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # 测试1: 查找附近目的地
    print("\n测试1: 查找附近目的地")
    lane_ids = map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            result = agent.execute_tool(
                "find_nearby_destination",
                x=coords[0][0],
                y=coords[0][1],
                radius=200
            )
            print(f"  找到路口数: {result.get('total_intersections', 0)}")
            print(f"  找到车道数: {result.get('total_lanes', 0)}")

    # 测试2: 路径规划
    print("\n测试2: 路径规划")
    # 找两个距离较远的位置
    cl_ids = map_api.get_all_centerline_ids()
    if len(cl_ids) >= 2:
        start_cl = map_api.get_centerline_info(cl_ids[0])
        end_cl = map_api.get_centerline_info(cl_ids[-1])

        if start_cl and end_cl and start_cl['coordinates'] and end_cl['coordinates']:
            start_pos = start_cl['coordinates'][0]
            end_pos = end_cl['coordinates'][0]

            result = agent.process(
                query="规划路线",
                origin=(start_pos[0], start_pos[1], start_pos[2] if len(start_pos) > 2 else 0),
                destination=(end_pos[0], end_pos[1], end_pos[2] if len(end_pos) > 2 else 0)
            )
            print(f"  建议: {result.get('advice', 'N/A')[:100]}...")
            print(f"  距离: {result.get('distance', 0):.0f}m")
            print(f"  预估时间: {result.get('estimated_time', 0):.1f}分钟")

    # 测试3: 估算行程时间
    print("\n测试3: 估算行程时间")
    if cl_ids:
        result = agent.execute_tool(
            "estimate_travel_time",
            path=cl_ids[:5],
            speed=15
        )
        print(f"  距离: {result.get('distance_meters', 0):.1f}m")
        print(f"  时间: {result.get('time_minutes', 0):.1f}分钟")


def main():
    print("=" * 60)
    print("MapAgent Phase 3 测试")
    print("=" * 60)

    # 检查地图文件
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    # 运行测试
    test_scene_agent()
    test_behavior_agent()
    test_path_agent()

    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()