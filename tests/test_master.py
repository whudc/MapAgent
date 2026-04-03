"""
主 Agent LLM 测试

演示 MapAgent 的 LLM 对话流程
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.master import MasterAgent, create_master_agent
from core.llm_client import LLMClient, LLMConfig


def test_llm_chat():
    """测试 LLM 对话"""
    print("\n" + "=" * 60)
    print("LLM 对话测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    # 创建 Agent
    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # 获取测试坐标
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        test_queries = [
            "这个地图里有什么？",
            f"位置 ({position[0]:.1f}, {position[1]:.1f}) 周围有什么车道？",
            "地图里有多少个路口？",
        ]

        print("\n对话测试:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 对话 {i} ---")
            print(f"用户: {query}")
            try:
                response = agent.chat(query)
                print(f"助手: {response}")
            except Exception as e:
                print(f"错误: {e}")


def test_function_calling():
    """测试 Function Calling"""
    print("\n" + "=" * 60)
    print("Function Calling 测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # 获取测试位置
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # 测试需要调用工具的查询
        tool_queries = [
            {
                "query": f"查找位置 ({position[0]:.1f}, {position[1]:.1f}) 最近的车道",
            },
            {
                "query": f"位置 ({position[0]:.1f}, {position[1]:.1f}) 100米范围内有什么车道？",
            },
            {
                "query": "获取地图概要信息",
            },
        ]

        print("\n工具调用测试:")
        for i, item in enumerate(tool_queries, 1):
            query = item["query"]
            print(f"\n--- 测试 {i} ---")
            print(f"用户: {query}")
            try:
                response = agent.chat(query)
                print(f"助手: {response}")
            except Exception as e:
                print(f"错误: {e}")


def test_multi_turn():
    """测试多轮对话"""
    print("\n" + "=" * 60)
    print("多轮对话测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # 多轮对话
    queries = [
        "这个地图里有什么？",
        "车道类型有哪些？",
        "双实线是什么意思？",
    ]

    print("\n多轮对话:")
    for i, query in enumerate(queries, 1):
        print(f"\n[第{i}轮]")
        print(f"用户: {query}")
        try:
            response = agent.chat(query)
            print(f"助手: {response}")
        except Exception as e:
            print(f"错误: {e}")

    # 查看对话历史
    print("\n对话历史记录:")
    for msg in agent.messages:
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  {role}: {content}")


def test_context_chat():
    """测试带上下文的对话"""
    print("\n" + "=" * 60)
    print("上下文对话测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # 获取测试位置
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # 测试带位置上下文
        print("\n带位置上下文的查询:")
        query = "这个位置附近有什么车道？"
        print(f"用户: {query}")
        print(f"上下文: location={position}")

        try:
            response = agent.chat(query, location=position, radius=100)
            print(f"助手: {response}")
        except Exception as e:
            print(f"错误: {e}")


def test_tool_execution():
    """测试工具直接执行"""
    print("\n" + "=" * 60)
    print("工具执行测试")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # 直接执行工具
    print("\n工具执行测试:")

    # 测试 get_map_summary
    result = agent._execute_tool("get_map_summary", {})
    print(f"\nget_map_summary:")
    print(f"  车道数: {result.get('total_lanes', 'N/A')}")
    print(f"  路口数: {result.get('total_intersections', 'N/A')}")

    # 获取测试位置
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # 测试 find_nearest_lane
        result = agent._execute_tool("find_nearest_lane", {
            "x": position[0],
            "y": position[1]
        })
        print(f"\nfind_nearest_lane:")
        print(f"  最近车道: {result.get('lane_id', 'N/A')}")
        print(f"  距离: {result.get('distance', 'N/A')}m")

        # 测试 get_area_statistics
        result = agent._execute_tool("get_area_statistics", {
            "x": position[0],
            "y": position[1],
            "radius": 100
        })
        print(f"\nget_area_statistics:")
        print(f"  车道数: {result.get('lane_count', 'N/A')}")
        print(f"  车道类型: {result.get('lane_types', {})}")


def main():
    print("=" * 60)
    print("MapAgent LLM 集成测试")
    print("=" * 60)

    # 检查地图文件
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    # 检查 API Key
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"\nLLM 提供商: {provider}")

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("警告: DEEPSEEK_API_KEY 未设置，部分测试可能失败")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("警告: ANTHROPIC_API_KEY 未设置，部分测试可能失败")

    # 运行测试
    test_tool_execution()  # 先测试工具执行（不需要 LLM）
    test_llm_chat()  # 测试 LLM 对话（需要 API Key）
    test_function_calling()  # 测试 Function Calling（需要 API Key）
    test_multi_turn()  # 测试多轮对话（需要 API Key）
    test_context_chat()  # 测试上下文对话（需要 API Key）

    print("\n" + "=" * 60)
    print("基础测试完成 (工具执行)")
    print("=" * 60)
    print("\n要运行完整 LLM 测试，请设置 API Key 并取消注释相关测试")


if __name__ == "__main__":
    main()