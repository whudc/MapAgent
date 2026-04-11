"""
 Agent LLM Testing

Demo MapAgent  LLM onWorkflow
"""

import sys
import os
from pathlib import Path

# AddProjectRootDirectorytoPath
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.master import MasterAgent, create_master_agent
from core.llm_client import LLMClient, LLMConfig


def test_llm_chat():
    """Testing LLM on"""
    print("\n" + "=" * 60)
    print("LLM onTesting")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    # Create Agent
    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # GetTestingCoordinate
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        test_queries = [
            "hereMaphavewhat？",
            f"location ({position[0]:.1f}, {position[1]:.1f}) circumferencehavewhatlanes？",
            "Maphaveintersection？",
        ]

        print("\nonTesting:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- on {i} ---")
            print(f": {query}")
            try:
                response = agent.chat(query)
                print(f": {response}")
            except Exception as e:
                print(f"Error: {e}")


def test_function_calling():
    """Testing Function Calling"""
    print("\n" + "=" * 60)
    print("Function Calling Testing")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # GetTestinglocation
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # TestingneededToolquery
        tool_queries = [
            {
                "query": f"Findlocation ({position[0]:.1f}, {position[1]:.1f}) lanes",
            },
            {
                "query": f"location ({position[0]:.1f}, {position[1]:.1f}) 100Rangeincenterhavewhatlanes？",
            },
            {
                "query": "GetMapwillinfo",
            },
        ]

        print("\nToolTesting:")
        for i, item in enumerate(tool_queries, 1):
            query = item["query"]
            print(f"\n--- Testing {i} ---")
            print(f": {query}")
            try:
                response = agent.chat(query)
                print(f": {response}")
            except Exception as e:
                print(f"Error: {e}")


def test_multi_turn():
    """Testingon"""
    print("\n" + "=" * 60)
    print("onTesting")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # on
    queries = [
        "hereMaphavewhat？",
        "lanestypehave？",
        "doubleSolid lineiswhat？",
    ]

    print("\non:")
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}]")
        print(f": {query}")
        try:
            response = agent.chat(query)
            print(f": {response}")
        except Exception as e:
            print(f"Error: {e}")

    # Viewonhistory
    print("\nonhistory:")
    for msg in agent.messages:
        role = "" if msg["role"] == "user" else ""
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  {role}: {content}")


def test_context_chat():
    """Testingtexton"""
    print("\n" + "=" * 60)
    print("textonTesting")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # GetTestinglocation
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # Testinglocationtext
        print("\nlocationtextquery:")
        query = "herelocationNearbyhavewhatlanes？"
        print(f": {query}")
        print(f"text: location={position}")

        try:
            response = agent.chat(query, location=position, radius=100)
            print(f": {response}")
        except Exception as e:
            print(f"Error: {e}")


def test_tool_execution():
    """TestingToolstraight lineinterfaceExecute"""
    print("\n" + "=" * 60)
    print("ToolExecuteTesting")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    )

    # straight lineinterfaceExecuteTool
    print("\nToolExecuteTesting:")

    # Testing get_map_summary
    result = agent._execute_tool("get_map_summary", {})
    print(f"\nget_map_summary:")
    print(f"  lanes: {result.get('total_lanes', 'N/A')}")
    print(f"  intersection: {result.get('total_intersections', 'N/A')}")

    # GetTestinglocation
    lane_ids = agent.map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = agent.map_api.get_lane_info(lane_ids[0])
        coords = lane_info['coordinates']
        position = tuple(coords[len(coords)//2])

        # Testing find_nearest_lane
        result = agent._execute_tool("find_nearest_lane", {
            "x": position[0],
            "y": position[1]
        })
        print(f"\nfind_nearest_lane:")
        print(f"  lanes: {result.get('lane_id', 'N/A')}")
        print(f"  Distance: {result.get('distance', 'N/A')}m")

        # Testing get_area_statistics
        result = agent._execute_tool("get_area_statistics", {
            "x": position[0],
            "y": position[1],
            "radius": 100
        })
        print(f"\nget_area_statistics:")
        print(f"  lanes: {result.get('lane_count', 'N/A')}")
        print(f"  lanestype: {result.get('lane_types', {})}")


def main():
    print("=" * 60)
    print("MapAgent LLM IntegrationTesting")
    print("=" * 60)

    # CheckMapFile
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    # Check API Key
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    print(f"\nLLM notificationQuotient: {provider}")

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Warning: DEEPSEEK_API_KEY notSet，Testingcanfail")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY notSet，Testingcanfail")

    # RunTesting
    test_tool_execution()  # TestingToolExecute（notneeded LLM）
    test_llm_chat()  # Testing LLM on（needed API Key）
    test_function_calling()  # Testing Function Calling（needed API Key）
    test_multi_turn()  # Testingon（needed API Key）
    test_context_chat()  # Testingtexton（needed API Key）

    print("\n" + "=" * 60)
    print("Testingto (ToolExecute)")
    print("=" * 60)
    print("\nwillRuncomplete LLM Testing，Set API Key andCancelcommentsTesting")


if __name__ == "__main__":
    main()