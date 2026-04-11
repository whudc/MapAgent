"""
Phase 3  Agent Testing

TestingSceneUnderstanding、behavioranalysis、Pathplan Agent
"""

import sys
from pathlib import Path

# Add src toPath
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from apis.map_api import MapAPI
from agents.base import AgentContext
from agents.scene import SceneAgent
from agents.behavior import BehaviorAgent
from agents.path import PathAgent


def test_scene_agent():
    """TestingSceneUnderstanding Agent"""
    print("\n" + "=" * 60)
    print("SceneUnderstanding Agent Testing")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = SceneAgent(context)

    # TestingTool
    print(f"\navailableTool:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # Testing1: analysislocation
    print("\nTesting1: analysislocationScene")
    lane_ids = map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            mid = len(coords) // 2
            position = (coords[mid][0], coords[mid][1], coords[mid][2] if len(coords[mid]) > 2 else 0)

            result = agent.process(
                query="herelocationhavewhatlanes?",
                location=position,
                radius=50
            )
            print(f"  Result: {result.get('summary', 'N/A')}")

    # Testing2: queryintersection
    print("\nTesting2: queryintersectioninfo")
    int_ids = map_api.get_all_intersection_ids()
    if int_ids:
        result = agent.process(
            query="hereintersectionstructure?",
            intersection_id=int_ids[0]
        )
        print(f"  Result: {result.get('summary', 'N/A')}")

    # Testing3: straight lineinterfaceTool
    print("\nTesting3: Tool")
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
            print(f"  areadomainincenterlanestype: {result.get('lane_types', {})}")


def test_behavior_agent():
    """Testingbehavioranalysis Agent"""
    print("\n" + "=" * 60)
    print("behavioranalysis Agent Testing")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = BehaviorAgent(context)

    # TestingTool
    print(f"\navailableTool:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # Testing1: Matchingvehicletolanes
    print("\nTesting1: Matchingvehicletolanes")
    lane_ids = map_api.get_all_lane_ids()
    if lane_ids:
        lane_info = map_api.get_lane_info(lane_ids[0])
        if lane_info and lane_info['coordinates']:
            coords = lane_info['coordinates']
            z_val = coords[0][2] if len(coords[0]) > 2 else 0
            position = (coords[0][0] + 2, coords[0][1] + 2, z_val)

            result = agent.process(
                query="herevehiclesunderlanes?",
                location=position,
                heading=45.0,
                speed=10.0
            )
            print(f"  Predictbehavior: {result.get('predicted_action', 'N/A')}")
            print(f"  : {result.get('confidence', 0):.0%}")
            print(f"  : {result.get('reasoning', 'N/A')}")
            print(f"  risketc.level: {result.get('risk_level', 'N/A')}")

    # Testing2: lane changecan
    print("\nTesting2: lane changecananalysis")
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
            print(f"  cantoleftlane change: {result.get('can_change_left', False)}")
            print(f"  cantorightlane change: {result.get('can_change_right', False)}")

    # Testing3: collisionriskanalysis
    print("\nTesting3: collisionriskanalysis")
    result = agent.execute_tool(
        "analyze_collision_risk",
        vehicle1_x=0, vehicle1_y=0, vehicle1_heading=0,
        vehicle2_x=20, vehicle2_y=5, vehicle2_heading=10
    )
    print(f"  Distance: {result.get('distance', 0)}m")
    print(f"  collisionrisk: {result.get('collision_risk', 'N/A')}")


def test_path_agent():
    """TestingPathplan Agent"""
    print("\n" + "=" * 60)
    print("Pathplan Agent Testing")
    print("=" * 60)

    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    map_api = MapAPI(map_file=str(map_path))
    context = AgentContext(map_api=map_api)

    agent = PathAgent(context)

    # TestingTool
    print(f"\navailableTool:")
    for tool in agent.get_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # Testing1: FindNearbyly
    print("\nTesting1: FindNearbyly")
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
            print(f"  tointersection: {result.get('total_intersections', 0)}")
            print(f"  tolanes: {result.get('total_lanes', 0)}")

    # Testing2: Pathplan
    print("\nTesting2: Pathplan")
    # TwoDistancelocation
    cl_ids = map_api.get_all_centerline_ids()
    if len(cl_ids) >= 2:
        start_cl = map_api.get_centerline_info(cl_ids[0])
        end_cl = map_api.get_centerline_info(cl_ids[-1])

        if start_cl and end_cl and start_cl['coordinates'] and end_cl['coordinates']:
            start_pos = start_cl['coordinates'][0]
            end_pos = end_cl['coordinates'][0]

            result = agent.process(
                query="planroute",
                origin=(start_pos[0], start_pos[1], start_pos[2] if len(start_pos) > 2 else 0),
                destination=(end_pos[0], end_pos[1], end_pos[2] if len(end_pos) > 2 else 0)
            )
            print(f"  suggestion: {result.get('advice', 'N/A')[:100]}...")
            print(f"  Distance: {result.get('distance', 0):.0f}m")
            print(f"  time: {result.get('estimated_time', 0):.1f}")

    # Testing3: estimationprogramtime
    print("\nTesting3: estimationprogramtime")
    if cl_ids:
        result = agent.execute_tool(
            "estimate_travel_time",
            path=cl_ids[:5],
            speed=15
        )
        print(f"  Distance: {result.get('distance_meters', 0):.1f}m")
        print(f"  time: {result.get('time_minutes', 0):.1f}")


def main():
    print("=" * 60)
    print("MapAgent Phase 3 Testing")
    print("=" * 60)

    # CheckMapFile
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    # RunTesting
    test_scene_agent()
    test_behavior_agent()
    test_path_agent()

    print("\n" + "=" * 60)
    print("haveTestingto")
    print("=" * 60)


if __name__ == "__main__":
    main()