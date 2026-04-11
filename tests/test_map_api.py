"""
Map API FeatureTesting

Validation Phase 2 achievinghaveFeature
"""

import sys
from pathlib import Path

# Add src toPath
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.map_data import Map Loader
from apis.map_api import MapAPI


def test_basic_queries(api: MapAPI):
    """Testingquery"""
    print("\n" + "=" * 60)
    print("queryTesting")
    print("=" * 60)

    # GetMapwill
    summary = api.get_map_summary()
    print(f"\nMapwill:")
    print(f"  - lanes: {summary['total_lanes']}")
    print(f"  - Center: {summary['total_centerlines']}")
    print(f"  - intersection: {summary['total_intersections']}")

    # Getlanesinfo
    lane_ids = api.get_all_lane_ids()[:3]
    print(f"\nlanesinfoquery:")
    for lane_id in lane_ids:
        info = api.get_lane_info(lane_id)
        if info:
            print(f"  lanes {lane_id}:")
            print(f"    - type: {info['type']}, Color: {info['color']}")
            print(f"    - length: {info['length']:.1f}m")
            print(f"    - CoordinatePoint: {len(info['coordinates'])}")

    # Getintersectioninfo
    int_ids = api.get_all_intersection_ids()[:2]
    print(f"\nintersectioninfoquery:")
    for int_id in int_ids:
        info = api.get_intersection_info(int_id)
        if info:
            print(f"  intersection {int_id}:")
            print(f"    - Center: [{info['center'][0]:.1f}, {info['center'][1]:.1f}]")
            print(f"    - Associationlanes: {info['lane_count']}")


def test_topology_queries(api: MapAPI):
    """Testingquery"""
    print("\n" + "=" * 60)
    print("queryTesting")
    print("=" * 60)

    # GetConnectlanes
    lane_ids = api.get_all_lane_ids()[:5]
    print(f"\nConnectlanesquery:")
    for lane_id in lane_ids:
        connected = api.get_connected_lanes(lane_id)
        if connected:
            print(f"  lanes {lane_id}:")
            print(f"    - CenterID: {connected['centerline_id']}")
            print(f"    - before: {len(connected['predecessors'])}")
            print(f"    - after: {len(connected['successors'])}")
            break

    # GetCenterinfo
    cl_ids = api.get_all_centerline_ids()[:3]
    print(f"\nCenterinfoquery:")
    for cl_id in cl_ids:
        info = api.get_centerline_info(cl_id)
        if info and (info['left_boundary_id'] or info['right_boundary_id']):
            print(f"  Center {cl_id}:")
            print(f"    - leftBoundary: {info['left_boundary_id']}")
            print(f"    - rightBoundary: {info['right_boundary_id']}")
            print(f"    - before: {info['predecessor_ids'][:3]}")
            print(f"    - after: {info['successor_ids'][:3]}")
            break


def test_spatial_queries(api: MapAPI):
    """TestingSpacequery"""
    print("\n" + "=" * 60)
    print("SpacequeryTesting")
    print("=" * 60)

    # GetaTestinglocation
    lane_ids = api.get_all_lane_ids()
    if not lane_ids:
        print("havelanesdata")
        return

    lane_info = api.get_lane_info(lane_ids[0])
    if not lane_info or not lane_info['coordinates']:
        print("haveCoordinatedata")
        return

    # UselanescenterPointTestinglocation
    coords = lane_info['coordinates']
    mid_idx = len(coords) // 2
    position = tuple(coords[mid_idx])

    print(f"\nTestinglocation: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")

    # Findlanes
    nearest = api.find_nearest_lane(position, max_distance=100)
    if nearest:
        print(f"\nlanes:")
        print(f"  - lanesID: {nearest['lane_id']}")
        print(f"  - type: {nearest['type']}")
        print(f"  - Distance: {nearest['distance']}m")

    # FindCenter
    nearest_cl = api.find_nearest_centerline(position, max_distance=100)
    if nearest_cl:
        print(f"\nCenter:")
        print(f"  - CenterID: {nearest_cl['centerline_id']}")
        print(f"  - Distance: {nearest_cl['distance']}m")

    # Findareadomainincenterlanes
    lanes = api.find_lanes_in_area(position, radius=50)
    print(f"\nareadomainincenterlanes (Radius50m): {len(lanes)}")

    # Findareadomainincenterintersection
    intersections = api.find_intersections_in_area(position, radius=100)
    print(f"areadomainincenterintersection (Radius100m): {len(intersections)}")

    # areadomain
    stats = api.get_area_statistics(position, radius=100)
    print(f"\nareadomain (Radius100m):")
    print(f"  - lanes: {stats['lane_count']}")
    print(f"  - Center: {stats['centerline_count']}")
    print(f"  - intersection: {stats['intersection_count']}")
    print(f"  - laneslength: {stats['total_lane_length']}m")


def test_vehicle_matching(api: MapAPI):
    """TestingvehicleMatching"""
    print("\n" + "=" * 60)
    print("vehicleMatchingTesting")
    print("=" * 60)

    # GetaTestinglocation
    lane_ids = api.get_all_lane_ids()
    if not lane_ids:
        return

    lane_info = api.get_lane_info(lane_ids[0])
    if not lane_info or not lane_info['coordinates']:
        return

    coords = lane_info['coordinates']
    # underlanesaPoint，
    test_pos = (coords[0][0] + 2, coords[0][1] + 2, coords[0][2])

    print(f"\nTestinglocation: ({test_pos[0]:.1f}, {test_pos[1]:.1f})")

    # Matchingvehicletolanes
    match = api.match_vehicle_to_lane(test_pos, heading=45.0, max_distance=20)
    if match:
        print(f"\nMatchingResult:")
        print(f"  - CenterID: {match['centerline_id']}")
        print(f"  - Distance: {match['distance']}m")
        print(f"  - lanesHeading: {match['heading']}°")
        print(f"  - drivingDirection: {'regularto' if match['is_forward'] else 'to'}")
    else:
        print("notMatchingtolanes")


def test_path_finding(api: MapAPI):
    """TestingPathFind"""
    print("\n" + "=" * 60)
    print("PathFindTesting")
    print("=" * 60)

    # TwohaveConnectlanes
    cl_ids = api.get_all_centerline_ids()
    if len(cl_ids) < 2:
        print("Centerdatanot")
        return

    # haveafterCenter
    start_cl = None
    end_cl = None

    for cl_id in cl_ids:
        info = api.get_centerline_info(cl_id)
        if info and info['successor_ids']:
            start_cl = cl_id
            end_cl = info['successor_ids'][0]
            break

    if not start_cl:
        print("havetocanConnectPath")
        return

    print(f"\nFromCenter {start_cl} to {end_cl}")

    # neededlanesID
    start_info = api.get_centerline_info(start_cl)
    start_lane = start_info['right_boundary_id'] or start_info['left_boundary_id'] or start_cl

    end_info = api.get_centerline_info(end_cl)
    end_lane = end_info['right_boundary_id'] or end_info['left_boundary_id'] or end_cl

    path = api.find_path_between_lanes(start_lane, end_lane)
    if path:
        print(f"toPath: {' -> '.join(path)}")
    else:
        print("nottoPath")


def main():
    print("=" * 60)
    print("MapAgent Phase 2 Testing")
    print("=" * 60)

    # LoadMap
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"Error: MapFilenotunder {map_path}")
        return

    print(f"\nLoadMap: {map_path}")
    api = MapAPI(map_file=str(map_path))

    # RunTesting
    test_basic_queries(api)
    test_topology_queries(api)
    test_spatial_queries(api)
    test_vehicle_matching(api)
    test_path_finding(api)

    print("\n" + "=" * 60)
    print("haveTestingto")
    print("=" * 60)


if __name__ == "__main__":
    main()