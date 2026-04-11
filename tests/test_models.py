"""Testingdatamodels"""

import pytest
from pathlib import Path
import sys

# Add src toPath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import Vector Map, Map Loader, LaneLine, Centerline
from apis import MapAPI


class TestMap Loader:
    """TestingMapLoader"""

    def test_load_vector_map(self):
        """TestingLoadMap"""
        map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
        if not map_path.exists():
            pytest.skip("vector_map.json not found")

        map_data = Map Loader.load_from_json(str(map_path))

        assert map_data is not None
        assert map_data.version == "1.0"
        assert len(map_data.lane_lines) > 0
        assert len(map_data.centerlines) > 0

    def test_lane_line_model(self):
        """Testinglanesmodels"""
        lane = LaneLine(
            id="test_1",
            type="solid",
            color="white",
            coordinates=[[0, 0, 0], [1, 1, 0], [2, 2, 0]],
            length=2.83
        )

        assert lane.id == "test_1"
        assert lane.type == "solid"
        assert lane.get_start_point() == (0, 0, 0)
        assert lane.get_end_point() == (2, 2, 0)


class TestMapAPI:
    """TestingMapAPI"""

    @pytest.fixture
    def map_api(self):
        """Create MapAPI solid"""
        map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
        if not map_path.exists():
            pytest.skip("vector_map.json not found")
        return MapAPI(map_file=str(map_path))

    def test_get_map_summary(self, map_api):
        """TestingGetMapwill"""
        summary = map_api.get_map_summary()

        assert summary is not None
        assert "total_lanes" in summary
        assert summary["total_lanes"] > 0

    def test_get_lane_info(self, map_api):
        """TestingGetlanesinfo"""
        # GetalanesID
        lane_ids = list(map_api.map.lane_lines.keys())
        if not lane_ids:
            pytest.skip("No lanes in map")

        lane_id = lane_ids[0]
        info = map_api.get_lane_info(lane_id)

        assert info is not None
        assert info["id"] == lane_id
        assert "type" in info
        assert "coordinates" in info

    def test_find_nearest_lane(self, map_api):
        """TestingFindlanes"""
        # UseMapcenterPointCoordinate
        lane_ids = list(map_api.map.lane_lines.keys())
        if not lane_ids:
            pytest.skip("No lanes in map")

        lane = map_api.map.lane_lines[lane_ids[0]]
        if lane.coordinates:
            position = tuple(lane.coordinates[0])
            result = map_api.find_nearest_lane(position)

            assert result is not None
            assert "lane_id" in result
            assert result["distance"] < 1.0  # non

    def test_get_area_statistics(self, map_api):
        """Testingareadomain"""
        # GetalanesCenterPoint
        lane_ids = list(map_api.map.lane_lines.keys())
        if not lane_ids:
            pytest.skip("No lanes in map")

        lane = map_api.map.lane_lines[lane_ids[0]]
        if lane.coordinates:
            center = tuple(lane.coordinates[len(lane.coordinates) // 2])
            stats = map_api.get_area_statistics(center, radius=100)

            assert stats is not None
            assert stats["lane_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])