"""测试数据模型"""

import pytest
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import VectorMap, MapLoader, LaneLine, Centerline
from apis import MapAPI


class TestMapLoader:
    """测试地图加载器"""

    def test_load_vector_map(self):
        """测试加载矢量地图"""
        map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
        if not map_path.exists():
            pytest.skip("vector_map.json not found")

        map_data = MapLoader.load_from_json(str(map_path))

        assert map_data is not None
        assert map_data.version == "1.0"
        assert len(map_data.lane_lines) > 0
        assert len(map_data.centerlines) > 0

    def test_lane_line_model(self):
        """测试车道线模型"""
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
    """测试地图API"""

    @pytest.fixture
    def map_api(self):
        """创建 MapAPI 实例"""
        map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
        if not map_path.exists():
            pytest.skip("vector_map.json not found")
        return MapAPI(map_file=str(map_path))

    def test_get_map_summary(self, map_api):
        """测试获取地图概要"""
        summary = map_api.get_map_summary()

        assert summary is not None
        assert "total_lanes" in summary
        assert summary["total_lanes"] > 0

    def test_get_lane_info(self, map_api):
        """测试获取车道信息"""
        # 获取第一个车道ID
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
        """测试查找最近车道"""
        # 使用地图中某个点的坐标
        lane_ids = list(map_api.map.lane_lines.keys())
        if not lane_ids:
            pytest.skip("No lanes in map")

        lane = map_api.map.lane_lines[lane_ids[0]]
        if lane.coordinates:
            position = tuple(lane.coordinates[0])
            result = map_api.find_nearest_lane(position)

            assert result is not None
            assert "lane_id" in result
            assert result["distance"] < 1.0  # 应该非常近

    def test_get_area_statistics(self, map_api):
        """测试区域统计"""
        # 获取一个车道中心点
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