"""
Function Calling 工具注册中心

定义和管理 Agent 可用的工具
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None

    def to_anthropic_format(self) -> Dict:
        """转换为 Anthropic 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
            }
        }


class ToolRegistry:
    """
    工具注册中心

    管理所有可用的 Function Calling 工具
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str,
                 parameters: Dict[str, Any], handler: Callable = None) -> None:
        """
        注册工具

        Args:
            name: 工具名称
            description: 工具描述
            parameters: 参数定义
            handler: 处理函数
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )
        logger.debug(f"Registered tool: {name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        """获取工具处理函数"""
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def get_all_definitions(self) -> List[Dict]:
        """获取所有工具定义（Anthropic 格式）"""
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        if not tool.handler:
            raise ValueError(f"Tool has no handler: {name}")

        logger.info(f"Executing tool: {name}({kwargs})")
        return tool.handler(**kwargs)


# ==================== 预定义的地图查询工具 ====================

def create_map_tools(map_api) -> ToolRegistry:
    """
    创建地图相关的工具

    Args:
        map_api: MapAPI 实例

    Returns:
        ToolRegistry
    """
    registry = ToolRegistry()

    # 获取车道信息
    registry.register(
        name="get_lane_info",
        description="获取指定车道的详细信息",
        parameters={
            "lane_id": {
                "type": "string",
                "description": "车道ID"
            }
        },
        handler=lambda lane_id: map_api.get_lane_info(lane_id)
    )

    # 获取路口信息
    registry.register(
        name="get_intersection_info",
        description="获取指定路口的详细信息",
        parameters={
            "intersection_id": {
                "type": "string",
                "description": "路口ID"
            }
        },
        handler=lambda intersection_id: map_api.get_intersection_info(intersection_id)
    )

    # 查找最近的车道
    registry.register(
        name="find_nearest_lane",
        description="查找距离指定位置最近的车道",
        parameters={
            "x": {"type": "number", "description": "X坐标"},
            "y": {"type": "number", "description": "Y坐标"},
            "z": {"type": "number", "description": "Z坐标", "default": 0}
        },
        handler=lambda x, y, z=0: map_api.find_nearest_lane((x, y, z))
    )

    # 获取车道拓扑关系
    registry.register(
        name="get_lane_topology",
        description="获取车道的前驱和后继车道",
        parameters={
            "lane_id": {
                "type": "string",
                "description": "车道ID"
            }
        },
        handler=lambda lane_id: map_api.get_lane_topology(lane_id)
    )

    # 获取区域统计
    registry.register(
        name="get_area_statistics",
        description="获取指定区域的车道统计信息",
        parameters={
            "center_x": {"type": "number", "description": "中心X坐标"},
            "center_y": {"type": "number", "description": "中心Y坐标"},
            "radius": {"type": "number", "description": "半径(米)", "default": 100}
        },
        handler=lambda center_x, center_y, radius=100: map_api.get_area_statistics(
            (center_x, center_y, 0), radius
        )
    )

    return registry