"""
主 Agent (MasterAgent)

真正基于 LLM 的实现：
- 使用 LLM 进行意图识别和实体提取
- 使用 Function Calling 调用 MapAPI
- 使用 LLM 生成自然语言回复
"""

from typing import Dict, List, Any, Optional
import json
import sys
import os
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.llm_client import LLMClient, LLMConfig
from apis.map_api import MapAPI


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的地图问答助手，能够回答关于道路、车道、路口、车辆行为和路径规划的问题。

你可以使用以下工具来获取地图信息：

1. get_lane_info - 获取车道详细信息
2. get_centerline_info - 获取中心线信息（包含前后拓扑关系）
3. get_intersection_info - 获取路口信息
4. find_nearest_lane - 查找最近的车道
5. find_nearest_centerline - 查找最近的中心线
6. find_lanes_in_area - 查找区域内的车道
7. find_intersections_in_area - 查找区域内的路口
8. get_area_statistics - 获取区域统计信息
9. get_traffic_signs_in_area - 获取区域内交通标志
10. match_vehicle_to_lane - 将车辆匹配到车道
11. find_path - 查找路径
12. get_map_summary - 获取地图概要信息
13. load_detection_results - 加载检测结果
14. reconstruct_traffic_flow - 重建交通流轨迹
15. get_trajectory_by_id - 获取指定车辆轨迹
16. analyze_vehicle_behavior - 分析车辆行为
17. save_reconstruction_result - 保存重建结果
18. get_traffic_flow_summary - 获取交通流摘要

回答要求：
1. 先理解用户问题，必要时调用工具获取数据
2. 基于获取的数据给出准确回答
3. 回答要简洁专业，包含具体数据
4. 如果缺少位置信息，请询问用户
5. 如果用户请求交通流重建，先加载检测结果，然后重建并保存"""


class MasterAgent:
    """
    主 Agent - 基于 LLM 的实现

    可以单独使用工具执行，也可以结合 LLM 进行智能对话
    """

    def __init__(self, map_api: MapAPI, llm_client: Optional[LLMClient] = None):
        """
        初始化

        Args:
            map_api: 地图 API 实例
            llm_client: LLM 客户端（可选，不提供时仅能执行工具）
        """
        self.map_api = map_api
        self.llm_client = llm_client

        # 注册工具
        self._tools = self._build_tools()

        # 对话历史
        self.messages: List[Dict] = []

        # 初始化交通流 Agent
        from agents.base import AgentContext
        context = AgentContext(map_api=map_api, llm_client=llm_client)
        self._traffic_flow_agent = None  # 懒加载

    def _build_tools(self) -> List[Dict]:
        """构建 Function Calling 工具定义"""
        return [
            {
                "name": "get_lane_info",
                "description": "获取指定车道的详细信息，包括类型、颜色、长度、坐标等",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lane_id": {"type": "string", "description": "车道ID"}
                    },
                    "required": ["lane_id"]
                }
            },
            {
                "name": "get_centerline_info",
                "description": "获取中心线信息，包括左右边界、前后连接关系",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "centerline_id": {"type": "string", "description": "中心线ID"}
                    },
                    "required": ["centerline_id"]
                }
            },
            {
                "name": "get_intersection_info",
                "description": "获取路口的详细信息",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "intersection_id": {"type": "string", "description": "路口ID"}
                    },
                    "required": ["intersection_id"]
                }
            },
            {
                "name": "find_nearest_lane",
                "description": "查找距离指定位置最近的车道",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X坐标"},
                        "y": {"type": "number", "description": "Y坐标"},
                        "z": {"type": "number", "description": "Z坐标", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_nearest_centerline",
                "description": "查找距离指定位置最近的中心线",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X坐标"},
                        "y": {"type": "number", "description": "Y坐标"},
                        "z": {"type": "number", "description": "Z坐标", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_lanes_in_area",
                "description": "查找指定位置和半径内的所有车道",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "中心X坐标"},
                        "y": {"type": "number", "description": "中心Y坐标"},
                        "radius": {"type": "number", "description": "半径(米)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_intersections_in_area",
                "description": "查找指定位置和半径内的所有路口",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "中心X坐标"},
                        "y": {"type": "number", "description": "中心Y坐标"},
                        "radius": {"type": "number", "description": "半径(米)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "get_area_statistics",
                "description": "获取指定区域的统计信息，包括车道数量、类型分布、路口数量等",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "中心X坐标"},
                        "y": {"type": "number", "description": "中心Y坐标"},
                        "radius": {"type": "number", "description": "半径(米)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "get_traffic_signs_in_area",
                "description": "获取指定区域内的交通标志",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "中心X坐标"},
                        "y": {"type": "number", "description": "中心Y坐标"},
                        "radius": {"type": "number", "description": "半径(米)", "default": 50}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "match_vehicle_to_lane",
                "description": "将车辆匹配到最近的车道，返回车道信息和车辆行驶方向",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "车辆X坐标"},
                        "y": {"type": "number", "description": "车辆Y坐标"},
                        "heading": {"type": "number", "description": "车辆航向角(度)", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_path",
                "description": "查找从起点到终点的路径",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "number", "description": "起点X坐标"},
                        "start_y": {"type": "number", "description": "起点Y坐标"},
                        "end_x": {"type": "number", "description": "终点X坐标"},
                        "end_y": {"type": "number", "description": "终点Y坐标"}
                    },
                    "required": ["start_x", "start_y", "end_x", "end_y"]
                }
            },
            {
                "name": "get_map_summary",
                "description": "获取地图概要信息，包括总车道数、路口数等",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "load_detection_results",
                "description": "加载检测结果数据，用于交通流重建",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "检测结果目录路径"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "reconstruct_traffic_flow",
                "description": "重建交通流轨迹，基于加载的检测结果",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer", "description": "起始帧ID"},
                        "end_frame": {"type": "integer", "description": "结束帧ID"},
                        "use_llm": {"type": "boolean", "description": "是否使用LLM补充推理"}
                    }
                }
            },
            {
                "name": "get_trajectory_by_id",
                "description": "获取指定车辆的轨迹信息",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "vehicle_id": {"type": "integer", "description": "车辆ID"}
                    },
                    "required": ["vehicle_id"]
                }
            },
            {
                "name": "save_reconstruction_result",
                "description": "保存交通流重建结果",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "输出文件路径"}
                    }
                }
            },
            {
                "name": "get_traffic_flow_summary",
                "description": "获取交通流重建摘要信息",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def _execute_tool(self, name: str, args: Dict) -> Any:
        """执行工具调用"""
        # 懒加载 TrafficFlowAgent
        if name in ["load_detection_results", "reconstruct_traffic_flow",
                    "get_trajectory_by_id", "save_reconstruction_result",
                    "get_traffic_flow_summary", "analyze_vehicle_behavior"]:
            if self._traffic_flow_agent is None:
                from agents.base import AgentContext
                from agents.traffic_flow import TrafficFlowAgent
                context = AgentContext(map_api=self.map_api, llm_client=self.llm_client)
                self._traffic_flow_agent = TrafficFlowAgent(context)

        try:
            if name == "get_lane_info":
                return self.map_api.get_lane_info(args["lane_id"])
            elif name == "get_centerline_info":
                return self.map_api.get_centerline_info(args["centerline_id"])
            elif name == "get_intersection_info":
                return self.map_api.get_intersection_info(args["intersection_id"])
            elif name == "find_nearest_lane":
                pos = (args["x"], args["y"], args.get("z", 0))
                return self.map_api.find_nearest_lane(pos)
            elif name == "find_nearest_centerline":
                pos = (args["x"], args["y"], args.get("z", 0))
                return self.map_api.find_nearest_centerline(pos)
            elif name == "find_lanes_in_area":
                pos = (args["x"], args["y"], 0)
                return self.map_api.find_lanes_in_area(pos, args.get("radius", 100))
            elif name == "find_intersections_in_area":
                pos = (args["x"], args["y"], 0)
                return self.map_api.find_intersections_in_area(pos, args.get("radius", 100))
            elif name == "get_area_statistics":
                pos = (args["x"], args["y"], 0)
                return self.map_api.get_area_statistics(pos, args.get("radius", 100))
            elif name == "get_traffic_signs_in_area":
                pos = (args["x"], args["y"], 0)
                return self.map_api.get_traffic_signs_in_area(pos, args.get("radius", 50))
            elif name == "match_vehicle_to_lane":
                pos = (args["x"], args["y"], 0)
                return self.map_api.match_vehicle_to_lane(pos, args.get("heading", 0))
            elif name == "find_path":
                origin = (args["start_x"], args["start_y"], 0)
                dest = (args["end_x"], args["end_y"], 0)
                return self.map_api.find_path_between_lanes(origin, dest)
            elif name == "get_map_summary":
                return self.map_api.get_map_summary()
            elif name == "load_detection_results":
                return self._traffic_flow_agent._load_detection_results(args["path"])
            elif name == "reconstruct_traffic_flow":
                return self._traffic_flow_agent._reconstruct_traffic_flow(
                    args.get("start_frame"),
                    args.get("end_frame"),
                    args.get("use_llm", True)
                )
            elif name == "get_trajectory_by_id":
                return self._traffic_flow_agent._get_trajectory_by_id(args["vehicle_id"])
            elif name == "save_reconstruction_result":
                return self._traffic_flow_agent._save_reconstruction_result(
                    args.get("output_path", "reconstruction_result.json")
                )
            elif name == "get_traffic_flow_summary":
                return self._traffic_flow_agent._get_traffic_flow_summary()
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}

    def chat(self, query: str, **kwargs) -> str:
        """
        与 Agent 对话

        Args:
            query: 用户问题
            **kwargs: 额外上下文（如位置坐标）

        Returns:
            回复
        """
        if not self.llm_client:
            return "错误：LLM 客户端未配置。请设置 API Key 后使用。"

        # 构建用户消息，包含上下文
        user_content = query
        if kwargs:
            context_parts = []
            if "location" in kwargs:
                loc = kwargs["location"]
                context_parts.append(f"当前位置: ({loc[0]:.1f}, {loc[1]:.1f})")
            if "radius" in kwargs:
                context_parts.append(f"查询半径: {kwargs['radius']}米")
            if "heading" in kwargs:
                context_parts.append(f"航向角: {kwargs['heading']}度")
            if "speed" in kwargs:
                context_parts.append(f"速度: {kwargs['speed']}m/s")
            if context_parts:
                user_content = query + "\n\n上下文信息:\n" + "\n".join(context_parts)

        # 添加到历史
        self.messages.append({"role": "user", "content": user_content})

        # 调用 LLM
        response = self.llm_client.chat(
            messages=self.messages,
            tools=self._tools,
            system=SYSTEM_PROMPT,
            use_tools=True,
            tool_handler=self._execute_tool,
            max_turns=5
        )

        # 保存回复
        self.messages.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """清空对话历史"""
        self.messages = []

    def route(self, query: str, **kwargs) -> str:
        """兼容接口"""
        return self.chat(query, **kwargs)

    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return [t["name"] for t in self._tools]


def create_master_agent(
    map_file: str = None,
    llm_provider: str = None,
    llm_model: str = None,
    api_key: str = None
) -> MasterAgent:
    """
    创建主 Agent

    Args:
        map_file: 地图文件路径（默认从 settings 读取）
        llm_provider: LLM 提供商（默认从 settings 读取）
        llm_model: 模型名称（默认从 settings 读取）
        api_key: API Key（默认从 settings 读取）

    Returns:
        MasterAgent 实例
    """
    # 尝试从 settings 加载配置
    try:
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from config import settings
        use_settings = True
    except ImportError:
        use_settings = False
        settings = None

    # 加载地图
    if map_file:
        map_api = MapAPI(map_file=map_file)
    elif use_settings and settings:
        map_api = MapAPI(map_file=str(settings.map_path))
    else:
        map_api = MapAPI()

    # 配置 LLM
    provider_map = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "deepseek": "deepseek",
        "openai": "openai",
        "local": "local",
    }

    # 获取 LLM 配置（优先参数，其次 settings，最后环境变量）
    if llm_provider:
        provider = provider_map.get(llm_provider, llm_provider)
    elif use_settings and settings:
        provider = provider_map.get(settings.llm_provider, settings.llm_provider)
    else:
        provider = provider_map.get(os.getenv("LLM_PROVIDER", "anthropic"), "anthropic")

    model = llm_model
    if not model:
        if use_settings and settings:
            model = settings.llm_model
        else:
            model = os.getenv("LLM_MODEL", "")

    # 获取 API Key
    resolved_api_key = api_key
    if not resolved_api_key:
        if use_settings and settings and settings.llm_api_key:
            resolved_api_key = settings.llm_api_key
        elif provider == "anthropic":
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "deepseek":
            resolved_api_key = os.getenv("DEEPSEEK_API_KEY")
        elif provider == "openai":
            resolved_api_key = os.getenv("OPENAI_API_KEY")

    # 获取 base_url
    base_url = None
    if use_settings and settings and settings.llm_base_url:
        base_url = settings.llm_base_url
    else:
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url and provider == "deepseek":
            base_url = "https://api.deepseek.com"

    # 创建 LLM 客户端（如果没有 API Key，则创建不带 LLM 的 Agent）
    llm_client = None
    if resolved_api_key or provider == "local":
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=resolved_api_key,
            base_url=base_url,
        )
        try:
            llm_client = LLMClient(config)
        except ImportError as e:
            print(f"警告: 无法创建 LLM 客户端: {e}")
            print("将创建不带 LLM 的 Agent，仅支持工具执行")

    return MasterAgent(map_api=map_api, llm_client=llm_client)