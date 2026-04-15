"""
Master Agent (MasterAgent)

LLM-based implementation:
- Use LLM for intent recognition and entity extraction
- Use Function Calling to call MapAPI
- Use LLM to generate natural language responses
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
from core.llm_client import LLMProvider
from apis.map_api import MapAPI


# System prompt
SYSTEM_PROMPT = """You are a professional map Q&A assistant, capable of answering questions about roads, lanes, intersections, vehicle behavior, and path planning.

You can use the following tools to retrieve map information:

1. get_lane_info - Get detailed lane information
2. get_centerline_info - Get centerline information (including forward/backward topology)
3. get_intersection_info - Get intersection information
4. find_nearest_lane - Find nearest lane
5. find_nearest_centerline - Find nearest centerline
6. find_lanes_in_area - Find lanes in area
7. find_intersections_in_area - Find intersections in area
8. get_area_statistics - Get area statistics
9. get_traffic_signs_in_area - Get traffic signs in area
10. match_vehicle_to_lane - Match vehicle to lane
11. find_path - Find path
12. get_map_summary - Get map summary
13. load_detection_results - Load detection results
14. reconstruct_traffic_flow - Reconstruct traffic flow trajectories
15. get_trajectory_by_id - Get trajectory by vehicle ID
16. analyze_vehicle_behavior - Analyze vehicle behavior
17. save_reconstruction_result - Save reconstruction results
18. get_traffic_flow_summary - Get traffic flow summary

Response requirements:
1. Understand user query, call tools when necessary
2. Provide accurate answers based on retrieved data
3. Keep responses concise and professional, include specific data
4. If location information is missing, ask the user
5. If user requests traffic flow reconstruction, load detection results first, then reconstruct and save"""


class MasterAgent:
    """
    Master Agent - LLM-based implementation

    Can be used for tool execution alone, or combined with LLM for intelligent conversation
    """

    def __init__(self, map_api: MapAPI, llm_client: Optional[LLMClient] = None):
        """
        Initialize

        Args:
            map_api: Map API instance
            llm_client: LLM client (optional, only tool execution without LLM)
        """
        self.map_api = map_api
        self.llm_client = llm_client

        # Register tools
        self._tools = self._build_tools()

        # Conversation history
        self.messages: List[Dict] = []

        # Initialize TrafficFlowAgent
        from agents.base import AgentContext
        context = AgentContext(map_api=map_api, llm_client=llm_client)
        self._traffic_flow_agent = None  # Lazy loading

    def _build_tools(self) -> List[Dict]:
        """Build Function Calling tool definitions"""
        return [
            {
                "name": "get_lane_info",
                "description": "Get detailed information about a lane, including type, color, length, coordinates, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lane_id": {"type": "string", "description": "Lane ID"}
                    },
                    "required": ["lane_id"]
                }
            },
            {
                "name": "get_centerline_info",
                "description": "Get centerline information, including left/right boundaries and forward/backward connections",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "centerline_id": {"type": "string", "description": "Centerline ID"}
                    },
                    "required": ["centerline_id"]
                }
            },
            {
                "name": "get_intersection_info",
                "description": "Get detailed intersection information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "intersection_id": {"type": "string", "description": "Intersection ID"}
                    },
                    "required": ["intersection_id"]
                }
            },
            {
                "name": "find_nearest_lane",
                "description": "Find the nearest lane to a specified position",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "z": {"type": "number", "description": "Z coordinate", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_nearest_centerline",
                "description": "Find the nearest centerline to a specified position",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "z": {"type": "number", "description": "Z coordinate", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_lanes_in_area",
                "description": "Find all lanes within a specified position and radius",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Center X coordinate"},
                        "y": {"type": "number", "description": "Center Y coordinate"},
                        "radius": {"type": "number", "description": "Radius (meters)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_intersections_in_area",
                "description": "Find all intersections within a specified position and radius",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Center X coordinate"},
                        "y": {"type": "number", "description": "Center Y coordinate"},
                        "radius": {"type": "number", "description": "Radius (meters)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "get_area_statistics",
                "description": "Get statistics for a specified area, including lane count, type distribution, intersection count, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Center X coordinate"},
                        "y": {"type": "number", "description": "Center Y coordinate"},
                        "radius": {"type": "number", "description": "Radius (meters)", "default": 100}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "get_traffic_signs_in_area",
                "description": "Get traffic signs within a specified area",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Center X coordinate"},
                        "y": {"type": "number", "description": "Center Y coordinate"},
                        "radius": {"type": "number", "description": "Radius (meters)", "default": 50}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "match_vehicle_to_lane",
                "description": "Match vehicle to nearest lane, return lane information and vehicle heading",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Vehicle X coordinate"},
                        "y": {"type": "number", "description": "Vehicle Y coordinate"},
                        "heading": {"type": "number", "description": "Vehicle heading angle (degrees)", "default": 0}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "find_path",
                "description": "Find path from start to end",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "number", "description": "Start X coordinate"},
                        "start_y": {"type": "number", "description": "Start Y coordinate"},
                        "end_x": {"type": "number", "description": "End X coordinate"},
                        "end_y": {"type": "number", "description": "End Y coordinate"}
                    },
                    "required": ["start_x", "start_y", "end_x", "end_y"]
                }
            },
            {
                "name": "get_map_summary",
                "description": "Get map summary, including total lane count, intersection count, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "load_detection_results",
                "description": "Load detection results for traffic flow reconstruction",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Detection results directory path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "reconstruct_traffic_flow",
                "description": "Reconstruct traffic flow trajectories based on detection results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_frame": {"type": "integer", "description": "Start frame ID"},
                        "end_frame": {"type": "integer", "description": "End frame ID"},
                        "max_distance": {"type": "number", "description": "Maximum matching distance (meters)", "default": 5.0},
                        "max_velocity": {"type": "number", "description": "Maximum velocity (m/s)", "default": 30.0}
                    }
                }
            },
            {
                "name": "get_trajectory_by_id",
                "description": "Get trajectory for a specific vehicle ID",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "vehicle_id": {"type": "integer", "description": "Vehicle ID"}
                    },
                    "required": ["vehicle_id"]
                }
            },
            {
                "name": "save_reconstruction_result",
                "description": "Save traffic flow reconstruction results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "Output file path"}
                    }
                }
            },
            {
                "name": "get_traffic_flow_summary",
                "description": "Get traffic flow reconstruction summary",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def _execute_tool(self, name: str, args: Dict) -> Any:
        """Execute tool call"""
        # Log tool execution
        print(f"[INFO] 主 Agent: 调用工具 - {name}")

        # Lazy loading TrafficFlowAgent (supports LLM enhanced mode)
        if name in ["load_detection_results", "reconstruct_traffic_flow",
                    "get_trajectory_by_id", "save_reconstruction_result",
                    "get_traffic_flow_summary", "analyze_vehicle_behavior"]:
            if self._traffic_flow_agent is None:
                from agents.base import AgentContext
                from agents.traffic_flow import TrafficFlowAgent
                context = AgentContext(map_api=self.map_api, llm_client=self.llm_client)
                # Decide whether to enable LLM optimization based on LLM client configuration
                use_llm = self.llm_client is not None
                self._traffic_flow_agent = TrafficFlowAgent(context, use_llm=use_llm)
                print(f"[INFO] 主 Agent: TrafficFlowAgent 已初始化")

        try:
            if name == "get_lane_info":
                print(f"[INFO] 主 Agent: 正在处理地图数据 - 获取车道 {args['lane_id']} 信息...")
                return self.map_api.get_lane_info(args["lane_id"])
            elif name == "get_centerline_info":
                print(f"[INFO] 主 Agent: 正在处理地图数据 - 获取中心线 {args['centerline_id']} 信息...")
                return self.map_api.get_centerline_info(args["centerline_id"])
            elif name == "get_intersection_info":
                print(f"[INFO] 主 Agent: 正在处理地图数据 - 获取路口 {args['intersection_id']} 信息...")
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
                print(f"[INFO] 主 Agent: 正在加载检测结果 - 路径：{args['path']}...")
                return self._traffic_flow_agent._load_detection_results(args["path"])
            elif name == "reconstruct_traffic_flow":
                print(f"[INFO] 主 Agent: 正在重建交通流 - 帧范围：{args.get('start_frame')} 到 {args.get('end_frame')}...")
                return self._traffic_flow_agent._reconstruct_traffic_flow(
                    args.get("start_frame"),
                    args.get("end_frame"),
                    max_distance=args.get("max_distance", 5.0),
                    max_velocity=args.get("max_velocity", 30.0),
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
            print(f"[ERROR] 主 Agent: 工具执行失败 - {name}, 错误：{str(e)}")
            return {"error": str(e)}

    def chat(self, query: str, **kwargs) -> str:
        """
        Chat with Agent

        Args:
            query: User query
            **kwargs: Additional context (e.g., location coordinates)

        Returns:
            Response
        """
        if not self.llm_client:
            return "Error: LLM client not configured. Please set API Key to use."

        # Log start of execution
        print(f"\n[INFO] 主 Agent: 开始执行任务...")
        print(f"[INFO] 主 Agent: 用户查询 - {query[:50]}...")

        # Build user message with context
        user_content = query
        if kwargs:
            context_parts = []
            if "location" in kwargs:
                loc = kwargs["location"]
                context_parts.append(f"Current location: ({loc[0]:.1f}, {loc[1]:.1f})")
            if "radius" in kwargs:
                context_parts.append(f"Query radius: {kwargs['radius']} meters")
            if "heading" in kwargs:
                context_parts.append(f"Heading angle: {kwargs['heading']} degrees")
            if "speed" in kwargs:
                context_parts.append(f"Speed: {kwargs['speed']} m/s")
            if context_parts:
                user_content = query + "\n\nContext:\n" + "\n".join(context_parts)

        # Add to history
        self.messages.append({"role": "user", "content": user_content})

        # Call LLM
        print(f"[INFO] 主 Agent: 调用 LLM 模型，正在处理请求...")
        response = self.llm_client.chat(
            messages=self.messages,
            tools=self._tools,
            system=SYSTEM_PROMPT,
            use_tools=True,
            tool_handler=self._execute_tool,
            max_turns=5
        )

        # Save response
        self.messages.append({"role": "assistant", "content": response})

        print(f"[INFO] 主 Agent: 任务执行完成，结果已生成。")

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.messages = []

    def route(self, query: str, **kwargs) -> str:
        """Compatibility interface"""
        return self.chat(query, **kwargs)

    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [t["name"] for t in self._tools]


def create_master_agent(
    map_file: str = None,
    llm_provider: str = None,
    llm_model: str = None,
    api_key: str = None
) -> MasterAgent:
    """
    Create Master Agent

    Args:
        map_file: Map file path (default from settings)
        llm_provider: LLM provider (default from settings)
        llm_model: Model name (default from settings)
        api_key: API Key (default from settings)

    Returns:
        MasterAgent instance
    """
    # Try to load config from settings
    try:
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from config import settings
        use_settings = True
    except ImportError:
        use_settings = False
        settings = None

    # Load map
    if map_file:
        map_api = MapAPI(map_file=map_file)
    elif use_settings and settings:
        map_api = MapAPI(map_file=str(settings.map_path))
    else:
        map_api = MapAPI()

    # Configure LLM - 使用统一配置
    from config.providers import get_provider, get_default_model, get_base_url, is_local_model

    # Get LLM config (priority: parameters, then settings, then env vars)
    if llm_provider:
        provider = get_provider(llm_provider)
    elif use_settings and settings:
        provider = get_provider(settings.llm_provider)
    else:
        provider = get_provider(os.getenv("LLM_PROVIDER", "anthropic"))

    model = llm_model
    if not model:
        # Use env var first (set when UI switches model)
        model = os.getenv("LLM_MODEL", "")
        if not model and use_settings and settings:
            model = settings.llm_model

    # Get API Key (use dummy key for local models)
    resolved_api_key = api_key
    if not resolved_api_key:
        if is_local_model(provider):
            resolved_api_key = "dummy"
        elif use_settings and settings and settings.llm_api_key:
            resolved_api_key = settings.llm_api_key
        elif provider == LLMProvider.ANTHROPIC:
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == LLMProvider.DEEPSEEK:
            resolved_api_key = os.getenv("DEEPSEEK_API_KEY")
        elif provider == LLMProvider.OPENAI:
            resolved_api_key = os.getenv("OPENAI_API_KEY")

    # Get base_url
    base_url = os.getenv("LLM_BASE_URL") or get_base_url(provider)
    if use_settings and settings and settings.llm_base_url and not base_url:
        base_url = settings.llm_base_url

    # Create LLM client (if no API Key, create Agent without LLM)
    llm_client = None
    if resolved_api_key or provider in ["local", "qwen_local", "gemma4_local"]:
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=resolved_api_key,
            base_url=base_url,
        )
        try:
            llm_client = LLMClient(config)
        except ImportError as e:
            print(f"Warning: Cannot create LLM client: {e}")
            print("Creating Agent without LLM, only supports tool execution")

    return MasterAgent(map_api=map_api, llm_client=llm_client)
