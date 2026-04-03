"""
Agent 基类

定义 Agent 的基本接口和工具注册机制
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import sys
from pathlib import Path

# 确保模块导入
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.llm_client import LLMClient, LLMConfig
from core.tools import ToolRegistry
from apis.map_api import MapAPI


@dataclass
class AgentContext:
    """Agent 上下文"""
    map_api: MapAPI
    llm_client: Optional[LLMClient] = None
    conversation_history: List[Dict] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Agent 基类

    所有子 Agent 需要继承此类并实现:
    - get_tools(): 返回可用的工具列表
    - process(query): 处理查询并返回结果
    """

    def __init__(self, context: AgentContext):
        self.context = context
        self.map_api = context.map_api
        self.llm_client = context.llm_client
        self.tools = ToolRegistry()

        # 注册工具
        self._register_tools()

    def _register_tools(self):
        """注册工具，子类可重写"""
        for tool in self.get_tools():
            self.tools.register(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                handler=tool.get("handler")
            )

    @abstractmethod
    def get_tools(self) -> List[Dict]:
        """返回工具定义列表"""
        pass

    @abstractmethod
    def process(self, query: str, **kwargs) -> Dict:
        """
        处理查询

        Args:
            query: 用户查询
            **kwargs: 额外参数

        Returns:
            处理结果字典
        """
        pass

    def get_tool_definitions(self) -> List[Dict]:
        """获取工具定义（用于 LLM Function Calling）"""
        return self.tools.get_all_definitions()

    def execute_tool(self, name: str, **kwargs) -> Any:
        """执行工具"""
        return self.tools.execute(name, **kwargs)

    def get_system_prompt(self) -> str:
        """获取系统提示，子类可重写"""
        return "你是一个地图助手，帮助用户回答地图相关问题。"