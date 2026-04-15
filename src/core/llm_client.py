"""
LLM 客户端封装

支持多种 LLM 提供商:
- Anthropic Claude
- Deepseek
- 本地模型 (通过 OpenAI 兼容 API)
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# 使用统一的提供商配置
from config.providers import LLMProvider, PROVIDER_MAP, DEFAULT_MODELS, get_base_url, is_local_model

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60

    # 本地模型配置
    local_model_path: Optional[str] = None
    local_model_type: Optional[str] = None  # qwen 或 gemma4

    # 项目根目录（用于查找本地模型）
    project_root: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量加载配置"""
        provider_str = os.getenv("LLM_PROVIDER", "anthropic").lower()

        # 使用统一配置获取 provider
        from config.providers import get_provider, get_default_model, get_base_url, is_local_model

        provider = get_provider(provider_str)

        # 项目根目录
        project_root = os.getenv("PROJECT_ROOT", "/data/DC/MapAgent")

        # 根据提供商选择 API key
        api_key = None
        if provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == LLMProvider.DEEPSEEK:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider in [LLMProvider.LOCAL, LLMProvider.QWEN_LOCAL, LLMProvider.GEMMA4_LOCAL]:
            api_key = "dummy"  # 本地模型不需要真实 API key

        # base_url
        base_url = os.getenv("LLM_BASE_URL") or get_base_url(provider_str)

        # 本地模型类型
        local_model_type = None
        if provider == LLMProvider.QWEN_LOCAL:
            local_model_type = "qwen"
        elif provider == LLMProvider.GEMMA4_LOCAL:
            local_model_type = "gemma4"

        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", get_default_model(provider_str)),
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            local_model_path=os.getenv("LOCAL_MODEL_PATH"),
            local_model_type=local_model_type,
            project_root=project_root,
        )

    @classmethod
    def for_deepseek(cls, model: str = "deepseek-reasoner", api_key: Optional[str] = None) -> "LLMConfig":
        """创建 Deepseek 配置"""
        return cls(
            provider=LLMProvider.DEEPSEEK,
            model=model,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    @classmethod
    def for_local(cls, model: str = "Qwen3___5-35B-A3B",
                  base_url: str = "http://localhost:8000/v1") -> "LLMConfig":
        """创建本地模型配置（通用）"""
        return cls(
            provider=LLMProvider.LOCAL,
            model=model,
            base_url=base_url,
            api_key="dummy",
        )

    @classmethod
    def for_qwen_local(cls, model_path: Optional[str] = None,
                       port: int = 8000) -> "LLMConfig":
        """创建本地 Qwen 模型配置

        Args:
            model_path: 模型路径，默认为 model/qwen
            port: 服务端口，默认 8000
        """
        project_root = os.getenv("PROJECT_ROOT", "/data/DC/MapAgent")
        default_path = os.path.join(project_root, "model", "qwen")

        return cls(
            provider=LLMProvider.QWEN_LOCAL,
            model="Qwen3_5",
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            local_model_path=model_path or default_path,
            local_model_type="qwen",
            project_root=project_root,
        )

    @classmethod
    def for_gemma4_local(cls, model_path: Optional[str] = None,
                         port: int = 8001) -> "LLMConfig":
        """创建本地 Gemma4 模型配置

        Args:
            model_path: 模型目录路径，默认为 model/gemma4
            port: 服务端口，默认 8001
        """
        project_root = os.getenv("PROJECT_ROOT", "/data/DC/MapAgent")
        default_path = os.path.join(project_root, "model", "gemma4")

        return cls(
            provider=LLMProvider.GEMMA4_LOCAL,
            model="Gemma4",
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            local_model_path=model_path or default_path,
            local_model_type="gemma4",
            project_root=project_root,
        )


@dataclass
class Message:
    """消息"""
    role: str  # user, assistant, system
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolResult:
    """工具调用结果"""
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool = False


class BaseLLMClient(ABC):
    """LLM 客户端基类"""

    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             system: str = "") -> str:
        """发送消息并获取回复"""
        pass

    @abstractmethod
    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        system: str = "",
                        tool_handler: Callable = None) -> str:
        """带工具调用的对话"""
        pass


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 客户端"""

    def __init__(self, config: LLMConfig):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed")
        self.config = config
        self.client = Anthropic(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             system: str = "") -> str:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # 提取文本
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        system: str = "",
                        tool_handler: Callable = None,
                        max_turns: int = 5) -> str:
        current_messages = messages.copy()

        for _ in range(max_turns):
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": current_messages,
                "tools": tools,
            }
            if system:
                kwargs["system"] = system

            response = self.client.messages.create(**kwargs)

            if response.stop_reason == "tool_use":
                # 提取工具调用
                tool_calls = []
                text_content = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block.type == "text":
                        text_content.append(block.text)

                # 添加助手消息
                current_messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # 执行工具并添加结果
                for tc in tool_calls:
                    if tool_handler:
                        result = tool_handler(tc["name"], tc["input"])
                    else:
                        result = {"error": "No tool handler"}

                    current_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False)
                        }]
                    })
            else:
                # 返回最终文本
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""

        return "超过最大工具调用轮数"


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 兼容客户端 (支持 Deepseek, 本地模型等)"""

    def __init__(self, config: LLMConfig):
        if not HAS_OPENAI:
            raise ImportError("openai package not installed")
        self.config = config

        client_kwargs = {"api_key": config.api_key or "dummy"}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = OpenAI(**client_kwargs)

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             system: str = "") -> str:
        # 添加系统消息
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": full_messages,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict],
                        system: str = "",
                        tool_handler: Callable = None,
                        max_turns: int = 5) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        for _ in range(max_turns):
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": full_messages,
                "temperature": self.config.temperature,
                "tools": self._convert_tools(tools),
            }

            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # 处理结构化 tool_calls
            if message.tool_calls:
                # 添加助手消息
                full_messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # 执行工具
                for tc in message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tool_handler:
                        result = tool_handler(tc.function.name, args)
                    else:
                        result = {"error": "No tool handler"}

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            elif message.content and "<|tool_call>" in message.content:
                # 处理 Gemma4 格式的 tool calls (文本格式)
                parsed_calls = self._parse_gemma4_tool_calls(message.content)
                if parsed_calls:
                    # 添加助手消息
                    full_messages.append({
                        "role": "assistant",
                        "content": message.content,
                    })

                    # 执行工具
                    for i, tc in enumerate(parsed_calls):
                        if tool_handler:
                            result = tool_handler(tc["name"], tc["arguments"])
                        else:
                            result = {"error": "No tool handler"}

                        full_messages.append({
                            "role": "tool",
                            "tool_call_id": f"call_{i}",
                            "content": json.dumps(result, ensure_ascii=False)
                        })
                else:
                    return message.content or ""
            else:
                return message.content or ""

        return "超过最大工具调用轮数"

    def _parse_gemma4_tool_calls(self, content: str) -> List[Dict]:
        """解析 Gemma4 格式的 tool calls

        格式: <|tool_call>call:function_name{param:value}<tool_call|>
        """
        import re
        tool_calls = []

        # 匹配 <|tool_call>call:name{json_args}<tool_call|>
        # 使用更宽松的匹配，因为参数可能包含嵌套内容
        pattern = r'<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>'
        matches = re.findall(pattern, content, re.DOTALL)

        for name, args_str in matches:
            try:
                # 尝试解析参数（可能是 JSON 或特殊格式）
                # 处理 <|"|>| 格式的引号（Gemma4 特殊格式）
                args_str_clean = args_str.replace('<|"|>', '"').replace('<|">|', '"').replace('<|">>', '"')

                # 尝试解析为 JSON
                args = {}
                if args_str_clean.strip():
                    try:
                        args = json.loads(args_str_clean)
                    except json.JSONDecodeError:
                        # 如果不是标准 JSON，尝试简单的键值解析
                        # 格式可能是 key:value 或 key:"value"
                        parts = args_str_clean.split(',')
                        for part in parts:
                            part = part.strip()
                            if ':' in part:
                                k, v = part.split(':', 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                args[k] = v

                tool_calls.append({
                    "name": name,
                    "arguments": args
                })
            except Exception as e:
                print(f"解析 tool call 失败: {e}")
                continue

        return tool_calls

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """转换工具格式为 OpenAI 格式"""
        openai_tools = []
        for tool in tools:
            input_schema = tool.get("input_schema", {})
            # 确保有 type: object
            if "type" not in input_schema:
                input_schema = {"type": "object", "properties": input_schema.get("properties", {})}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": input_schema
                }
            })
        return openai_tools


class LLMClient:
    """
    统一 LLM 客户端

    支持:
    - 多种 LLM 提供商
    - 多轮对话
    - Function Calling
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._client = self._create_client()
        self._tools: List[Dict] = []
        self._tool_handlers: Dict[str, Callable] = {}
        self._conversation_history: List[Message] = []

    def _create_client(self) -> BaseLLMClient:
        """创建底层客户端"""
        if self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(self.config)
        else:
            # Deepseek, OpenAI, Local, Qwen_local, Gemma4_local 都使用 OpenAI 兼容接口
            return OpenAICompatibleClient(self.config)

    def register_tool(self, name: str, description: str,
                      parameters: Dict, handler: Callable) -> None:
        """注册工具"""
        tool_def = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
            }
        }
        self._tools.append(tool_def)
        self._tool_handlers[name] = handler

    def clear_history(self):
        """清空对话历史"""
        self._conversation_history = []

    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self._conversation_history.append(Message(role=role, content=content))

    def _handle_tool_call(self, name: str, args: Dict) -> Any:
        """处理工具调用"""
        handler = self._tool_handlers.get(name)
        if handler:
            try:
                return handler(**args)
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown tool: {name}"}

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             system: str = "", use_tools: bool = True,
             tool_handler: Callable = None, max_turns: int = 5) -> str:
        """
        发送消息并获取回复

        支持两种调用方式:
        1. messages + tools: 完整对话控制（MasterAgent 使用）
        2. user_message + system_prompt: 简单对话（已废弃）

        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
            tools: 工具定义列表
            system: 系统提示
            use_tools: 是否使用工具
            tool_handler: 工具调用处理函数
            max_turns: 最大工具调用轮数

        Returns:
            助手回复文本
        """
        # 如果提供了 tools 和 tool_handler，直接使用
        if use_tools and tools and tool_handler:
            response = self._client.chat_with_tools(
                messages=messages,
                tools=tools,
                system=system,
                tool_handler=tool_handler,
                max_turns=max_turns
            )
            return response
        else:
            response = self._client.chat(
                messages=messages,
                tools=tools,
                system=system
            )
            return response

    def chat_simple(self, user_message: str, system_prompt: str = "") -> str:
        """简单对话（不使用工具）"""
        messages = [{"role": "user", "content": user_message}]
        return self._client.chat(messages=messages, system=system_prompt)


# 便捷函数
def create_client(provider: str = "anthropic", **kwargs) -> LLMClient:
    """
    创建 LLM 客户端

    Args:
        provider: 提供商名称，支持:
            - anthropic / claude
            - deepseek
            - openai
            - local
            - qwen / qwen_local
            - gemma4 / gemma4_local
        **kwargs: 配置参数

    Returns:
        LLMClient 实例
    """
    provider_map = {
        "anthropic": LLMProvider.ANTHROPIC,
        "claude": LLMProvider.ANTHROPIC,
        "deepseek": LLMProvider.DEEPSEEK,
        "openai": LLMProvider.OPENAI,
        "local": LLMProvider.LOCAL,
        "qwen": LLMProvider.QWEN_LOCAL,
        "qwen_local": LLMProvider.QWEN_LOCAL,
        "gemma4": LLMProvider.GEMMA4_LOCAL,
        "gemma4_local": LLMProvider.GEMMA4_LOCAL,
    }

    config = LLMConfig(
        provider=provider_map.get(provider.lower(), LLMProvider.ANTHROPIC),
        **kwargs
    )
    return LLMClient(config)


def create_qwen_client(port: int = 8000, **kwargs) -> LLMClient:
    """创建本地 Qwen 模型客户端"""
    config = LLMConfig.for_qwen_local(port=port, **kwargs)
    return LLMClient(config)


def create_gemma4_client(port: int = 8001, gguf_file: str = "gemma-4-31B-it-Q4_K_M.gguf", **kwargs) -> LLMClient:
    """创建本地 Gemma4 模型客户端"""
    config = LLMConfig.for_gemma4_local(port=port, gguf_file=gguf_file, **kwargs)
    return LLMClient(config)