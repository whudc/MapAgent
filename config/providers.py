"""
LLM Provider Configuration

统一 LLM 提供商配置，避免在多处重复定义
"""

from enum import Enum
from typing import Dict, Optional


class LLMProvider(str, Enum):
    """LLM 提供商"""
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    LOCAL = "local"
    QWEN_LOCAL = "qwen_local"
    GEMMA4_LOCAL = "gemma4_local"


# 提供商显示名称
PROVIDER_NAMES: Dict[str, str] = {
    "deepseek": "Deepseek",
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "qwen": "Qwen (本地)",
    "qwen_local": "Qwen (本地)",
    "gemma4": "Gemma4 (本地)",
    "gemma4_local": "Gemma4 (本地)",
    "local": "本地模型",
}


# 默认模型
DEFAULT_MODELS: Dict[str, str] = {
    LLMProvider.ANTHROPIC: "claude-sonnet-4-6",
    LLMProvider.DEEPSEEK: "deepseek-chat",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.LOCAL: "Qwen3_5",
    LLMProvider.QWEN_LOCAL: "Qwen3_5",
    LLMProvider.GEMMA4_LOCAL: "Gemma4",
}


# 本地模型端口
LOCAL_MODEL_PORTS: Dict[str, int] = {
    "qwen": 8000,
    "qwen_local": 8000,
    "local": 8000,
    "gemma4": 8001,
    "gemma4_local": 8001,
}


# 提供商映射（字符串 -> LLMProvider）
PROVIDER_MAP: Dict[str, LLMProvider] = {
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


def get_provider(provider_str: str) -> LLMProvider:
    """
    获取 LLMProvider 枚举

    Args:
        provider_str: 提供商字符串

    Returns:
        LLMProvider 枚举值
    """
    return PROVIDER_MAP.get(provider_str.lower(), LLMProvider.ANTHROPIC)


def get_default_model(provider: str) -> str:
    """
    获取默认模型名称

    Args:
        provider: 提供商字符串

    Returns:
        默认模型名称
    """
    prov = get_provider(provider)
    return DEFAULT_MODELS.get(prov, "claude-sonnet-4-6")


def get_local_model_port(provider: str) -> int:
    """
    获取本地模型端口

    Args:
        provider: 提供商字符串

    Returns:
        端口号，如果不是本地模型返回 8000
    """
    return LOCAL_MODEL_PORTS.get(provider.lower(), 8000)


def is_local_model(provider: str) -> bool:
    """
    判断是否是本地模型

    Args:
        provider: 提供商字符串

    Returns:
        是否是本地模型
    """
    return provider.lower() in ["qwen", "qwen_local", "gemma4", "gemma4_local", "local"]


def get_base_url(provider: str, port: Optional[int] = None) -> Optional[str]:
    """
    获取 base_url

    Args:
        provider: 提供商字符串
        port: 端口号（本地模型使用）

    Returns:
        base_url
    """
    provider = provider.lower()

    if provider in ["qwen", "qwen_local"]:
        return f"http://localhost:{port or 8000}/v1"
    elif provider in ["gemma4", "gemma4_local"]:
        return f"http://localhost:{port or 8001}/v1"
    elif provider == "local":
        return f"http://localhost:{port or 8000}/v1"
    elif provider == "deepseek":
        return "https://api.deepseek.com"

    return None
