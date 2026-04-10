"""配置管理"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置"""

    # 应用信息
    app_name: str = "MapAgent"
    debug: bool = False

    # 地图配置
    map_file: str = Field(
        default="data/vector_map.json",
        description="矢量地图文件路径"
    )

    # LLM 配置
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"  # deepseek-chat 或 deepseek-reasoner
    llm_api_key: Optional[str] = "sk-96f9b91a59b749b68dafb650f6966e8b" 
    llm_base_url: Optional[str] = "https://api.deepseek.com"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # 本地模型配置
    local_model_type: str = "qwen"  # qwen 或 gemma4
    local_model_path: Optional[str] = None  # 本地模型路径
    local_model_base_url: str = "http://localhost:8000/v1"  # vLLM/llama.cpp 服务地址
    local_model_port: int = 8000  # 服务端口

    # 日志配置
    log_level: str = "INFO"

    # 项目根目录
    project_root: Path = Path(__file__).parent.parent

    class Config:
        env_file = ".env"
        env_prefix = "MAPAGENT_"

    @property
    def map_path(self) -> Path:
        """获取地图文件的绝对路径"""
        path = Path(self.map_file)
        if not path.is_absolute():
            # 相对路径，相对于项目根目录
            path = self.project_root / path
        return path

    def get_local_model_config(self) -> dict:
        """获取本地模型配置"""
        configs = {
            "qwen": {
                "model_name": "Qwen3_5",
                "model_path": str(self.project_root / "model" / "qwen"),
                "default_port": 8000,
            },
            "gemma4": {
                "model_name": "Gemma4",
                "model_path": str(self.project_root / "model" / "gemma4"),
                "default_port": 8001,
            }
        }
        return configs.get(self.local_model_type, configs["qwen"])


# 全局配置实例
settings = Settings()