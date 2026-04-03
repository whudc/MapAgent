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
    llm_api_key: Optional[str] = "sk-96f9b91a59b749b68dafb650f6966e8b"  # 从环境变量读取
    llm_base_url: Optional[str] = "https://api.deepseek.com"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # 日志配置
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "MAPAGENT_"

    @property
    def map_path(self) -> Path:
        """获取地图文件的绝对路径"""
        path = Path(self.map_file)
        if not path.is_absolute():
            # 相对路径，相对于项目根目录
            root = Path(__file__).parent.parent  # config -> MapAgent
            path = root / path
        return path


# 全局配置实例
settings = Settings()