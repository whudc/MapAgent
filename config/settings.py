"""Configuration"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # Application info
    app_name: str = "MapAgent"
    debug: bool = False

    # Map settings
    map_file: str = Field(
        default="data/vector_map.json",
        description="Vector map file path"
    )

    # LLM Configuration
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"  # deepseek-chat  deepseek-reasoner
    llm_api_key: Optional[str] = "sk-96f9b91a59b749b68dafb650f6966e8b" 
    llm_base_url: Optional[str] = "https://api.deepseek.com"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.7

    # LocalmodelsConfiguration
    local_model_type: str = "qwen"  # qwen  gemma4
    local_model_path: Optional[str] = None  # LocalmodelsPath
    local_model_base_url: str = "http://localhost:8000/v1"  # vLLM/llama.cpp ly
    local_model_port: int = 8000  # Service port

    # LogConfiguration
    log_level: str = "INFO"

    # Project root
    project_root: Path = Path(__file__).parent.parent

    class Config:
        env_file = ".env"
        env_prefix = "MAPAGENT_"

    @property
    def map_path(self) -> Path:
        """Get absolute path of map file"""
        path = Path(self.map_file)
        if not path.is_absolute():
            # onPath，onProject root
            path = self.project_root / path
        return path

    def get_local_model_config(self) -> dict:
        """GetLocalmodelsConfiguration"""
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


# Configurationsolid
settings = Settings()