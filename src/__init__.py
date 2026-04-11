"""MapAgent Source package"""

import sys
from pathlib import Path

# Ensure module import
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models.map_data import (
    LaneLine,
    Centerline,
    VectorMap,
    MapLoader,
)
from models.agent_io import (
    IntentType,
    Intent,
    SceneQuery,
    SceneResult,
    BehaviorQuery,
    BehaviorResult,
    PathQuery,
    PathResult,
)
from core.llm_client import LLMClient, LLMConfig
from core.tools import ToolRegistry
from apis.map_api import MapAPI
from agents import (
    SceneAgent,
    BehaviorAgent,
    PathAgent,
    MasterAgent,
    create_master_agent,
)

__version__ = "0.1.0"

__all__ = [
    # Data models
    "LaneLine",
    "Centerline",
    "VectorMap",
    "MapLoader",
    # Agent IO
    "IntentType",
    "Intent",
    "SceneQuery",
    "SceneResult",
    "BehaviorQuery",
    "BehaviorResult",
    "PathQuery",
    "PathResult",
    # Core
    "LLMClient",
    "LLMConfig",
    "ToolRegistry",
    # API
    "MapAPI",
    # Agents
    "SceneAgent",
    "BehaviorAgent",
    "PathAgent",
    "MasterAgent",
    "create_master_agent",
]
