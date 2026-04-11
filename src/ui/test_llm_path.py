#!/usr/bin/env python
"""Testing LLM Path"""

import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.traffic_flow import TrafficFlowAgent
from agents.base import AgentContext
from config import settings
from core.llm_client import LLMClient, LLMConfig, LLMProvider

print("=" * 60)
print("Testing LLM Path")
print("=" * 60)

# 1. Create MapAPI
print("\n[1] Create MapAPI...")
map_api = MapAPI(map_file=str(settings.map_path))
print(f"    map_api is not None: {map_api is not None}")

# 2. Create LLMClient
print("\n[2] Create LLMClient...")
config = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key="test_key")
llm_client = LLMClient(config)
print(f"    llm_client is not None: {llm_client is not None}")

# 3. Create AgentContext
print("\n[3] Create AgentContext...")
context = AgentContext(map_api=map_api, llm_client=llm_client)
print(f"    context.llm_client is not None: {context.llm_client is not None}")

# 4. Create TrafficFlowAgent (use_llm=True)
print("\n[4] Create TrafficFlowAgent (use_llm=True)...")
tf_agent = TrafficFlowAgent(context, use_llm=True)
print(f"    tf_agent._use_llm: {tf_agent._use_llm}")
print(f"    tf_agent._llm_optimizer is not None: {tf_agent._llm_optimizer is not None}")
print(f"    tf_agent.map_api is not None: {tf_agent.map_api is not None}")
print(f"    tf_agent.name: {tf_agent.name}")

# 5. Checkcondition
print("\n[5] Check LLM condition...")
condition = tf_agent._use_llm and tf_agent._llm_optimizer is not None and tf_agent.map_api is not None
print(f"    use_llm AND llm_optimizer AND map_api: {condition}")

if condition:
    print("\n✅ LLM condition， LLM")
else:
    print("\n❌ LLM conditionnot")
    if not tf_agent._use_llm:
        print("   - _use_llm is False")
    if not tf_agent._llm_optimizer:
        print("   - _llm_optimizer is None")
    if not tf_agent.map_api:
        print("   - map_api is None")

print("\n" + "=" * 60)
