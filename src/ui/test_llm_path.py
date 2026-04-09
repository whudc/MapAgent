#!/usr/bin/env python
"""测试 LLM 调用路径"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.traffic_flow import TrafficFlowAgent
from agents.base import AgentContext
from config import settings
from core.llm_client import LLMClient, LLMConfig, LLMProvider

print("=" * 60)
print("测试 LLM 调用路径")
print("=" * 60)

# 1. 创建 MapAPI
print("\n[1] 创建 MapAPI...")
map_api = MapAPI(map_file=str(settings.map_path))
print(f"    map_api is not None: {map_api is not None}")

# 2. 创建 LLMClient
print("\n[2] 创建 LLMClient...")
config = LLMConfig(provider=LLMProvider.DEEPSEEK, api_key="test_key")
llm_client = LLMClient(config)
print(f"    llm_client is not None: {llm_client is not None}")

# 3. 创建 AgentContext
print("\n[3] 创建 AgentContext...")
context = AgentContext(map_api=map_api, llm_client=llm_client)
print(f"    context.llm_client is not None: {context.llm_client is not None}")

# 4. 创建 TrafficFlowAgent (use_llm=True)
print("\n[4] 创建 TrafficFlowAgent (use_llm=True)...")
tf_agent = TrafficFlowAgent(context, use_llm=True)
print(f"    tf_agent._use_llm: {tf_agent._use_llm}")
print(f"    tf_agent._llm_optimizer is not None: {tf_agent._llm_optimizer is not None}")
print(f"    tf_agent.map_api is not None: {tf_agent.map_api is not None}")
print(f"    tf_agent.name: {tf_agent.name}")

# 5. 检查条件
print("\n[5] 检查 LLM 调用条件...")
condition = tf_agent._use_llm and tf_agent._llm_optimizer is not None and tf_agent.map_api is not None
print(f"    use_llm AND llm_optimizer AND map_api: {condition}")

if condition:
    print("\n✅ LLM 调用条件满足，应该能够调用 LLM")
else:
    print("\n❌ LLM 调用条件不满足")
    if not tf_agent._use_llm:
        print("   - _use_llm is False")
    if not tf_agent._llm_optimizer:
        print("   - _llm_optimizer is None")
    if not tf_agent.map_api:
        print("   - map_api is None")

print("\n" + "=" * 60)
