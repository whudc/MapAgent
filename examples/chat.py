#!/usr/bin/env python3
"""
MapAgent 交互式命令行界面

使用方法:
    python examples/chat.py

命令:
    输入问题进行对话
    输入 'quit' 或 'exit' 退出
    输入 'clear' 清空对话历史
    输入 'help' 查看帮助
    输入 '/model <provider>' 切换模型 (如: /model qwen_local)
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.master import MasterAgent, create_master_agent
from core.llm_client import LLMClient, LLMConfig
from config import settings


# 支持的模型提供商
MODEL_PROVIDERS = {
    "deepseek": {"name": "Deepseek", "default_model": "deepseek-chat", "need_key": True},
    "anthropic": {"name": "Anthropic Claude", "default_model": "claude-sonnet-4-6", "need_key": True},
    "openai": {"name": "OpenAI", "default_model": "gpt-4o", "need_key": True},
    "qwen": {"name": "Qwen (本地)", "default_model": "Qwen3___5-35B-A3B", "need_key": False, "port": 8000},
    "qwen_local": {"name": "Qwen (本地)", "default_model": "Qwen3___5-35B-A3B", "need_key": False, "port": 8000},
    "gemma4": {"name": "Gemma4 (本地)", "default_model": "gemma-4-31B-it", "need_key": False, "port": 8001},
    "gemma4_local": {"name": "Gemma4 (本地)", "default_model": "gemma-4-31B-it", "need_key": False, "port": 8001},
    "local": {"name": "本地模型", "default_model": "Qwen3___5-35B-A3B", "need_key": False, "port": 8000},
}


def print_banner(current_provider: str):
    """打印欢迎信息"""
    provider_info = MODEL_PROVIDERS.get(current_provider, {})
    model_name = provider_info.get("name", current_provider)

    print("\n" + "=" * 60)
    print("  MapAgent - 多Agent地图问答系统 (LLM驱动)")
    print("=" * 60)
    print(f"\n当前模型: {model_name}")
    print("\n欢迎使用 MapAgent！我可以帮助你：")
    print("  - 理解道路场景（车道数量、类型、路口结构）")
    print("  - 分析车辆行为（预测转向、变道、风险评估）")
    print("  - 规划行驶路径（路线建议、时间估算）")
    print("\n输入 'help' 查看帮助，输入 'quit' 退出")
    print("输入 '/model <provider>' 切换模型")
    print("-" * 60)


def print_help():
    """打印帮助信息"""
    print("\n帮助信息:")
    print("-" * 40)
    print("场景理解类问题示例:")
    print("  - 这个位置有几条车道？")
    print("  - 这个路口的结构是什么？")
    print("  - 附近有什么交通标志？")
    print("\n行为分析类问题示例:")
    print("  - 这辆车会右转吗？")
    print("  - 预测这辆车的行为")
    print("  - 两车会有碰撞风险吗？")
    print("\n路径规划类问题示例:")
    print("  - 从这到路口怎么走？")
    print("  - 帮我规划一条路线")
    print("  - 到目的地要多久？")
    print("\n模型切换命令:")
    print("  /model deepseek      - 切换到 Deepseek")
    print("  /model qwen          - 切换到本地 Qwen 模型")
    print("  /model gemma4        - 切换到本地 Gemma4 模型")
    print("  /model list          - 显示所有可用模型")
    print("\n其他命令:")
    print("  help  - 显示帮助")
    print("  clear - 清空对话历史")
    print("  quit  - 退出程序")
    print("-" * 40)


def print_model_list():
    """打印可用模型列表"""
    print("\n可用模型列表:")
    print("-" * 40)
    for key, info in MODEL_PROVIDERS.items():
        key_status = "需API Key" if info["need_key"] else "本地部署"
        port_info = f" (端口 {info.get('port')})" if not info["need_key"] else ""
        print(f"  {key:15} - {info['name']}{port_info} [{key_status}]")
    print("-" * 40)


def switch_model(provider: str, map_path: str, api_key: str = None) -> tuple:
    """
    切换模型

    Args:
        provider: 模型提供商
        map_path: 地图文件路径
        api_key: API Key (可选)

    Returns:
        (agent, success, message)
    """
    provider = provider.lower().strip()

    if provider == "list":
        print_model_list()
        return None, False, None

    if provider not in MODEL_PROVIDERS:
        return None, False, f"未知模型: {provider}。可用模型: {', '.join(MODEL_PROVIDERS.keys())}"

    info = MODEL_PROVIDERS[provider]

    # 本地模型需要检查服务是否运行
    if not info["need_key"]:
        port = info.get("port", 8000)
        import requests
        try:
            requests.get(f"http://localhost:{port}/v1/models", timeout=2)
        except:
            return None, False, f"本地模型服务未启动！请先运行: ./scripts/start_local_model.sh --{provider.replace('_local', '')}"

    # 获取 API Key
    resolved_api_key = api_key
    if not resolved_api_key and info["need_key"]:
        if provider == "deepseek":
            resolved_api_key = os.getenv("DEEPSEEK_API_KEY") or settings.llm_api_key
        elif provider == "anthropic":
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openai":
            resolved_api_key = os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            return None, False, f"需要 API Key！请设置环境变量或通过 /key 命令提供"

    if not resolved_api_key and not info["need_key"]:
        resolved_api_key = "dummy"

    # 设置环境变量
    if not info["need_key"]:
        port = info.get("port", 8000)
        if provider in ["qwen", "qwen_local", "local"]:
            os.environ["QWEN_BASE_URL"] = f"http://localhost:{port}/v1"
        elif provider in ["gemma4", "gemma4_local"]:
            os.environ["GEMMA4_BASE_URL"] = f"http://localhost:{port}/v1"

    try:
        agent = create_master_agent(
            map_file=map_path,
            llm_provider=provider,
            api_key=resolved_api_key
        )
        return agent, True, f"已切换到 {info['name']}"
    except Exception as e:
        return None, False, f"切换失败: {e}"


def main():
    # 检查 LLM 配置
    llm_provider = os.getenv("LLM_PROVIDER", settings.llm_provider)

    print(f"LLM 提供商: {llm_provider}")

    # 加载地图
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    print(f"正在加载地图...")

    # 初始化地图 API
    map_api = MapAPI(map_file=str(map_path))

    # 创建 Agent
    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=llm_provider,
    )

    summary = map_api.get_map_summary()
    print(f"地图加载完成: {summary['total_lanes']} 条车道, {summary['total_intersections']} 个路口\n")

    print_banner(llm_provider)

    # 当前提供商
    current_provider = llm_provider

    # 主循环
    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if user_input.lower() == 'clear':
                agent.clear_history()
                print("对话历史已清空。")
                continue

            # 处理 /model 命令
            if user_input.startswith('/model'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("用法: /model <provider> 或 /model list")
                    print_model_list()
                    continue

                target_provider = parts[1].strip()
                new_agent, success, msg = switch_model(target_provider, str(map_path))

                if success:
                    agent = new_agent
                    current_provider = target_provider
                    print(f"\n✓ {msg}")
                else:
                    print(f"\n✗ {msg}")
                continue

            # 处理 /key 命令 (设置 API Key)
            if user_input.startswith('/key'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("用法: /key <your-api-key>")
                    continue
                api_key = parts[1].strip()
                new_agent, success, msg = switch_model(current_provider, str(map_path), api_key)
                if success:
                    agent = new_agent
                    print(f"\n✓ API Key 已更新")
                else:
                    print(f"\n✗ {msg}")
                continue

            # 处理查询 (通过 LLM)
            response = agent.chat(user_input)
            print(f"\n助手: {response}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()