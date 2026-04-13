#!/usr/bin/env python3
"""
MapAgent 统一启动脚本

支持两种运行模式:
- chat: 交互式命令行对话
- ui:   Web 可视化界面

用法:
    python run.py chat          # 启动命令行对话
    python run.py ui            # 启动 Web 界面
    python run.py               # 默认启动 Web 界面

注意：请在 mapagent conda 环境中运行:
    conda activate mapagent
    python run.py [chat|ui]
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def check_map_file(map_path: str) -> bool:
    """检查地图文件是否存在"""
    path = Path(map_path)
    if not path.exists():
        print(f"错误：地图文件不存在：{path}")
        print(f"\n请先准备地图数据:")
        print(f"  1. 将矢量地图文件放置于：data/vector_map.json")
        print(f"  2. 或使用生成脚本:")
        print(f"     python generate_vector_map.py --data-dir ./data/00/annotations --output ./data/vector_map.json")
        return False
    return True


def check_llm_config(provider: str) -> bool:
    """检查 LLM 配置"""
    from config import settings

    # 本地模型不需要 API Key
    if provider in ["qwen", "qwen_local", "gemma4", "gemma4_local", "local"]:
        return True

    # 检查 API Key
    api_key = os.getenv("LLM_API_KEY") or settings.llm_api_key
    if not api_key:
        print(f"\n警告：未检测到 {provider.upper()} API Key")
        print(f"请设置环境变量:")
        if provider == "deepseek":
            print(f"  export DEEPSEEK_API_KEY=your-key")
        elif provider == "anthropic":
            print(f"  export ANTHROPIC_API_KEY=your-key")
        elif provider == "openai":
            print(f"  export OPENAI_API_KEY=your-key")
        print(f"\n或在使用时通过 /key 命令提供")
        return False
    return True


def run_chat_mode(args):
    """启动命令行对话模式"""
    from examples.chat import main as chat_main

    print("\n" + "=" * 50)
    print("  MapAgent - 命令行对话模式")
    print("=" * 50)

    # 检查地图文件
    map_path = project_root / "data" / "vector_map.json"
    if not check_map_file(str(map_path)):
        return

    # 检查 LLM 配置
    check_llm_config(os.getenv("LLM_PROVIDER", "deepseek"))

    # 启动对话
    chat_main()


def run_ui_mode(args):
    """启动 Web 界面模式"""
    from src.ui.server import main as ui_main

    print("\n" + "=" * 50)
    print("  MapAgent - Web 可视化界面")
    print("=" * 50)

    # 检查地图文件
    map_path = project_root / "data" / "vector_map.json"
    if not check_map_file(str(map_path)):
        return

    # 检查 LLM 配置
    check_llm_config(os.getenv("LLM_PROVIDER", "deepseek"))

    print(f"\n访问地址：http://localhost:7860")
    print(f"按 Ctrl+C 停止服务\n")

    # 启动 Web 服务
    ui_main()


def main():
    parser = argparse.ArgumentParser(
        description="MapAgent - 多 Agent 地图问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py chat     启动命令行对话模式
  python run.py ui       启动 Web 可视化界面
  python run.py          默认启动 Web 界面

环境变量:
  LLM_PROVIDER         LLM 提供商 (deepseek/anthropic/openai/qwen/gemma4)
  DEEPSEEK_API_KEY     DeepSeek API Key
  ANTHROPIC_API_KEY    Anthropic API Key
  OPENAI_API_KEY       OpenAI API Key
        """
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["chat", "ui"],
        default="ui",
        help="运行模式：chat(命令行对话) 或 ui(Web 界面)，默认 ui"
    )

    args = parser.parse_args()

    if args.mode == "chat":
        run_chat_mode(args)
    elif args.mode == "ui":
        run_ui_mode(args)


if __name__ == "__main__":
    main()
