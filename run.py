#!/usr/bin/env python3
"""
MapAgent unified startup script

Supports two running modes:
- chat: Interactive CLI chat
- ui:   Web Visualization UI

Usage:
    python run.py chat          # Start CLI chat
    python run.py ui            # Start Web UI
    python run.py               # Default to Web UI

Note: Please run in mapagent conda environment:
    conda activate mapagent
    python run.py [chat|ui]
"""

import sys
import os
import argparse
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def check_conda_env():
    """Check if in correct conda environment"""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")

    # Check environment name
    if env_name and "mapagent" in env_name.lower():
        return True

    # Check path
    if conda_prefix and "mapagent" in conda_prefix.lower():
        return True

    return False


def check_map_file(map_path: str) -> bool:
    """Check if map file exists"""
    path = Path(map_path)
    if not path.exists():
        print(f"Error：Map file does not exist：{path}")
        print(f"\nPlease prepare map data first:")
        print(f"  1. Place vector map file at：data/vector_map.json")
        print(f"  2. Or use generation script:")
        print(f"     python generate_vector_map.py --data-dir ./data/00/annotations --output ./data/vector_map.json")
        return False
    return True


def check_llm_config(provider: str) -> bool:
    """Check LLM configuration"""
    from config import settings

    # Local model does not require API Key
    if provider in ["qwen", "qwen_local", "gemma4", "gemma4_local", "local"]:
        return True

    # Check API Key
    api_key = os.getenv("LLM_API_KEY") or settings.llm_api_key
    if not api_key:
        print(f"\nWarning：Not detected {provider.upper()} API Key")
        print(f"Please set environment variable:")
        if provider == "deepseek":
            print(f"  export DEEPSEEK_API_KEY=your-key")
        elif provider == "anthropic":
            print(f"  export ANTHROPIC_API_KEY=your-key")
        elif provider == "openai":
            print(f"  export OPENAI_API_KEY=your-key")
        print(f"\nOr provide via /key command when using")
        return False
    return True


def run_chat_mode(args):
    """Start CLI chat mode"""
    from examples.chat import main as chat_main

    print("\n" + "=" * 50)
    print("  MapAgent - CLI chatMode")
    print("=" * 50)

    # Check map file
    map_path = project_root / "data" / "vector_map.json"
    if not check_map_file(str(map_path)):
        return

    # Check LLM configuration
    check_llm_config(os.getenv("LLM_PROVIDER", "deepseek"))

    # Start chat
    chat_main()


def run_ui_mode(args):
    """Start Web UI mode"""
    from src.ui.server import main as ui_main

    print("\n" + "=" * 50)
    print("  MapAgent - Web Visualization UI")
    print("=" * 50)

    # Check map file
    map_path = project_root / "data" / "vector_map.json"
    if not check_map_file(str(map_path)):
        return

    # Check LLM configuration
    check_llm_config(os.getenv("LLM_PROVIDER", "deepseek"))

    print(f"\nAccess URL：http://localhost:7860")
    print(f"Press Ctrl+C to stop service\n")

    # Start Web service
    ui_main()


def main():
    parser = argparse.ArgumentParser(
        description="MapAgent - Multi-Agent Map Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python run.py chat     Start CLI chat mode
  python run.py ui       Start Web Visualization UI
  python run.py          DefaultStart Web UI

Environment variables:
  LLM_PROVIDER         LLM Provider (deepseek/anthropic/openai/qwen/gemma4)
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
        help="Running mode：chat(CLI chat) or ui(Web UI)，Default ui"
    )

    args = parser.parse_args()

    # Check conda environment
    if not check_conda_env():
        print("\n" + "=" * 50)
        print("  Warning：Not detected mapagent conda Environment")
        print("=" * 50)
        print("\nPlease activate conda environment first:")
        print("  conda activate mapagent")
        print("\nOr directly use Python in environment:")
        print("  D:\\program\\conda_envs\\mapagnet\\python.exe run.py", args.mode)
        print("=" * 50)

        # Try to continue using current Python path
        if "mapagent" in sys.executable.lower():
            print("\nDetected executable path contains mapagent, continuing...\n")
        else:
            print("\nWill continue using current Python environment\n")

    if args.mode == "chat":
        run_chat_mode(args)
    elif args.mode == "ui":
        run_ui_mode(args)


if __name__ == "__main__":
    main()
