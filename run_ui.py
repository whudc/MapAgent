#!/usr/bin/env python3
"""
启动 MapAgent Web UI

使用方法:
    python run_ui.py

然后访问 http://localhost:7860
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ui.app import create_ui

if __name__ == "__main__":
    print("=" * 50)
    print("MapAgent Web UI")
    print("=" * 50)
    print("\n正在启动服务...")
    print("访问地址: http://localhost:7860")
    print("\n按 Ctrl+C 停止服务\n")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )