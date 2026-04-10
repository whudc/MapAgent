#!/usr/bin/env python3
"""
启动 MapAgent Web Server

使用方法:
    python run_ui.py                              # 使用默认配置
    python run_ui.py --port 9000                  # 在端口 9000 启动
    python run_ui.py --share                      # 启用 ngrok 分享

命令行参数:
    --port         Web 服务端口 (默认 7860)
    --debug        启用调试模式
    --share        启用 ngrok 公共分享链接
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config import settings


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动 MapAgent Web Server")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Web 服务端口 (默认 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="启用 ngrok 公共分享链接"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("MapAgent Web Server")
    print("=" * 50)
    print(f"地图文件: {settings.map_path}")
    print(f"访问地址: http://localhost:{args.port}")

    if args.share:
        print("分享模式: 将生成 ngrok 公共链接")

    print("\n按 Ctrl+C 停止服务")
    print("=" * 50)

    # 导入并启动 Flask 应用
    from ui.server import app

    # ngrok 分享
    if args.share:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(args.port)
            print(f"\n公共访问链接: {public_url}")
            print("=" * 50)
        except ImportError:
            print("\n警告: pyngrok 未安装，无法启用分享功能")
            print("安装: pip install pyngrok")

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()