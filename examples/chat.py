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
"""

import sys
from pathlib import Path

# 添加 src 到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from apis.map_api import MapAPI
from agents.master import MasterAgent, create_master_agent
from core.llm_client import LLMClient, LLMConfig


def print_banner():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("  MapAgent - 多Agent地图问答系统 (LLM驱动)")
    print("=" * 60)
    print("\n欢迎使用 MapAgent！我可以帮助你：")
    print("  - 理解道路场景（车道数量、类型、路口结构）")
    print("  - 分析车辆行为（预测转向、变道、风险评估）")
    print("  - 规划行驶路径（路线建议、时间估算）")
    print("\n输入 'help' 查看帮助，输入 'quit' 退出")
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
    print("\n其他命令:")
    print("  help  - 显示帮助")
    print("  clear - 清空对话历史")
    print("  quit  - 退出程序")
    print("-" * 40)


def main():
    import os

    # 检查 LLM 配置
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek")

    print(f"LLM 提供商: {llm_provider}")

    # 加载地图
    map_path = Path(__file__).parent.parent / "data" / "vector_map.json"
    if not map_path.exists():
        print(f"错误: 地图文件不存在 {map_path}")
        return

    print(f"正在加载地图...")

    # 使用便捷函数创建 Agent（会自动配置 LLM）
    agent = create_master_agent(
        map_file=str(map_path),
        llm_provider=llm_provider,
    )

    summary = agent.map_api.get_map_summary()
    print(f"地图加载完成: {summary['total_lanes']} 条车道, {summary['total_intersections']} 个路口\n")

    print_banner()

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