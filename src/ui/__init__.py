"""MapAgent UI 模块

提供两种 UI 模式：
1. Gradio 版 (app.py) - 使用 Plotly 渲染
2. Flask 纯前端版 (server.py) - 使用 Canvas 渲染，性能更优
"""

__all__ = ["create_ui", "MapAgentUI", "FastMapVisualizer"]

# 懒加载，避免导入错误
try:
    from .app import create_ui, MapAgentUI, FastMapVisualizer
except ImportError:
    # Gradio 未安装时使用占位符
    create_ui = None
    MapAgentUI = None
    FastMapVisualizer = None