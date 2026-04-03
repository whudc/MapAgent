"""
MapAgent Web UI - 优化版

使用缓存和视图裁剪优化渲染速度
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import gradio as gr
import plotly.graph_objects as go
import numpy as np

from apis.map_api import MapAPI
from agents.master import MasterAgent, create_master_agent
from config import settings


class FastMapVisualizer:
    """快速地图可视化器 - 使用 Plotly"""

    def __init__(self, map_api: MapAPI):
        self.map_api = map_api
        self._cache = None
        self._build_cache()

    def _build_cache(self):
        """预构建数据缓存"""
        self._cache = {
            'lanes': {},
            'centerlines': {}
        }

        # 车道颜色映射
        lane_colors = {
            'solid': '#FFD700',
            'dashed': '#90EE90',
            'double_solid': '#FF6B6B',
            'double_dashed': '#87CEEB',
            'bilateral': '#DDA0DD',
            'left_dashed_right_solid': '#F0E68C',
            'curb': '#808080',
            'fence': '#8B4513',
            'diversion_boundary': '#00CED1',
            'no_lane': '#FFFFFF',
        }

        # 缓存车道
        for lane_id, lane in self.map_api.map.lane_lines.items():
            coords = np.array(lane.coordinates)
            if len(coords.shape) == 1:
                coords = coords.reshape(1, -1)
            if len(coords) >= 2:
                self._cache['lanes'][lane_id] = {
                    'x': coords[:, 0].tolist(),
                    'y': coords[:, 1].tolist(),
                    'type': lane.type,
                    'color': lane_colors.get(lane.type, '#CCCCCC')
                }

        # 缓存中心线
        for cl_id, cl in self.map_api.map.centerlines.items():
            coords = np.array(cl.coordinates)
            if len(coords.shape) == 1:
                coords = coords.reshape(1, -1)
            if len(coords) >= 2:
                self._cache['centerlines'][cl_id] = {
                    'x': coords[:, 0].tolist(),
                    'y': coords[:, 1].tolist()
                }

    def draw_map(self, center_x=None, center_y=None, zoom=1.0,
                 show_lanes=True, show_centerlines=True,
                 highlight_point=None, highlight_radius=100,
                 start_point=None, end_point=None, path_coords=None):
        """使用 Plotly 绘制地图（快速）"""

        fig = go.Figure()

        # 计算视图范围
        if center_x is not None and center_y is not None:
            half_size = 500 / zoom
            x_range = (center_x - half_size, center_x + half_size)
            y_range = (center_y - half_size, center_y + half_size)
        else:
            all_x, all_y = [], []
            for lane in self._cache['lanes'].values():
                all_x.extend(lane['x'])
                all_y.extend(lane['y'])
            if all_x:
                margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.1
                x_range = (min(all_x) - margin, max(all_x) + margin)
                y_range = (min(all_y) - margin, max(all_y) + margin)
            else:
                x_range, y_range = (0, 100), (0, 100)

        # 裁剪函数
        def clip_to_view(x_list, y_list):
            x_arr, y_arr = np.array(x_list), np.array(y_list)
            mask = (x_arr >= x_range[0]) & (x_arr <= x_range[1]) & \
                   (y_arr >= y_range[0]) & (y_arr <= y_range[1])
            if not mask.any():
                return None, None
            segments_x, segments_y = [], []
            current_x, current_y = [], []
            for i, m in enumerate(mask):
                if m:
                    current_x.append(x_list[i])
                    current_y.append(y_list[i])
                else:
                    if len(current_x) >= 2:
                        segments_x.append(current_x)
                        segments_y.append(current_y)
                    current_x, current_y = [], []
            if len(current_x) >= 2:
                segments_x.append(current_x)
                segments_y.append(current_y)
            return segments_x, segments_y

        # 绘制车道线
        if show_lanes:
            for lane_id, lane_data in self._cache['lanes'].items():
                segs_x, segs_y = clip_to_view(lane_data['x'], lane_data['y'])
                if segs_x:
                    for sx, sy in zip(segs_x, segs_y):
                        # 每个车道线段都添加悬停信息和可点击性
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy,
                            mode='lines',
                            line=dict(color=lane_data['color'], width=3),
                            showlegend=False,
                            hovertemplate='<b>点击选择此位置</b><br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>',
                            name=f'lane_{lane_id}'
                        ))

        # 绘制中心线
        if show_centerlines:
            for cl_id, cl_data in self._cache['centerlines'].items():
                segs_x, segs_y = clip_to_view(cl_data['x'], cl_data['y'])
                if segs_x:
                    for sx, sy in zip(segs_x, segs_y):
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy,
                            mode='lines',
                            line=dict(color='blue', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        # 高亮点
        if highlight_point:
            theta = np.linspace(0, 2*np.pi, 50)
            circle_x = highlight_point[0] + highlight_radius * np.cos(theta)
            circle_y = highlight_point[1] + highlight_radius * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=circle_x, y=circle_y,
                mode='lines',
                fill='toself',
                fillcolor='rgba(0,255,0,0.2)',
                line=dict(color='green', width=2),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[highlight_point[0]], y=[highlight_point[1]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                showlegend=False
            ))

        # 可点击的背景区域 - 使用密集网格点覆盖整个视图
        # 创建密集网格，让用户可以点击任意位置
        grid_count = 15  # 每行/列的点数
        grid_x = np.linspace(x_range[0], x_range[1], grid_count)
        grid_y = np.linspace(y_range[0], y_range[1], grid_count)
        click_points_x = []
        click_points_y = []
        for gx in grid_x:
            for gy in grid_y:
                click_points_x.append(gx)
                click_points_y.append(gy)

        fig.add_trace(go.Scatter(
            x=click_points_x,
            y=click_points_y,
            mode='markers',
            marker=dict(size=10, color='rgba(150,150,150,0.05)', symbol='circle'),
            hovertemplate='点击选择: x=%{x:.1f}, y=%{y:.1f}<extra></extra>',
            showlegend=False,
            name='click_grid',
            hoverinfo='x+y'
        ))

        # 起点标记（绿色）
        if start_point:
            fig.add_trace(go.Scatter(
                x=[start_point[0]], y=[start_point[1]],
                mode='markers+text',
                marker=dict(size=18, color='green', symbol='star'),
                text=['起点'],
                textposition='top center',
                textfont=dict(size=12, color='green'),
                showlegend=False,
                name='start'
            ))

        # 终点标记（红色）
        if end_point:
            fig.add_trace(go.Scatter(
                x=[end_point[0]], y=[end_point[1]],
                mode='markers+text',
                marker=dict(size=18, color='red', symbol='star'),
                text=['终点'],
                textposition='top center',
                textfont=dict(size=12, color='red'),
                showlegend=False,
                name='end'
            ))

        # 路径轨迹（蓝色粗线）
        if path_coords and len(path_coords) >= 2:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in path_coords],
                y=[p[1] for p in path_coords],
                mode='lines',
                line=dict(color='#1E90FF', width=5),
                showlegend=False,
                name='path'
            ))

        fig.update_layout(
            xaxis=dict(range=x_range, scaleratio=1),
            yaxis=dict(range=y_range, scaleratio=1),
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=30, b=0),
            height=600,
            dragmode='pan',
        )
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')

        return fig


class MapAgentUI:
    """MapAgent Web UI"""

    def __init__(self):
        self.map_api = None
        self.agent = None
        self.visualizer = None
        self.current_position = None
        self.start_position = None  # 起点
        self.end_position = None    # 终点
        self.current_path = None    # 当前规划路径
        # 预初始化 Agent
        self._pre_init_agent()

    def _pre_init_agent(self):
        """预初始化 Agent，避免首次聊天时的延迟"""
        try:
            self.init_agent(settings.llm_provider, settings.llm_api_key)
            self._last_api_key = settings.llm_api_key
            print("Agent 预初始化完成")
        except Exception as e:
            print(f"Agent 预初始化失败: {e}")

    def init_map(self, map_file: str):
        self.map_api = MapAPI(map_file=map_file)
        self.visualizer = FastMapVisualizer(self.map_api)

    def init_agent(self, provider: str, api_key: str = None):
        if api_key:
            if provider == "deepseek":
                os.environ["DEEPSEEK_API_KEY"] = api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key

        self.agent = create_master_agent(
            map_file=str(settings.map_path),
            llm_provider=provider,
            api_key=api_key
        )

    def render_map(self, center_x, center_y, zoom, show_lanes, show_centerlines):
        if not self.visualizer:
            fig = go.Figure()
            fig.add_annotation(text="加载中...", x=0.5, y=0.5, showarrow=False)
            return fig

        center_x = None if center_x == "" or center_x is None else float(center_x)
        center_y = None if center_y == "" or center_y is None else float(center_y)

        try:
            return self.visualizer.draw_map(
                center_x=center_x,
                center_y=center_y,
                zoom=zoom,
                show_lanes=show_lanes,
                show_centerlines=show_centerlines,
                highlight_point=self.current_position,
                start_point=self.start_position,
                end_point=self.end_position,
                path_coords=self.current_path
            )
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"错误: {e}", x=0.5, y=0.5, showarrow=False)
            return fig

    def handle_map_click_wrapper(self, x: float, y: float, mode: str):
        """处理地图点击事件（从 JavaScript 传递的坐标）"""
        if x is None or y is None:
            return None, None, "点击无效"

        if mode == "设置起点":
            self.start_position = (x, y, 0)
            return x, y, f"起点已设置: ({x:.1f}, {y:.1f})"
        else:
            self.end_position = (x, y, 0)
            return x, y, f"终点已设置: ({x:.1f}, {y:.1f})"

    def clear_positions(self):
        """清空起点终点"""
        self.start_position = None
        self.end_position = None
        self.current_path = None
        return None, None, "起点终点已清空"

    def set_position(self, x: float, y: float):
        self.current_position = (x, y, 0)
        return f"已设置位置: ({x:.1f}, {y:.1f})"

    def chat(self, message: str, provider: str, api_key: str, history: list):
        if not message.strip():
            return history, ""

        if not self.agent or (api_key and api_key != getattr(self, '_last_api_key', None)):
            self.init_agent(provider, api_key)
            self._last_api_key = api_key

        history = history or []
        history.append({"role": "user", "content": message})

        try:
            # 构建上下文，传递起点终点
            context = {}
            if self.current_position:
                context['location'] = self.current_position
                context['radius'] = 100
            if self.start_position:
                context['origin'] = self.start_position
            if self.end_position:
                context['destination'] = self.end_position

            response = self.agent.chat(message, **context)

            # 检查是否需要更新路径轨迹
            # 如果对话中返回了路径信息，尝试解析并更新地图
            if self.start_position and self.end_position:
                # 尝试从 Agent 内部获取路径结果
                try:
                    from agents.path import PathAgent
                    from agents.base import AgentContext
                    path_agent = PathAgent(AgentContext(map_api=self.map_api))
                    path_result = path_agent.process(
                        "",
                        origin=self.start_position,
                        destination=self.end_position
                    )
                    if path_result and path_result.get('best_path'):
                        waypoints = path_result['best_path'].get('waypoints', [])
                        if waypoints:
                            self.current_path = [tuple(w) for w in waypoints]
                except Exception as e:
                    print(f"路径规划失败: {e}")

            history.append({"role": "assistant", "content": response})
        except Exception as e:
            history.append({"role": "assistant", "content": f"错误: {str(e)}"})

        return history, ""

    def clear_history(self):
        if self.agent:
            self.agent.clear_history()
        return [], "对话历史已清空"

    def get_map_stats(self):
        if not self.map_api:
            return "地图未加载"
        summary = self.map_api.get_map_summary()
        return f"""地图统计:
- 车道线: {summary['total_lanes']} 条
- 中心线: {summary['total_centerlines']} 条
- 交通标志: {summary['total_traffic_signs']} 个"""


def create_ui():
    """创建 UI"""
    ui = MapAgentUI()

    # CSS 样式 - 默认普通光标，选点模式时十字光标
    css = """
    /* 默认地图光标 */
    .js-plotly-plot { cursor: grab !important; }
    .js-plotly-plot * { cursor: grab !important; }

    /* 选点模式时十字光标 */
    body.mode-start .js-plotly-plot,
    body.mode-start .js-plotly-plot *,
    body.mode-end .js-plotly-plot,
    body.mode-end .js-plotly-plot * {
        cursor: crosshair !important;
    }

    /* 按钮激活状态 */
    body.mode-start #set-start-btn button {
        background: #22c55e !important;
        color: white !important;
        box-shadow: 0 0 10px #22c55e;
    }
    body.mode-end #set-end-btn button {
        background: #ef4444 !important;
        color: white !important;
        box-shadow: 0 0 10px #ef4444;
    }
    """

    with gr.Blocks(title="MapAgent", css=css) as demo:
        gr.Markdown("# 🗺️ MapAgent - 地图问答系统")

        # 隐藏的输入框接收点击坐标和模式
        click_coords = gr.Textbox(elem_id="click-coords", visible=False)
        click_mode_state = gr.Textbox(value="", elem_id="click-mode", visible=False)

        # JavaScript 用于处理选点模式
        js_html = """
        <script>
        (function() {
            var mode = "";  // "", "start", "end"

            // 设置选点模式
            window.setSelectMode = function(m) {
                console.log('setSelectMode:', m);
                mode = m;

                // 更新 body 类
                document.body.classList.remove('mode-start', 'mode-end');
                if (m === 'start') {
                    document.body.classList.add('mode-start');
                } else if (m === 'end') {
                    document.body.classList.add('mode-end');
                }
            };

            // 退出选点模式
            window.exitSelectMode = function() {
                mode = "";
                document.body.classList.remove('mode-start', 'mode-end');
            };

            // 绑定按钮点击
            function bindButtons() {
                var startBtn = document.querySelector('#set-start-btn button');
                var endBtn = document.querySelector('#set-end-btn button');

                if (startBtn && !startBtn._bound) {
                    startBtn.addEventListener('click', function() {
                        window.setSelectMode('start');
                    });
                    startBtn._bound = true;
                }
                if (endBtn && !endBtn._bound) {
                    endBtn.addEventListener('click', function() {
                        window.setSelectMode('end');
                    });
                    endBtn._bound = true;
                }
            }

            // 绑定 Plotly 点击
            function bindClick() {
                var plots = document.querySelectorAll('.js-plotly-plot');
                plots.forEach(function(p) {
                    if (p._bound) return;
                    try {
                        p.on('plotly_click', function(e) {
                            console.log('Plot clicked, mode:', mode);
                            if (!mode) return;

                            var pt = e.points[0];
                            var x = pt.x.toFixed(2);
                            var y = pt.y.toFixed(2);

                            var c = document.querySelector('#click-coords textarea');
                            var m = document.querySelector('#click-mode textarea');

                            if (c) {
                                c.value = x + ',' + y;
                                c.dispatchEvent(new Event('input', {bubbles:true}));
                            }
                            if (m) {
                                m.value = mode === 'start' ? '设置起点' : '设置终点';
                                m.dispatchEvent(new Event('input', {bubbles:true}));
                            }

                            window.exitSelectMode();
                        });
                        p._bound = true;
                    } catch(err) {}
                });
            }

            // 初始化
            function init() {
                bindButtons();
                bindClick();
            }

            setTimeout(init, 500);
            setTimeout(init, 1500);
            setInterval(init, 1000);
        })();
        </script>
        """
        gr.HTML(js_html)

        with gr.Row():
            # 左侧：地图
            with gr.Column(scale=2):
                gr.Markdown("### 地图可视化")

                with gr.Row():
                    center_x = gr.Textbox(label="中心X", placeholder="可选")
                    center_y = gr.Textbox(label="中心Y", placeholder="可选")
                    zoom = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="缩放")

                with gr.Row():
                    show_lanes = gr.Checkbox(value=True, label="车道线")
                    show_centerlines = gr.Checkbox(value=False, label="中心线")

                map_plot = gr.Plot(label="地图")

                # 选点按钮
                with gr.Row():
                    set_start_btn = gr.Button("🟢 设置起点", variant="secondary", elem_id="set-start-btn")
                    set_end_btn = gr.Button("🔴 设置终点", variant="secondary", elem_id="set-end-btn")
                    clear_pos_btn = gr.Button("🗑️ 清空", size="sm")

                with gr.Row():
                    start_x = gr.Number(label="起点X", interactive=False, value=None)
                    start_y = gr.Number(label="起点Y", interactive=False, value=None)
                    end_x = gr.Number(label="终点X", interactive=False, value=None)
                    end_y = gr.Number(label="终点Y", interactive=False, value=None)

                pos_status = gr.Textbox(label="状态", interactive=False, lines=1)

                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新地图")
                    stats_btn = gr.Button("📊 统计")
                stats_output = gr.Textbox(label="统计", lines=4)

            # 右侧：对话
            with gr.Column(scale=3):
                gr.Markdown("### 智能对话")

                with gr.Row():
                    provider = gr.Dropdown(
                        choices=["deepseek", "anthropic", "openai"],
                        value=settings.llm_provider,
                        label="LLM 提供商"
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="可选"
                    )

                chatbot = gr.Chatbot(label="对话", height=500)
                user_input = gr.Textbox(label="输入问题", placeholder="例如：这个地图里有什么？", lines=2)

                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")

        # 事件
        def update_map(cx, cy, z, sl, sc):
            if not ui.map_api:
                map_path = Path("/data/DC/MapAgent/data/vector_map.json")
                ui.init_map(str(map_path))

            try:
                return ui.render_map(cx, cy, z, sl, sc)
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"错误: {e}", x=0.5, y=0.5, showarrow=False)
                return fig

        def handle_click(coords_str, mode):
            """处理从 JavaScript 传来的点击坐标"""
            if not coords_str or not mode:
                return gr.update(), gr.update(), gr.update(), gr.update(), "请先点击按钮选择模式"
            try:
                parts = coords_str.split(',')
                x = float(parts[0])
                y = float(parts[1])

                if mode == "设置起点":
                    ui.start_position = (x, y, 0)
                    return x, y, gr.update(), gr.update(), f"起点已设置: ({x:.1f}, {y:.1f})"
                elif mode == "设置终点":
                    ui.end_position = (x, y, 0)
                    return gr.update(), gr.update(), x, y, f"终点已设置: ({x:.1f}, {y:.1f})"
            except Exception as e:
                return gr.update(), gr.update(), gr.update(), gr.update(), f"错误: {e}"

        def clear_all_positions():
            """清空起点终点"""
            ui.start_position = None
            ui.end_position = None
            ui.current_path = None
            return None, None, None, None, "起点终点已清空"

        refresh_btn.click(
            fn=update_map,
            inputs=[center_x, center_y, zoom, show_lanes, show_centerlines],
            outputs=map_plot
        )

        show_lanes.change(fn=update_map, inputs=[center_x, center_y, zoom, show_lanes, show_centerlines], outputs=map_plot)
        show_centerlines.change(fn=update_map, inputs=[center_x, center_y, zoom, show_lanes, show_centerlines], outputs=map_plot)

        # 设置起点按钮
        set_start_btn.click(
            fn=lambda: "请在地图上点击选择起点",
            outputs=pos_status
        )

        # 设置终点按钮
        set_end_btn.click(
            fn=lambda: "请在地图上点击选择终点",
            outputs=pos_status
        )

        # 地图点击事件 - 通过隐藏输入框接收坐标
        click_coords.change(
            fn=handle_click,
            inputs=[click_coords, click_mode_state],
            outputs=[start_x, start_y, end_x, end_y, pos_status]
        ).then(
            fn=update_map,
            inputs=[center_x, center_y, zoom, show_lanes, show_centerlines],
            outputs=map_plot
        )

        # 清空起点终点按钮
        clear_pos_btn.click(
            fn=clear_all_positions,
            outputs=[start_x, start_y, end_x, end_y, pos_status]
        ).then(
            fn=update_map,
            inputs=[center_x, center_y, zoom, show_lanes, show_centerlines],
            outputs=map_plot
        )

        # 发送消息后更新地图（显示路径）
        send_btn.click(
            fn=ui.chat,
            inputs=[user_input, provider, api_key, chatbot],
            outputs=[chatbot, user_input]
        ).then(
            fn=update_map,
            inputs=[center_x, center_y, zoom, show_lanes, show_centerlines],
            outputs=map_plot
        )

        user_input.submit(
            fn=ui.chat,
            inputs=[user_input, provider, api_key, chatbot],
            outputs=[chatbot, user_input]
        ).then(
            fn=update_map,
            inputs=[center_x, center_y, zoom, show_lanes, show_centerlines],
            outputs=map_plot
        )

        clear_btn.click(fn=ui.clear_history, outputs=[chatbot, pos_status])
        stats_btn.click(fn=ui.get_map_stats, outputs=stats_output)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)