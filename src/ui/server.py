"""
MapAgent Web Server - 基于 Flask 的高性能前端渲染服务

使用纯前端 Canvas 渲染，后端仅提供数据 API
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import json

from flask import Flask, render_template, jsonify, request, Response, stream_with_context

# CORS 支持（如果可用）
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from apis.map_api import MapAPI
from agents.traffic_flow import TrafficFlowAgent
from agents.base import AgentContext
from config import settings


app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}
if CORS_AVAILABLE:
    CORS(app)

# 禁用静态文件缓存
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# 全局状态
_map_api = None
_map_data_cache = None


def get_map_api():
    """懒加载地图 API"""
    global _map_api, _map_data_cache

    if _map_api is None:
        _map_api = MapAPI(map_file=str(settings.map_path))

        # 预构建地图数据缓存（用于前端）
        _map_data_cache = {
            'lanes': [],
            'centerlines': []
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
            'no_lane': '#CCCCCC',
        }

        # 缓存车道
        for lane_id, lane in _map_api.map.lane_lines.items():
            coords = lane.coordinates
            if len(coords) >= 2:
                _map_data_cache['lanes'].append({
                    'id': lane_id,
                    'x': [c[0] for c in coords],
                    'y': [c[1] for c in coords],
                    'color': lane_colors.get(lane.type, '#CCCCCC'),
                    'type': lane.type  # 添加类型信息
                })

        # 缓存中心线
        for cl_id, cl in _map_api.map.centerlines.items():
            coords = cl.coordinates
            if len(coords) >= 2:
                _map_data_cache['centerlines'].append({
                    'id': cl_id,
                    'x': [c[0] for c in coords],
                    'y': [c[1] for c in coords]
                })

    return _map_api


@app.route('/')
def index():
    """主页"""
    # 禁用模板缓存
    from flask import make_response
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response


@app.route('/api/map_data')
def get_map_data():
    """获取地图数据"""
    get_map_api()
    return jsonify(_map_data_cache)


@app.route('/api/reconstruct', methods=['POST'])
def reconstruct_traffic_flow():
    """重建交通流"""
    global _llm_reasoning_cache
    data = request.json
    detection_path = data.get('path', 'data/json_results')
    start_frame = data.get('start_frame', -1)
    end_frame = data.get('end_frame', -1)
    use_llm = data.get('use_llm', False)
    llm_provider = data.get('llm_provider', 'deepseek')
    llm_api_key = data.get('llm_api_key', '')
    llm_port = data.get('llm_port', 8000)

    # 清空之前的推理记录
    _llm_reasoning_cache = []

    # 调试输出
    print(f"[DEBUG] use_llm={use_llm}, llm_provider={llm_provider}, has_api_key={bool(llm_api_key)}")

    try:
        # 创建 Agent
        map_api = get_map_api()

        # 调试输出
        print(f"[DEBUG] map_api is None: {map_api is None}")

        # LLM 进度回调
        def llm_progress_callback(event_type: str, data: dict):
            global _llm_reasoning_cache
            # 确保错误信息可序列化
            if 'error' in data and data['error']:
                # 只保留错误信息的第一行，避免过长
                error_msg = str(data['error']).split('\n')[0][:200]
                data = dict(data)  # copy to avoid modifying original
                data['error'] = error_msg
            event = {
                'event_type': event_type,
                'data': data,
                'timestamp': __import__('time').time()
            }
            _llm_reasoning_cache.append(event)

        # 如果使用 LLM，需要初始化 LLMClient
        llm_client = None
        if use_llm:
            from core.llm_client import LLMClient, LLMConfig, LLMProvider
            import os

            # 映射前端提供商名称到 LLMProvider
            provider_map = {
                "deepseek": LLMProvider.DEEPSEEK,
                "anthropic": LLMProvider.ANTHROPIC,
                "openai": LLMProvider.OPENAI,
                "qwen": LLMProvider.QWEN_LOCAL,
                "gemma4": LLMProvider.GEMMA4_LOCAL,
            }
            provider = provider_map.get(llm_provider, LLMProvider.DEEPSEEK)

            # 设置环境变量
            if llm_api_key:
                if llm_provider == "deepseek":
                    os.environ["DEEPSEEK_API_KEY"] = llm_api_key
                elif llm_provider == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = llm_api_key
                elif llm_provider == "openai":
                    os.environ["OPENAI_API_KEY"] = llm_api_key

            # 本地模型端口
            if llm_provider in ["qwen", "gemma4"]:
                os.environ["LOCAL_LLM_PORT"] = str(llm_port)
                if llm_provider == "qwen":
                    os.environ["QWEN_BASE_URL"] = f"http://localhost:{llm_port}/v1"
                elif llm_provider == "gemma4":
                    os.environ["GEMMA4_BASE_URL"] = f"http://localhost:{llm_port}/v1"

            # 创建 LLMConfig（即使没有 API Key 也尝试创建）
            config = LLMConfig(
                provider=provider,
                api_key=llm_api_key or os.getenv(f"{llm_provider.upper()}_API_KEY") or "dummy",
            )
            # 设置 DeepSeek 的 base_url 和模型
            if llm_provider == "deepseek":
                config.base_url = "https://api.deepseek.com"
                config.model = "deepseek-chat"  # 使用有效的模型名称

            llm_client = LLMClient(config)
            # 显示 API Key 前缀用于验证
            key_prefix = config.api_key[:10] + "..." if config.api_key and len(config.api_key) > 10 else (config.api_key or "EMPTY")
            print(f"[DEBUG] LLM Client created: provider={provider}, model={config.model}, api_key={key_prefix}, base_url={config.base_url}")

        context = AgentContext(map_api=map_api, llm_client=llm_client)
        tf_agent = TrafficFlowAgent(context, use_llm=use_llm)

        # 调试输出：Agent 状态
        print(f"[DEBUG] TrafficFlowAgent created: use_llm={tf_agent._use_llm}, llm_optimizer={tf_agent._llm_optimizer is not None}, map_api={tf_agent.map_api is not None}")

        # 设置 LLM 进度回调
        if use_llm:
            tf_agent.set_llm_progress_callback(llm_progress_callback)

        # 执行重建
        result = tf_agent.process(
            query=f"基于检测结果重建交通流，检测结果位于 {detection_path}",
            detection_path=detection_path,
            start_frame=start_frame if start_frame >= 0 else None,
            end_frame=end_frame if end_frame >= 0 else None,
            output_path="reconstruction_result.json"
        )

        if result.get('success'):
            # 转换为前端格式
            frames = result.get('frames', [])
            trajectories = result.get('trajectories', [])

            # 简化数据，减少传输量
            simplified_frames = []
            for frame in frames:
                simplified_frame = {
                    'frame_id': frame.get('frame_id', 0),
                    'vehicles': []
                }
                for v in frame.get('vehicles', []):
                    simplified_frame['vehicles'].append({
                        'vehicle_id': v.get('vehicle_id', 0),
                        'vehicle_type': v.get('vehicle_type', 'Unknown'),
                        'position': v.get('position', [0, 0, 0])[:2],  # 只传输 x, y
                    })
                simplified_frames.append(simplified_frame)

            # 简化轨迹数据
            simplified_trajectories = []
            for traj in trajectories:
                simplified_traj = {
                    'vehicle_id': traj.get('vehicle_id', 0),
                    'vehicle_type': traj.get('vehicle_type', 'Unknown'),
                    'states': []
                }
                for state in traj.get('states', []):
                    simplified_traj['states'].append({
                        'frame_id': state.get('frame_id', 0),
                        'position': state.get('position', [0, 0, 0])[:2]
                    })
                simplified_trajectories.append(simplified_traj)

            return jsonify({
                'success': True,
                'frames': simplified_frames,
                'trajectories': simplified_trajectories,
                'total_frames': len(simplified_frames),
                'total_vehicles': result.get('total_vehicles', 0),
                'statistics': result.get('statistics', {})
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '重建失败')
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    map_api = get_map_api()
    summary = map_api.get_map_summary()
    return jsonify(summary)


@app.route('/api/find_path', methods=['POST'])
def find_path():
    """路径规划"""
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({
            'success': False,
            'error': '缺少起点或终点参数'
        }), 400

    try:
        from agents.path import PathAgent
        from agents.base import AgentContext

        map_api = get_map_api()
        context = AgentContext(map_api=map_api)
        path_agent = PathAgent(context)

        # 执行路径规划
        result = path_agent.process(
            "",
            origin=tuple(origin),
            destination=tuple(destination)
        )

        if result and result.get('best_path'):
            best_path = result['best_path']
            waypoints = best_path.get('waypoints', [])

            return jsonify({
                'success': True,
                'path': {
                    'waypoints': waypoints,
                    'length': best_path.get('length', 0),
                    'lane_ids': best_path.get('lane_ids', [])
                },
                'all_paths': [
                    {
                        'waypoints': p.get('waypoints', []),
                        'length': p.get('length', 0),
                        'cost': p.get('cost', 0)
                    }
                    for p in result.get('all_paths', [])[:3]  # 只返回前 3 条
                ]
            })
        else:
            return jsonify({
                'success': False,
                'error': '未找到可行路径'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 全局聊天历史
_chat_history = []
_last_llm_config = {}
# LLM 推理过程记录
_llm_reasoning_cache = []


@app.route('/api/llm_progress')
def llm_progress_stream():
    """SSE 推送 LLM 推理过程"""
    def generate():
        global _llm_reasoning_cache
        # 先发送历史记录
        for event in _llm_reasoning_cache[-50:]:  # 最多发送最近 50 条
            yield f"data: {json.dumps(event)}\n\n"
        # 保持连接
        while True:
            import time
            time.sleep(0.5)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'})


@app.route('/api/chat', methods=['POST'])
def chat():
    """LLM 聊天接口"""
    global _chat_history, _last_llm_config

    data = request.json
    message = data.get('message', '')
    provider = data.get('provider', 'deepseek')
    api_key = data.get('api_key', '')
    port = data.get('port', 8000)
    context = data.get('context', {})

    if not message.strip():
        return jsonify({
            'success': False,
            'error': '消息不能为空'
        }), 400

    try:
        # 检查是否需要重新初始化 Agent
        need_reinit = (
            provider != _last_llm_config.get('provider') or
            api_key != _last_llm_config.get('api_key') or
            port != _last_llm_config.get('port')
        )

        if need_reinit or not hasattr(chat, 'agent') or chat.agent is None:
            from agents.master import create_master_agent

            # 设置环境变量
            if api_key:
                if provider == "deepseek":
                    os.environ["DEEPSEEK_API_KEY"] = api_key
                elif provider == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                elif provider == "openai":
                    os.environ["OPENAI_API_KEY"] = api_key

            # 本地模型配置
            if provider == "qwen":
                os.environ["QWEN_BASE_URL"] = f"http://localhost:{port}/v1"
                os.environ["LLM_PROVIDER"] = "qwen_local"
                api_key = "dummy"
            elif provider == "gemma4":
                os.environ["GEMMA4_BASE_URL"] = f"http://localhost:{port}/v1"
                os.environ["LLM_PROVIDER"] = "gemma4_local"
                api_key = "dummy"

            chat.agent = create_master_agent(
                map_file=str(settings.map_path),
                llm_provider=provider,
                api_key=api_key
            )
            _last_llm_config = {'provider': provider, 'api_key': api_key, 'port': port}

        # 构建上下文
        chat_context = {}
        if context.get('start_position'):
            chat_context['origin'] = context['start_position']
        if context.get('end_position'):
            chat_context['destination'] = context['end_position']

        # 聊天
        response = chat.agent.chat(message, **chat_context)

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """主函数"""
    print("=" * 50)
    print("MapAgent Web Server - 前端渲染版")
    print("=" * 50)
    print(f"地图文件：{settings.map_path}")
    print(f"访问地址：http://localhost:7860")
    print(f" threaded: True (支持 SSE 并发)")
    print("=" * 50)

    app.run(host='0.0.0.0', port=7860, debug=True, threaded=True)


if __name__ == '__main__':
    main()

