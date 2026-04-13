"""
MapAgent Web Server - 基于 Flask 的高性能前端渲染服务

使用纯前端 Canvas 渲染，后端仅提供数据 API
"""

import sys
import os
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
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

# 点云数据缓存
_pointcloud_cache = {}

# 全局 DetectionLoader 实例（用于复用 ego2global 变换矩阵）
_detection_loader: Optional['DetectionLoader'] = None


def _get_detection_loader() -> Optional['DetectionLoader']:
    """懒加载 DetectionLoader 实例"""
    global _detection_loader

    if _detection_loader is None:
        try:
            from utils.detection_loader import DetectionLoader

            # 尝试加载 result_all_V1 格式的检测结果
            result_path = project_root / "data" / "00" / "annotations" / "result_all_V1"
            if result_path.exists():
                _detection_loader = DetectionLoader(str(result_path), enable_tracking=False)
        except Exception as e:
            print(f"[WARNING] 无法加载 DetectionLoader: {e}")

    return _detection_loader

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


@app.route('/api/pointcloud/<int:frame_id>')
def get_pointcloud(frame_id):
    """获取指定帧的点云数据"""
    try:
        # 查询参数
        use_perception = request.args.get('use_perception', 'true').lower() == 'true'
        density = int(request.args.get('density', 30))  # 默认降低密度到 30% 以提高性能
        use_semantic = request.args.get('use_semantic', 'true').lower() == 'true'
        use_voxel = request.args.get('use_voxel', 'true').lower() == 'true'  # 是否使用网格下采样

        # 从 DetectionLoader 获取 ego2global 变换矩阵（复用加载器的缓存）
        loader = _get_detection_loader()
        transform = None
        if loader:
            transform = loader.get_ego_transform(frame_id)

        # 转换为 numpy 数组
        if transform:
            transform_matrix = np.array(transform, dtype=np.float64)
        else:
            # 如果没有加载器或找不到变换，使用单位矩阵
            transform_matrix = np.eye(4)

        # 构建路径
        if use_perception:
            pcd_path = project_root / "data" / "00" / "lidar" / "perception" / f"{frame_id:06d}.pcd"
        else:
            pcd_path = project_root / "data" / "00" / "lidar" / "front_lidar" / f"{frame_id:06d}.pcd"

        if not pcd_path.exists():
            # 尝试另一种路径格式
            pcd_path = project_root / "data" / "00" / "lidar" / "perception" / f"{frame_id}.pcd"
            if not pcd_path.exists():
                return jsonify({
                    'success': False,
                    'error': f'点云文件不存在：{frame_id}'
                }), 404

        # 加载语义标注
        semantic_labels = None
        if use_semantic:
            annot_path = project_root / "data" / "00" / "annotations" / "annot_seg_10hz_v4" / f"{frame_id:06d}.json"
            if annot_path.exists():
                with open(annot_path, 'r', encoding='utf-8') as f:
                    semantic_labels = json.load(f)

        # 读取 PCD 文件并应用变换矩阵
        points = read_pcd_file(str(pcd_path), density, transform_matrix, use_voxel_grid=use_voxel)

        # 如果有语义标注，添加语义信息
        if semantic_labels:
            points = add_semantic_labels_sampled(points, semantic_labels, density)

        return jsonify({
            'success': True,
            'frame_id': frame_id,
            'num_points': len(points),
            'points': points,
            'use_perception': use_perception,
            'use_semantic': use_semantic
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def read_pcd_file(pcd_path, density=100, transform_matrix=None, use_voxel_grid=True):
    """读取 PCD 文件并返回点云数据（使用 Open3D 进行下采样）

    Args:
        pcd_path: PCD 文件路径
        density: 点云密度百分比（1-100）
        transform_matrix: 4x4 齐次变换矩阵（从 LiDAR 坐标系到全局坐标系）
        use_voxel_grid: 是否使用体素下采样
    """
    try:
        import open3d as o3d
        USE_OPEN3D = True
    except ImportError:
        USE_OPEN3D = False
        import struct

    if USE_OPEN3D:
        # 使用 Open3D 读取 PCD 文件
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        # 应用坐标转换（使用 4x4 变换矩阵）
        if transform_matrix is not None and transform_matrix.shape == (4, 4):
            points = np.asarray(pcd.points)

            # 转换为齐次坐标
            ones = np.ones((points.shape[0], 1))
            points_homogeneous = np.hstack([points, ones])  # (N, 4)

            # 应用变换矩阵
            transformed = points_homogeneous @ transform_matrix.T  # (N, 4)

            # 转换回 Cartesian 坐标
            points = transformed[:, :3] / transformed[:, 3:4]

            pcd.points = o3d.utility.Vector3dVector(points)

        # 使用 Open3D 体素下采样
        if use_voxel_grid and density < 100:
            # 根据密度计算体素大小
            # 密度 100% -> 0.15m, 密度 10% -> 1.5m
            voxel_size = 0.15 * (100 / density)
            voxel_size = max(0.15, min(voxel_size, 3.0))
            pcd = pcd.voxel_down_sample(voxel_size)

        # 转换为返回格式
        points = np.asarray(pcd.points)
        result = []
        for p in points:
            result.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])})

        return result
    else:
        # Open3D 不可用，使用纯 Python 实现
        return _read_pcd_file_fallback(pcd_path, density, transform_matrix, use_voxel_grid)


def _read_pcd_file_fallback(pcd_path, density=100, transform_matrix=None, use_voxel_grid=True):
    """纯 Python 读取 PCD 文件（当 Open3D 不可用时）"""
    import struct

    raw_points = []
    with open(pcd_path, 'rb') as f:
        # 读取头部
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('DATA'):
                break
            parts = line.split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1:]

        # 解析点云数据
        num_points = int(header.get('POINTS', ['0'])[0])
        data_format = header.get('DATA', ['binary'])[0]
        fields = header.get('FIELDS', [])
        sizes = [int(s) for s in header.get('SIZE', ['4'] * len(fields))]

        # 计算每个点的字节数
        point_size = sum(sizes)

        if data_format == 'binary':
            # 二进制格式：直接读取所有点
            data = f.read(num_points * point_size)

            # 确定 x, y, z 的偏移量
            x_offset = 0
            y_offset = sizes[0] if len(sizes) > 0 else 4
            z_offset = sum(sizes[:2]) if len(sizes) > 1 else 8

            for i in range(num_points):
                idx = i * point_size
                x = struct.unpack('f', data[idx:idx + 4])[0]
                y = struct.unpack('f', data[idx + y_offset:idx + y_offset + 4])[0]
                z = struct.unpack('f', data[idx + z_offset:idx + z_offset + 4])[0]

                if transform_matrix is not None and transform_matrix.shape == (4, 4):
                    # 使用 4x4 变换矩阵进行坐标转换
                    # 齐次坐标变换
                    x_new = transform_matrix[0, 0] * x + transform_matrix[0, 1] * y + transform_matrix[0, 2] * z + transform_matrix[0, 3]
                    y_new = transform_matrix[1, 0] * x + transform_matrix[1, 1] * y + transform_matrix[1, 2] * z + transform_matrix[1, 3]
                    z_new = transform_matrix[2, 0] * x + transform_matrix[2, 1] * y + transform_matrix[2, 2] * z + transform_matrix[2, 3]
                    w = transform_matrix[3, 0] * x + transform_matrix[3, 1] * y + transform_matrix[3, 2] * z + transform_matrix[3, 3]

                    if w != 0:
                        x_new /= w
                        y_new /= w
                        z_new /= w

                    raw_points.append((x_new, y_new, z_new))
                else:
                    raw_points.append((x, y, z))

    # 简单降采样
    if use_voxel_grid:
        return voxel_grid_downsample(raw_points, density)
    else:
        step = max(1, 100 // density) if density < 100 else 1
        return [{'x': p[0], 'y': p[1], 'z': p[2]} for i, p in enumerate(raw_points) if i % step == 0]


def voxel_grid_downsample(points, density=100):
    """
    网格下采样（Voxel Grid Downsampling）

    将点云空间划分为体素网格，每个体素内的点用一个点表示（中心点或重心）

    Args:
        points: 点列表，每个点为 (x, y, z) 元组
        density: 密度百分比 (1-100)，值越小下采样越强烈

    Returns:
        下采样后的点列表，格式为 [{'x': x, 'y': y, 'z': z}, ...]
    """
    if not points or density <= 0:
        return []

    if density >= 100:
        # 不需要下采样
        return [{'x': p[0], 'y': p[1], 'z': p[2]} for p in points]

    # 根据密度计算体素大小
    # 密度 100% -> 体素大小为 0.15m
    # 密度 50% -> 体素大小为 0.30m
    # 密度 10% -> 体素大小为 1.5m
    voxel_size = 0.15 * (100 / density)
    voxel_size = max(0.15, min(voxel_size, 3.0))  # 限制在 0.15-3.0 米之间

    # 构建体素网格 - 使用重心法
    voxel_sums = {}  # voxel_key -> [sum_x, sum_y, sum_z, count]

    for p in points:
        x, y, z = p
        # 计算体素索引
        voxel_key = (
            int(x / voxel_size) if x >= 0 else int(x / voxel_size) - 1,
            int(y / voxel_size) if y >= 0 else int(y / voxel_size) - 1
        )

        if voxel_key not in voxel_sums:
            voxel_sums[voxel_key] = [0.0, 0.0, 0.0, 0]

        voxel_sums[voxel_key][0] += x
        voxel_sums[voxel_key][1] += y
        voxel_sums[voxel_key][2] += z
        voxel_sums[voxel_key][3] += 1

    # 计算每个体素的重心
    downsampled = []
    for (vx, vy), (sum_x, sum_y, sum_z, count) in voxel_sums.items():
        if count > 0:
            downsampled.append({
                'x': sum_x / count,
                'y': sum_y / count,
                'z': sum_z / count
            })

    return downsampled


def add_semantic_labels(points, semantic_labels):
    """为点云添加语义标签"""
    # 语义类型到颜色的映射
    semantic_colors = {
        'unlabeled': [128, 128, 128],  # 灰色
        'ground': [0, 255, 0],  # 绿色
        'traffic_sign': [255, 255, 0],  # 黄色
        'building': [0, 0, 255],  # 蓝色
        'noise': [128, 0, 128],  # 紫色
        'pedestrian': [255, 0, 0],  # 红色
        'free': [200, 200, 200],  # 浅灰
        'crash_barrel': [255, 128, 0],  # 橙色
        'bus': [0, 128, 255],  # 深蓝
        'truck': [128, 0, 255],  # 紫红
        'non_motorized_vehicle': [0, 255, 128],  # 青绿
        'fence': [128, 128, 0],  # 棕色
        'car': [0, 0, 128],  # 深蓝
        'curbside': [255, 0, 255],  # 粉红
        'vegetation': [255, 165, 0],  # 橙色
        'tree': [34, 139, 34],  # 深绿
    }

    # 构建索引到语义的映射
    index_to_semantic = {}
    for label in semantic_labels.get('labels', []):
        semantic_type = label.get('type', 'unlabeled')
        color = semantic_colors.get(semantic_type, [128, 128, 128])
        for idx in label.get('pointsIndex', []):
            index_to_semantic[idx] = color

    # 为每个点添加颜色
    for i, point in enumerate(points):
        if i in index_to_semantic:
            point['color'] = index_to_semantic[i]
        else:
            point['color'] = [128, 128, 128]  # 默认灰色

    return points


def add_semantic_labels_sampled(points, semantic_labels, density):
    """为降采样后的点云添加语义标签"""
    # 语义类型到颜色的映射
    semantic_colors = {
        'unlabeled': [128, 128, 128],  # 灰色
        'ground': [0, 255, 0],  # 绿色
        'traffic_sign': [255, 255, 0],  # 黄色
        'building': [0, 0, 255],  # 蓝色
        'noise': [128, 0, 128],  # 紫色
        'pedestrian': [255, 0, 0],  # 红色
        'free': [200, 200, 200],  # 浅灰
        'crash_barrel': [255, 128, 0],  # 橙色
        'bus': [0, 128, 255],  # 深蓝
        'truck': [128, 0, 255],  # 紫红
        'non_motorized_vehicle': [0, 255, 128],  # 青绿
        'fence': [128, 128, 0],  # 棕色
        'car': [0, 0, 128],  # 深蓝
        'curbside': [255, 0, 255],  # 粉红
        'vegetation': [255, 165, 0],  # 橙色
        'tree': [34, 139, 34],  # 深绿
    }

    # 构建索引到语义的映射（只存储采样的点）
    index_to_semantic = {}
    step = max(1, 100 // density) if density < 100 else 1

    for label in semantic_labels.get('labels', []):
        semantic_type = label.get('type', 'unlabeled')
        color = semantic_colors.get(semantic_type, [128, 128, 128])
        for idx in label.get('pointsIndex', []):
            if idx % step == 0:  # 只保留采样的点
                index_to_semantic[idx // step] = color

    # 为每个点添加颜色
    for i, point in enumerate(points):
        if i in index_to_semantic:
            point['color'] = index_to_semantic[i]
        else:
            point['color'] = [128, 128, 128]  # 默认灰色

    return points


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
                    os.environ["LLM_MODEL"] = "Qwen3_5"
                elif llm_provider == "gemma4":
                    os.environ["GEMMA4_BASE_URL"] = f"http://localhost:{llm_port}/v1"
                    os.environ["LLM_MODEL"] = "Gemma4"

            # 创建 LLMConfig（即使没有 API Key 也尝试创建）
            # 获取模型名
            model_name = os.getenv("LLM_MODEL", "")
            if not model_name:
                default_models = {
                    "deepseek": "deepseek-chat",
                    "anthropic": "claude-sonnet-4-6",
                    "openai": "gpt-4o",
                    "qwen": "Qwen3_5",
                    "gemma4": "Gemma4",
                }
                model_name = default_models.get(llm_provider, "")

            config = LLMConfig(
                provider=provider,
                model=model_name,
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
                        'position': v.get('position', [0, 0, 0])[:2],
                        'frame_id': frame.get('frame_id', 0),  # 添加帧 ID 用于计算运动方向
                        'heading': v.get('heading', 0.0),  # 使用优化后的朝向
                        'velocity': v.get('velocity', [0, 0, 0])[:2],
                        'speed': v.get('speed', 0.0),
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
                        'position': state.get('position', [0, 0, 0])[:2],
                        'heading': state.get('heading', 0.0),  # 使用优化后的朝向
                        'velocity': state.get('velocity', [0, 0])[:2],
                        'speed': state.get('speed', 0.0),
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
                os.environ["LLM_MODEL"] = "Qwen3_5"
                api_key = "dummy"
            elif provider == "gemma4":
                os.environ["GEMMA4_BASE_URL"] = f"http://localhost:{port}/v1"
                os.environ["LLM_PROVIDER"] = "gemma4_local"
                os.environ["LLM_MODEL"] = "Gemma4"
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
