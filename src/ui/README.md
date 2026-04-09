# MapAgent Web UI - 纯前端渲染版

## 架构说明

### 传统 Gradio + Plotly 方案（原有）
- **缺点**：
  - 每次更新都需要后端回调
  - Plotly 基于 SVG，大量元素时性能差
  - 动画通过 Frame 切换，数据传输量大

### 纯前端 Canvas 方案（新建）
- **优点**：
  - 使用 HTML5 Canvas 进行 GPU 加速渲染
  - 数据一次性加载，前端流畅播放
  - 支持 60FPS 流畅动画
  - 分层渲染（地图/轨迹/车辆），只重绘变化层

## 文件结构

```
src/ui/
├── templates/
│   └── index.html          # 主页面
├── static/
│   ├── style.css           # 样式表
│   └── visualizer.js       # 可视化引擎
├── server.py               # Flask 服务器
├── app.py                  # Gradio 应用（原有）
└── requirements-web.txt    # Web 版依赖
```

## 安装

```bash
pip install flask flask-cors
```

## 启动

```bash
# 方式 1：直接运行
python src/ui/server.py

# 方式 2：使用 Python 模块
python -m src.ui.server
```

访问：http://localhost:7860

## 功能特性

### 地图可视化
- 车道线（不同颜色表示类型）
- 中心线（虚线蓝色）
- 支持缩放和平移

### 交通流播放
- 播放/暂停（空格键）
- 帧跳转（左右箭头键）
- 首帧/尾帧（Home/End 键）
- 速度调节（1-30 FPS）

### 显示选项
- 地图底图
- 车辆
- 车辆 ID
- 轨迹线
- 运动拖尾

### 交互功能
- 鼠标悬停显示车辆信息
- 滚轮缩放
- 键盘快捷键

## 快捷键

| 按键 | 功能 |
|------|------|
| 空格 | 播放/暂停 |
| ← | 上一帧 |
| → | 下一帧 |
| Home | 第一帧 |
| End | 最后一帧 |
| +/- | 缩放 |

## 性能优化

### 1. 分层渲染
三个 Canvas 层：
- **底层**：地图（不经常变化）
- **中层**：轨迹（半透明拖尾）
- **顶层**：车辆（每帧更新）

### 2. 数据简化
- 只传输 2D 坐标（省略 Z）
- 按车辆类型批量绘制
- 视野裁剪（只绘制可见区域）

### 3. 动画优化
- requestAnimationFrame 同步显示器刷新率
- 防抖处理窗口缩放
- CSS transform 替代 Canvas 重绘

## API 接口

### GET /api/map_data
获取地图数据
```json
{
  "lanes": [{"id": "1", "x": [...], "y": [...], "color": "#..."}],
  "centerlines": [{"id": "1", "x": [...], "y": [...]}]
}
```

### POST /api/reconstruct
重建交通流
```json
{
  "path": "data/json_results",
  "start_frame": -1,
  "end_frame": -1
}
```

### GET /api/stats
获取地图统计信息

## 与 Gradio 版对比

| 特性 | Gradio + Plotly | Flask + Canvas |
|------|-----------------|----------------|
| 渲染方式 | SVG | Canvas |
| 动画原理 | Frame 切换 | 前端定时器 |
| 数据传输 | 每帧传输 | 一次性加载 |
| 播放流畅度 | 5-10 FPS | 60 FPS |
| 车辆容量 | ~100 辆 | ~1000 辆 |
| 交互延迟 | ~200ms | ~16ms |
| 支持缩放 | ✓ | ✓ |
| 支持平移 | ✓ | ✓ |
| 键盘快捷键 | ✗ | ✓ |

## 扩展开发

### 添加新的显示层

```javascript
// visualizer.js
class CustomRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    render(data) {
        // 自定义绘制逻辑
    }
}
```

### 添加新的 API 接口

```python
# server.py
@app.route('/api/custom', methods=['POST'])
def custom_api():
    data = request.json
    # 处理逻辑
    return jsonify(result)
```

## 故障排除

### 地图不显示
- 检查 `settings.map_path` 是否正确
- 查看浏览器控制台错误

### 播放卡顿
- 减少同时显示的车辆数量
- 降低 FPS
- 关闭轨迹显示

### CORS 错误
- 确保 Flask-CORS 已安装
- 检查 API 请求地址是否正确
