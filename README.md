# MapAgent - 多Agent地图问答系统

基于大语言模型的多Agent地图问答系统，能够理解自然语言查询，并基于车道线矢量地图提供智能化的地图相关回答。

## 功能特性

- **场景理解**：分析车道数量、类型、路口结构、交通规则
- **行为分析**：车辆位置匹配、行为预测、风险评估
- **路径规划**：多路径搜索、行程估算、行驶建议
- **多模型支持**：支持 Claude、Deepseek、本地模型
- **Web可视化**：Gradio 实现的交互式地图界面

## 项目结构

```
MapAgent/
├── src/                          # 源码
│   ├── agents/                   # Agent 实现
│   │   ├── base.py              # Agent 基类
│   │   ├── scene.py             # 场景理解 Agent
│   │   ├── behavior.py          # 行为分析 Agent
│   │   ├── path.py              # 路径规划 Agent
│   │   └── master.py            # 主 Agent
│   ├── apis/map_api.py          # 地图查询 API
│   ├── core/                    # 核心模块
│   │   ├── llm_client.py        # LLM 客户端
│   │   └── tools.py             # Function Calling 工具
│   ├── models/                  # 数据模型
│   │   ├── map_data.py          # 地图数据结构
│   │   └── agent_io.py          # Agent 输入输出
│   ├── ui/app.py                # Web UI
│   └── utils/geo.py             # 地理计算工具
├── tests/                        # 测试
├── examples/                     # 使用示例
│   ├── basic_usage.py           # 基础用法
│   └── chat.py                  # 交互式命令行
├── config/                       # 配置
│   └── settings.py              # 配置管理
├── generate_vector_map.py        # 地图生成脚本
├── run_ui.py                     # UI 启动脚本
└── requirements.txt              # 依赖
```

## 安装

### 1. 创建虚拟环境

```bash
conda create -n mapagent python=3.10
conda activate mapagent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备地图数据

将矢量地图文件放置于 `data/vector_map.json`。

可以使用 `generate_vector_map.py` 从原始标注数据生成：

```bash
python generate_vector_map.py \
    --data-dir ./data/00/annotations \
    --output ./data/vector_map.json
```

## 快速开始

### 基础使用

```python
from src.apis.map_api import MapAPI
from src.agents import MasterAgent

# 加载地图
map_api = MapAPI(map_file="data/vector_map.json")

# 创建主 Agent
agent = MasterAgent(map_api=map_api)

# 对话
response = agent.chat("这个路口有几条车道？")
print(response)
```

### 交互式命令行

```bash
python examples/chat.py
```

### Web UI

```bash
python run_ui.py
```

访问 http://localhost:7860 使用 Web 界面。

## API 文档

### MapAPI - 地图查询接口

```python
from src.apis.map_api import MapAPI

api = MapAPI(map_file="data/vector_map.json")
```

#### 基础查询

| 方法 | 描述 |
|------|------|
| `get_lane_info(lane_id)` | 获取车道信息 |
| `get_intersection_info(id)` | 获取路口信息 |
| `get_centerline_info(id)` | 获取中心线信息 |
| `get_connected_lanes(lane_id)` | 获取连接车道 |
| `get_map_summary()` | 获取地图概要 |

#### 空间查询

| 方法 | 描述 |
|------|------|
| `find_nearest_lane(position)` | 查找最近车道 |
| `find_nearest_centerline(position)` | 查找最近中心线 |
| `find_lanes_in_area(center, radius)` | 查找区域内车道 |
| `find_intersections_in_area(center, radius)` | 查找区域内路口 |
| `get_area_statistics(center, radius)` | 获取区域统计 |
| `match_vehicle_to_lane(position, heading)` | 匹配车辆到车道 |

#### 使用示例

```python
# 查询车道信息
lane_info = api.get_lane_info("1")
print(f"车道类型: {lane_info['type']}, 长度: {lane_info['length']}m")

# 查找最近车道
nearest = api.find_nearest_lane((100.0, 200.0, 0))
print(f"最近车道: {nearest['lane_id']}, 距离: {nearest['distance']}m")

# 区域统计
stats = api.get_area_statistics((100.0, 200.0, 0), radius=100)
print(f"区域内车道: {stats['lane_count']}, 路口: {stats['intersection_count']}")
```

### Agent 使用

#### 场景理解 Agent

```python
from src.agents import SceneAgent
from src.agents.base import AgentContext

context = AgentContext(map_api=api)
scene_agent = SceneAgent(context)

# 分析场景
result = scene_agent.process(
    query="这个位置有什么车道？",
    location=(100.0, 200.0, 0),
    radius=100
)
```

#### 行为分析 Agent

```python
from src.agents import BehaviorAgent

behavior_agent = BehaviorAgent(context)

# 预测车辆行为
result = behavior_agent.process(
    query="这辆车会右转吗？",
    location=(100.0, 200.0, 0),
    heading=45.0,
    speed=10.0
)
# 返回: {"predicted_action": "straight", "confidence": 0.8, "risk_level": "low"}
```

#### 路径规划 Agent

```python
from src.agents import PathAgent

path_agent = PathAgent(context)

# 规划路径
result = path_agent.process(
    query="怎么走？",
    origin=(100.0, 200.0, 0),
    destination=(150.0, 250.0, 0)
)
# 返回: {"distance": 1000, "estimated_time": 2.5, "advice": "..."}
```

#### 主 Agent

```python
from src.agents import MasterAgent, create_master_agent

# 方式1: 直接创建
agent = MasterAgent(map_api=api)

# 方式2: 便捷函数
agent = create_master_agent(map_file="data/vector_map.json")

# 对话
response = agent.chat("这个路口有几条车道？")
```

## 数据格式

### 矢量地图 (vector_map.json)

```json
{
  "version": "1.0",
  "type": "static_map",
  "lane_lines": {
    "1": {
      "id": "1",
      "type": "double_solid",
      "color": "yellow",
      "coordinates": [[x, y, z], ...],
      "length": 278.4
    }
  },
  "centerlines": {
    "1": {
      "id": "1",
      "coordinates": [[x, y, z], ...],
      "left_boundary_id": "10",
      "right_boundary_id": "11",
      "predecessor_ids": ["202"],
      "successor_ids": ["203"]
    }
  },
  "intersections": {
    "intersection_0": {
      "id": "intersection_0",
      "center": [x, y, z],
      "lanes": ["lane_1", "lane_2"]
    }
  },
  "traffic_signs": {
    "1": {
      "id": "1",
      "category": "lane_direction",
      "function": {"direction_arrow": "left_turn"}
    }
  }
}
```

## LLM 配置

### 支持的模型

| 提供商 | 模型 | 环境变量 |
|--------|------|----------|
| Anthropic | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| Deepseek | deepseek-reasoner | `DEEPSEEK_API_KEY` |
| 本地模型 | Qwen, Gemma 等 | `LLM_BASE_URL` |

### 配置方式

```bash
# 方式1: 环境变量
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key

# 方式2: 代码配置
from src.core.llm_client import LLMClient, LLMConfig

config = LLMConfig(
    provider="deepseek",
    model="deepseek-reasoner",
    api_key="your-key"
)
client = LLMClient(config)
```

### 本地模型

```python
# 连接本地模型服务
config = LLMConfig(
    provider="local",
    model="Qwen3___5-35B-A3B",
    base_url="http://localhost:8000/v1"
)
```

### 配置文件

在 `config/settings.py` 中可以配置：

```python
class Settings(BaseSettings):
    # 地图配置
    map_file: str = "data/vector_map.json"

    # LLM 配置
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
```

## 意图识别

系统支持以下意图类型：

| 意图 | 关键词示例 | 对应 Agent |
|------|-----------|-----------|
| scene | 车道、路口、几条、道路结构 | SceneAgent |
| behavior | 车辆、行为、预测、碰撞、风险 | BehaviorAgent |
| path | 路线、怎么走、路径、规划 | PathAgent |
| general | 其他问题 | 直接回复 |

### 示例

```python
agent = MasterAgent(map_api=api)

# 场景理解
agent.chat("这个路口有几条车道？")          # → SceneAgent
agent.chat("附近有什么交通标志？")         # → SceneAgent

# 行为分析
agent.chat("这辆车会右转吗？")             # → BehaviorAgent
agent.chat("预测一下这个车辆的行为")        # → BehaviorAgent

# 路径规划
agent.chat("从这到机场怎么走？")           # → PathAgent
agent.chat("帮我规划一条最快的路线")        # → PathAgent
```

## 工具 (Function Calling)

各 Agent 提供的工具：

### SceneAgent
- `get_lane_count_by_type`: 获取区域内车道类型统计
- `get_intersection_structure`: 获取路口结构
- `get_nearby_traffic_signs`: 获取附近交通标志
- `analyze_road_scene`: 分析道路场景

### BehaviorAgent
- `match_vehicle_to_lane`: 匹配车辆到车道
- `predict_vehicle_action`: 预测车辆行为
- `analyze_collision_risk`: 分析碰撞风险
- `get_lane_change_possibility`: 变道可能性分析

### PathAgent
- `find_path`: 查找路径
- `get_route_advice`: 获取路线建议
- `estimate_travel_time`: 估算行程时间
- `find_nearby_destination`: 查找附近目的地

## 运行测试

```bash
# 测试 Map API
python tests/test_map_api.py

# 测试子 Agent
python tests/test_agents.py

# 测试主 Agent
python tests/test_master.py
```

## 示例对话

```
用户: 这个地图里有什么？
助手: 地图共包含 105 条车道，158 条中心线，35 个路口。

用户: 这个位置周围有什么车道？
助手: 位置 (63.9, -149.4) 周围 100米范围内共有 82 条车道。
      车道类型：solid 8条、bilateral 43条、dashed 7条...
      注意：存在双实线，禁止跨越。

用户: 这辆车会右转吗？
助手: 预测行为：straight（置信度 80%）。
      依据：双实线，不允许变道；速度较低(10.0m/s)。

用户: 从这里到路口怎么走？
助手: 推荐路径：距离 1.0 公里，预计用时 2 分钟。
      注意：路径中存在双实线路段，禁止变道。
```

## 开发指南

### 添加新的 Agent

1. 继承 `BaseAgent` 类
2. 实现 `get_tools()` 方法定义工具
3. 实现 `process()` 方法处理查询
4. 在 `MasterAgent` 中注册

```python
from src.agents.base import BaseAgent, AgentContext

class MyAgent(BaseAgent):
    def __init__(self, context: AgentContext):
        super().__init__(context)
        self.name = "my_agent"

    def get_tools(self):
        return [
            {
                "name": "my_tool",
                "description": "工具描述",
                "parameters": {...},
                "handler": self._my_tool_handler
            }
        ]

    def process(self, query: str, **kwargs):
        # 处理逻辑
        return {"result": "..."}
```

### 添加新的工具

```python
# 在 Agent 中添加工具定义
{
    "name": "tool_name",
    "description": "工具描述",
    "parameters": {
        "param1": {"type": "string", "description": "参数描述"},
        "param2": {"type": "number", "description": "参数描述"}
    },
    "handler": self._tool_handler
}

# 实现处理函数
def _tool_handler(self, param1: str, param2: float) -> Dict:
    # 调用 MapAPI 或其他服务
    return {"result": "..."}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。