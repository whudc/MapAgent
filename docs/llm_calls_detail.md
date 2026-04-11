# MapAgent LLM 调用详解

## 一、LLM 客户端架构

项目使用 `src/core/llm_client.py` 中的 `LLMClient` 类统一封装多种模型：

```python
# 支持的提供商
LLMProvider.ANTHROPIC   # Claude (claude-sonnet-4-6)
LLMProvider.DEEPSEEK    # DeepSeek API
LLMProvider.OPENAI      # GPT-4o
LLMProvider.QWEN_LOCAL  # 本地 Qwen (端口 8000)
LLMProvider.GEMMA4_LOCAL # 本地 Gemma4 (端口 8001)
```

---

## 二、调用方式

所有 LLM 调用通过 `chat_simple()` 方法：

```python
# llm_client.py:604-608
def chat_simple(self, user_message: str, system_prompt: str = "") -> str:
    """简单对话（不使用工具）"""
    messages = [{"role": "user", "content": user_message}]
    return self._client.chat(messages=messages, system=system_prompt)
```

**输入格式**：
- 单条用户消息 `{"role": "user", "content": "..."}`
- 可选的系统提示（作为单独参数传递）
- 最终组装成 OpenAI/Anthropic 兼容的 messages 格式

---

## 三、具体调用场景（共 6 种）

### 1. 车道数量守恒分析 (`traffic_flow.py:301-315`)

**触发条件**：车道内车辆数量变化超过 1 时

**输入 Prompt**：
```text
【车道数量分析】
车道: {lane_id}, 帧: {prev_frame_id}→{curr_frame_id}
车辆数: {prev_count}→{curr_count} ({diff:+d})

【前一帧轨迹】(最多5条)
- ID{track_id}: ({pos_x:.1f},{pos_y:.1f})

【当前帧检测】(最多5个)
- 检测{i}: ({pos_x:.1f},{pos_y:.1f})

分析数量变化原因:
- miss: 漏检(遮挡)
- false: 误检
- exit: 驶离
- enter: 驶入

返回JSON: {"cause":"miss/false/exit/enter", "confidence":0.0-1.0, "action":"keep/remove/interpolate", "reasoning":"说明"}
```

---

### 2. 遮挡分析 (`traffic_flow.py:420-450`)

**触发条件**：轨迹丢失时判断是否被其他车辆遮挡

**输入 Prompt**：
```text
【遮挡分析】
轨迹ID: {track_id}
最后位置: {last_pos}
预测位置: {predicted_pos}
丢失帧数: {lost_count}

【附近轨迹】(最多3条)
- ID{track_id}: ({pos_x:.1f},{pos_y:.1f})

【当前检测】(最多3个)
- ({pos_x:.1f},{pos_y:.1f})

判断: 是否被遮挡?
返回JSON: {"is_occluded":true/false, "occluder_id":ID或null, "confidence":0.0-1.0, "action":"keep/interpolate/remove", "reasoning":"说明"}
```

---

### 3. ID 一致性分析（防止帧间 ID 跳变） (`traffic_flow.py:507-672`)

#### 问题场景

同一物理目标在连续帧之间被 DeepSORT 分配了不同 ID。例如：
- 帧 N：检测到车辆，DeepSORT 分配 ID=5
- 帧 N+1：同一车辆被再次检测到，但 DeepSORT 分配了新 ID=6

这就是 **ID 跳变问题** —— 同一辆车的 ID 不连续。

#### 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  帧 N+1 处理流程                                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. DeepSORT 跟踪 → 得到 N 条轨迹                                 │
│     例如：ID=5, ID=6, ID=7, ...                                  │
│                                                                  │
│  2. 先收集所有活跃轨迹信息：                                       │
│     [{"track_id": 5, "pos": [x,y], "matched": True},             │
│      {"track_id": 6, "pos": [x,y], "matched": True},             │
│      {"track_id": 7, "pos": [x,y], "matched": False}, ...]       │
│                                                                  │
│  3. 对每条轨迹调用 LLM 分析：                                      │
│                                                                  │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │  输入参数：                                               │ │
│     │  - track_id: 当前分析的轨迹 ID                            │ │
│     │  - track_history: 轨迹历史（最近 5-10 帧）                │ │
│     │  - track_matched: 该轨迹是否被 DeepSORT 匹配到检测        │ │
│     │  - matched_det_pos: 匹配到的检测位置（如果 matched=True） │ │
│     │  - other_tracks_info: 其他所有活跃轨迹的信息              │ │
│     │  - recent_detections: 当前帧附近检测结果                  │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │  【情况 A】轨迹已匹配（track_matched=True）                │ │
│     │                                                          │ │
│     │  问题：该轨迹的跟踪是否正确？                              │ │
│     │                                                          │ │
│     │  LLM 分析：                                               │ │
│     │  1. 当前位置 vs 前一帧位置，移动距离是否合理（< 3m/帧）？  │ │
│     │  2. 匹配的检测位置是否合理？                               │ │
│     │                                                          │ │
│     │  输出：                                                   │ │
│     │  - "correct": 跟踪正确，位置连续                          │ │
│     │  - "error": 跟踪错误，位置跳跃过大，可能误匹配            │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │  【情况 B】轨迹未匹配（track_matched=False，轨迹丢失）     │ │
│     │                                                          │ │
│     │  问题：该轨迹去哪了？是否被其他 ID 占用？                  │ │
│     │                                                          │ │
│     │  LLM 分析：                                               │ │
│     │  1. 查看 other_tracks_info 中其他轨迹的位置               │ │
│     │  2. 判断是否有其他轨迹位置接近该轨迹的预测位置             │ │
│     │  3. 如果有 → ID 跳变！其他轨迹实际是该轨迹的延续          │ │
│     │                                                          │ │
│     │  输出：                                                   │ │
│     │  - "disappeared": 轨迹确实消失（驶离/遮挡）               │ │
│     │  - "id_jump_to_X": 轨迹被 ID=X 占用，应合并               │ │
│     │  - "unknown": 无法判断                                    │ │
│     └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### 判断轨迹是否匹配的标准

```python
# track.time_since_update == 0 表示刚刚被更新（匹配到了检测）
track_matched = track.time_since_update == 0
```

#### 输入 Prompt（已匹配轨迹）

```text
【ID 一致性分析 - 帧间跟踪验证】

轨迹 ID: {track_id}
当前帧: {frame_id}

【轨迹历史】(最近 {n} 帧)
- 帧{frame_id}: ({pos_x:.1f}, {pos_y:.1f})

【前一帧位置】: ({prev_x:.1f}, {prev_y:.1f})
【轨迹当前位置】: ({curr_x:.1f}, {curr_y:.1f})
【移动距离】: {movement:.2f}m

【DeepSORT 匹配状态】: 已匹配到检测
【匹配到的检测位置】: ({det_x:.1f}, {det_y:.1f})

【当前帧附近检测】(共 {n} 个，30米范围内)
- 检测0: ({pos_x:.1f}, {pos_y:.1f}), 类型={type} → 已被ID=5匹配
- 检测1: ({pos_x:.1f}, {pos_y:.1f}), 类型={type} → 已被ID=6匹配
- 检测2: ({pos_x:.1f}, {pos_y:.1f}), 类型={type}
...

【其他活跃轨迹】(共 {n} 条)
- ID{id}: ({pos_x:.1f}, {pos_y:.1f}), 状态=已匹配/丢失

【分析任务 - 匹配验证】
该轨迹已被 DeepSORT 匹配到检测，请验证：
1. 当前位置与前一帧位置是否连续？（移动距离应 < 3m/帧）
2. 匹配的检测是否合理？

可能结论：
- "correct": 跟踪正确，位置连续
- "error": 跟踪错误，位置跳跃过大，可能是误匹配

返回 JSON:
{"decision": "correct或error", "confidence": 0.0-1.0, "reasoning": "简要说明"}
```

#### 输入 Prompt（未匹配轨迹）

```text
【ID 一致性分析 - 帧间跟踪验证】

轨迹 ID: {track_id}
当前帧: {frame_id}

【轨迹历史】(最近 {n} 帧)
- 帧{frame_id}: ({pos_x:.1f}, {pos_y:.1f})

【前一帧位置】: ({prev_x:.1f}, {prev_y:.1f})
【轨迹当前位置】: ({curr_x:.1f}, {curr_y:.1f})  # 预测位置
【移动距离】: {movement:.2f}m

【DeepSORT 匹配状态】: 未匹配（轨迹丢失）

【当前帧附近检测】(共 {n} 个，30米范围内)
- 检测0: ({pos_x:.1f}, {pos_y:.1f}), 类型={type} → 已被ID=6匹配
- 检测1: ({pos_x:.1f}, {pos_y:.1f}), 类型={type} → 已被ID=7匹配
- 检测2: ({pos_x:.1f}, {pos_y:.1f}), 类型={type}
...

【其他活跃轨迹】(共 {n} 条)
- ID{id}: ({pos_x:.1f}, {pos_y:.1f}), 状态=已匹配/丢失

【分析任务 - 丢失轨迹分析】
该轨迹在当前帧丢失（未匹配到检测），请分析：
1. 轨迹是否真的消失（驶离/遮挡）？
2. 其他已匹配轨迹的位置是否可能是该轨迹的延续？（检查 ID 是否被抢占）

如果其他已匹配轨迹位置接近该轨迹的预测位置，可能是 ID 跳变。

可能结论：
- "disappeared": 轨迹确实消失（驶离/遮挡）
- "id_jump_to_X": 轨迹被 ID=X 占用，应合并
- "unknown": 无法判断

返回 JSON:
{"decision": "disappeared或id_jump_to_X或unknown", "confidence": 0.0-1.0, "jump_to_id": null或ID号, "reasoning": "简要说明"}
```

#### 实际场景示例

**场景：ID 跳变**
```
帧 100：检测位置 (10.5, 20.3)，DeepSORT 分配 ID=5
帧 101：检测位置 (10.8, 20.5)，DeepSORT 分配 ID=6（错误）

LLM 分析 ID=5 的轨迹（未匹配状态）：
┌──────────────────────────────────────┐
│ 轨迹 ID=5：                           │
│ - 最后位置：(10.5, 20.3)              │
│ - 预测位置：(10.7, 20.4)              │
│ - 匹配状态：未匹配                    │
│                                      │
│ 【其他活跃轨迹】                      │
│ - ID6: (10.8, 20.5), 状态=已匹配      │
│                                      │
│ LLM 输出：                            │
│ {"decision": "id_jump_to_6",         │
│  "jump_to_id": 6,                    │
│  "confidence": 0.9,                  │
│  "reasoning": "ID6位置接近ID5预测"}  │
└──────────────────────────────────────┘

后续处理：将 ID=6 的轨迹合并/修正为 ID=5
```

**场景：跟踪正确**
```
帧 100：检测位置 (10.5, 20.3)，ID=5
帧 101：检测位置 (10.8, 20.5)，ID=5（正确匹配）

LLM 分析：
- 匹配状态：已匹配
- 移动距离：0.4m
- 输出：{"decision": "correct", "reasoning": "位置连续，跟踪正确"}
```

**场景：车辆驶离**
```
帧 100：检测位置 (10.5, 20.3)，ID=5
帧 101：ID=5 所在位置附近无任何检测

LLM 分析：
- 匹配状态：未匹配
- 其他轨迹都在远处
- 输出：{"decision": "disappeared", "reasoning": "轨迹附近无检测，已驶离"}
```

---

### 4. 重新出现判断 (`traffic_flow.py:685-714`)

**触发条件**：目标消失后重新出现时判断是否为同一车辆

**输入 Prompt**：
```text
判断重新出现的目标是否是同一轨迹：

旧轨迹信息：
- ID: {id}
- 最后位置：{last_pos}
- 最后车道：{lane_id}
- 丢失帧数：{lost_frames}
- 预测位置：{predicted_pos}

新检测信息：
- 位置：{pos}
- 车道：{lane_id}
- 类型：{type}

车道一致性：{是/否}
- {old_lane} 的后继车道：{successor_lanes}

返回 JSON：
{
    "is_same": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "推理说明"
}
```

---

### 5. 轨迹质量分析 (`traffic_flow.py:780-823`)

**触发条件**：轨迹出现异常问题（速度异常、间隙等）

**输入 Prompt**：
```text
【轨迹质量分析】

轨迹 ID: {track_id}
轨迹长度：{len} 帧

【检测到的问题】
- {type}: {description}

【轨迹历史】（最近 {len} 帧）
- 帧 {frame_id}: 位置 ({pos})

请分析：
1. 这条轨迹是否存在问题？
2. 问题的根本原因是什么？
3. 建议的处理方式是什么？

返回 JSON:
{
    "action": "keep|merge|remove|interpolate",
    "confidence": 0.0-1.0,
    "reasoning": "推理说明",
    "merge_with": 123  // 仅当 action=merge 时需要
}
```

---

### 6. ID 跳变批量分析（跟踪后轨迹合并） (`traffic_flow.py:864-988`)

**问题场景**：跟踪完成后发现两条独立的轨迹可能是同一辆车。例如：
- 轨迹 1：帧 100-105，ID=5，位置从 (10,20) 移动到 (15,25)
- 轨迹 2：帧 107-120，ID=6，位置从 (16,26) 开始

两条轨迹时间不重叠、位置连续，可能是 ID 跳变导致同一辆车被分成两条轨迹。

**触发条件**：跟踪完成后，检测同车道内时间不重叠但位置连续的轨迹对

**判断条件**（`_check_possible_id_jumping`）：
1. 时间不重叠（track1 结束帧 < track2 开始帧）
2. 时间间隔小（< 10 帧）
3. 位置接近（预测位置偏差 < 15 米）
4. 速度方向一致

**输入 Prompt**：
```text
【ID 跳变分析】

检测到一个可能的 ID 跳变情况：

【轨迹 1】（时间上在先）
- ID: {id}                 # 例如 ID=5
- 帧范围：{start} → {end}  # 例如 100 → 105
- 长度：{len} 帧
- 结束位置：{end_pos}      # 例如 (15, 25)
- 结束速度：{end_vel}      # 轨迹 1 结束时的速度向量

【轨迹 2】（时间上在后）
- ID: {id}                 # 例如 ID=6
- 帧范围：{start} → {end}  # 例如 107 → 120
- 长度：{len} 帧
- 起始位置：{start_pos}    # 例如 (16, 26)
- 起始速度：{start_vel}    # 轨迹 2 开始时的速度向量

【间隙信息】
- 时间间隔：{gap} 帧       # 例如 2 帧（106 帧缺失）
- 预测位置：{predicted}    # 基于 track1 速度预测的位置
- 实际起始位置：{actual}   # track2 的实际起始位置
- 位置偏差：{dist:.1f} 米  # 预测位置与实际位置的偏差

请分析：
1. 这两条轨迹是否代表同一辆车？
2. 如果是，合并它们的置信度是多少？

返回 JSON:
{
    "should_merge": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "推理说明"
}
```

**与场景 3 的区别**：
- **场景 3**：实时逐帧分析，防止 ID 在帧间跳变（帧 N → 帧 N+1）
- **场景 6**：跟踪后批量分析，合并已被错误分割的完整轨迹

---

## 四、输入流程总结

```
1. 构建场景数据（轨迹位置、检测结果、车道信息等）
   ↓
2. 格式化为结构化 Prompt（中文标题 + 数据 + JSON 输出要求）
   ↓
3. llm_client.chat_simple(prompt)
   ↓
4. 内部组装为 [{"role": "user", "content": prompt}]
   ↓
5. 发送到配置的 LLM 提供商（Anthropic API / 本地模型 API）
   ↓
6. 解析返回的 JSON 结果
```

---

## 五、关键代码路径

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/core/llm_client.py` | 604-608 | `chat_simple()` 入口 |
| `src/core/llm_client.py` | 341-359 | OpenAI 兼容接口 `chat()` |
| `src/core/llm_client.py` | 244-262 | Anthropic 接口 `chat()` |
| `src/agents/traffic_flow.py` | 301-315 | 车道守恒分析调用 |
| `src/agents/traffic_flow.py` | 420-450 | 遮挡分析调用 |
| `src/agents/traffic_flow.py` | 553-598 | ID 一致性分析调用 |
| `src/agents/traffic_flow.py` | 685-714 | 重新出现判断调用 |
| `src/agents/traffic_flow.py` | 780-823 | 轨迹质量分析调用 |
| `src/agents/traffic_flow.py` | 918-964 | ID 跳变批量分析调用 |