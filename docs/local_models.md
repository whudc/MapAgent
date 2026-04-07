# 本地模型使用指南

MapAgent 支持使用本地部署的 Qwen 和 Gemma4 模型，无需依赖云端 API。

## 目录结构

```
model/
├── Qwen/
│   └── Qwen3___5-35B-A3B/    # Qwen 模型目录
├── gemma4/
│   ├── gemma-4-31B-it-Q4_K_M.gguf
│   ├── gemma-4-31B-it-Q4_0.gguf
│   ├── gemma-4-31B-it-Q5_K_S.gguf
│   ├── gemma-4-31B-it-Q6_K.gguf
│   └── ...
```

## 启动本地模型服务

### 方法 1: 使用启动脚本

```bash
# 启动 Qwen 模型 (端口 8000)
./scripts/start_local_model.sh --qwen

# 启动 Gemma4 模型 (端口 8001)
./scripts/start_local_model.sh --gemma4

# 启动 Gemma4 指定量化版本
./scripts/start_local_model.sh --gemma4 --model Q6_K

# 使用 llama.cpp 后端启动 Gemma4
./scripts/start_local_model.sh --gemma4 --backend llama.cpp
```

### 方法 2: 手动启动 vLLM

**Qwen 模型:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model model/Qwen/Qwen3___5-35B-A3B \
    --served-model-name Qwen3___5-35B-A3B \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype auto
```

### 方法 3: 使用 llama.cpp (Gemma4 GGUF)

```bash
llama-server \
    --model model/gemma4/gemma-4-31B-it-Q4_K_M.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    --ctx-size 4096 \
    --n-gpu-layers -1
```

## 在 Web UI 中使用本地模型

1. 启动 Web UI: `python run_ui.py`
2. 在 LLM 提供商下拉菜单中选择 `qwen_local` 或 `gemma4_local`
3. 本地模型无需 API Key，可直接使用
4. 如果需要更改端口，可以在 UI 中配置

## 代码中使用本地模型

```python
from core.llm_client import create_client, create_qwen_client, create_gemma4_client

# 方式 1: 通用创建方式
client = create_client("qwen_local", port=8000)
client = create_client("gemma4_local", port=8001)

# 方式 2: 专用创建方式
client = create_qwen_client(port=8000)
client = create_gemma4_client(port=8001, gguf_file="gemma-4-31B-it-Q6_K.gguf")

# 简单对话
response = client.chat_simple("你好，介绍一下你自己")
print(response)
```

## 环境变量配置

```bash
# Qwen 配置
export LLM_PROVIDER=qwen_local
export QWEN_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=Qwen3___5-35B-A3B

# Gemma4 配置
export LLM_PROVIDER=gemma4_local
export GEMMA4_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=gemma-4-31B-it
```

## 注意事项

1. **硬件要求**: 本地模型需要足够的 GPU 内存
   - Qwen 35B: 建议至少 24GB VRAM
   - Gemma4 31B Q4_K_M: 建议至少 16GB VRAM

2. **量化选择**: GGUF 模型有多种量化版本
   - Q4_K_M: 平衡质量和速度 (推荐)
   - Q5_K_S: 更高质量，需要更多内存
   - Q6_K: 最高质量，内存需求最大

3. **服务状态**: 确保 vLLM 或 llama.cpp 服务正常运行
   - Qwen 默认端口: 8000
   - Gemma4 默认端口: 8001

4. **首次加载**: 模型首次加载可能需要较长时间