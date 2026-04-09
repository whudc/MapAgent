#!/bin/bash
# 启动本地模型服务脚本
# 支持 Qwen3_5 和 Gemma4 模型

set -e

PROJECT_ROOT="/data/DC/MapAgent"
MODEL_DIR="$PROJECT_ROOT/model"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  MapAgent 本地模型服务启动脚本"
echo "=========================================="

# 显示帮助信息
show_help() {
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --qwen          启动 Qwen3_5 模型服务 (端口 8000)"
    echo "  --gemma4        启动 Gemma4 模型服务 (端口 8001)"
    echo "  --port PORT     指定端口"
    echo "  --backend TYPE  指定后端类型: vllm 或 llama.cpp (默认: vllm)"
    echo "  --gpus N        指定GPU数量 (默认: 4)"
    echo "  --gpu-ids IDS   指定GPU卡号列表 (如: 0,1,2,3 或 0,2)"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --qwen                    # 启动 Qwen3_5 模型 (4卡)"
    echo "  $0 --gemma4                  # 启动 Gemma4 模型 (4卡)"
    echo "  $0 --qwen --port 9000        # 在端口 9000 启动 Qwen"
    echo "  $0 --qwen --gpus 2           # 使用2卡启动 Qwen"
    echo "  $0 --qwen --gpu-ids 0,2      # 使用GPU 0和2启动 Qwen"
    echo "  $0 --gemma4 --gpu-ids 4,5,6,7  # 使用GPU 4-7启动 Gemma4"
    echo ""
}

# 参数解析
MODEL_TYPE=""
PORT=""
BACKEND="vllm"
NUM_GPUS=4
GPU_IDS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --qwen)
            MODEL_TYPE="qwen"
            PORT=8000
            shift
            ;;
        --gemma4)
            MODEL_TYPE="gemma4"
            PORT=8001
            shift
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        --backend)
            BACKEND=$2
            shift 2
            ;;
        --gpus)
            NUM_GPUS=$2
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS=$2
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_TYPE" ]; then
    echo -e "${RED}请指定模型类型: --qwen 或 --gemma4${NC}"
    show_help
    exit 1
fi

# 检查模型是否存在
if [ "$MODEL_TYPE" == "qwen" ]; then
    QWEN_MODEL="$MODEL_DIR/qwen"
    if [ ! -d "$QWEN_MODEL" ]; then
        echo -e "${RED}错误: Qwen 模型目录不存在: $QWEN_MODEL${NC}"
        exit 1
    fi
    MODEL_PATH="$QWEN_MODEL"
    MODEL_NAME="Qwen3_5"
elif [ "$MODEL_TYPE" == "gemma4" ]; then
    GEMMA_MODEL="$MODEL_DIR/gemma4"
    if [ ! -d "$GEMMA_MODEL" ]; then
        echo -e "${RED}错误: Gemma4 模型目录不存在: $GEMMA_MODEL${NC}"
        exit 1
    fi
    MODEL_PATH="$GEMMA_MODEL"
    MODEL_NAME="Gemma4"
fi

# 处理 GPU 配置
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    GPU_INFO="GPU卡号: $GPU_IDS"
else
    GPU_INFO="GPU数量: $NUM_GPUS"
fi

echo -e "${GREEN}模型配置:${NC}"
echo "  类型: $MODEL_TYPE"
echo "  模型: $MODEL_NAME"
echo "  路径: $MODEL_PATH"
echo "  端口: $PORT"
echo "  后端: $BACKEND"
echo "  $GPU_INFO"
echo ""

# 启动服务
echo -e "${YELLOW}正在启动服务...${NC}"

if [ "$BACKEND" == "vllm" ]; then
    if [ "$MODEL_TYPE" == "qwen" ]; then
        echo "使用 vLLM 启动 Qwen3_5 模型 (${NUM_GPUS}卡张量并行)..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "Qwen3_5" \
            --port $PORT \
            --host 0.0.0.0 \
            --trust-remote-code \
            --dtype auto \
            --tensor-parallel-size $NUM_GPUS \
            --enable-auto-tool-choice \
            --tool-call-parser "hermes"
    elif [ "$MODEL_TYPE" == "gemma4" ]; then
        echo "使用 vLLM 启动 Gemma4 模型 (${NUM_GPUS}卡张量并行)..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "Gemma4" \
            --port $PORT \
            --host 0.0.0.0 \
            --trust-remote-code \
            --dtype auto \
            --tensor-parallel-size $NUM_GPUS \
            --max-model-len 8192 \
            --enable-auto-tool-choice \
            --tool-call-parser "hermes"
    fi
elif [ "$BACKEND" == "llama.cpp" ]; then
    echo -e "${YELLOW}注意: 当前模型为 safetensors 格式，推荐使用 vLLM 后端${NC}"
    echo "llama.cpp 后端需要 GGUF 格式模型"
    exit 1
else
    echo -e "${RED}未知后端类型: $BACKEND${NC}"
    echo "支持的后端: vllm, llama.cpp"
    exit 1
fi