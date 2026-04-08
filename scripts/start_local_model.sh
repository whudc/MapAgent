#!/bin/bash
# 启动本地模型服务脚本
# 支持 Qwen 和 Gemma4 模型

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
    echo "  --qwen          启动 Qwen 模型服务 (端口 8000)"
    echo "  --gemma4        启动 Gemma4 模型服务 (端口 8001)"
    echo "  --model MODEL   指定 GGUF 模型文件 (仅 Gemma4)"
    echo "  --port PORT     指定端口"
    echo "  --backend TYPE  指定后端类型: vllm 或 llama.cpp"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --qwen                    # 启动 Qwen 模型"
    echo "  $0 --gemma4                  # 启动 Gemma4 默认模型"
    echo "  $0 --gemma4 --model Q6_K     # 启动 Gemma4 Q6_K 量化模型"
    echo "  $0 --qwen --port 9000        # 在端口 9000 启动 Qwen"
    echo ""
}

# 参数解析
MODEL_TYPE=""
PORT=""
GGUF_MODEL="gemma-4-31B-it-Q4_K_M.gguf"
BACKEND="vllm"

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
        --model)
            GGUF_MODEL="gemma-4-31B-it-$2.gguf"
            shift 2
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        --backend)
            BACKEND=$2
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
    QWEN_MODEL="$MODEL_DIR/Qwen/Qwen3___5-35B-A3B"
    if [ ! -d "$QWEN_MODEL" ]; then
        echo -e "${RED}错误: Qwen 模型目录不存在: $QWEN_MODEL${NC}"
        exit 1
    fi
    MODEL_PATH="$QWEN_MODEL"
    MODEL_NAME="Qwen3___5-35B-A3B"
elif [ "$MODEL_TYPE" == "gemma4" ]; then
    GEMMA_MODEL="$MODEL_DIR/gemma4/$GGUF_MODEL"
    if [ ! -f "$GEMMA_MODEL" ]; then
        echo -e "${RED}错误: Gemma4 GGUF 模型文件不存在: $GEMMA_MODEL${NC}"
        echo "可用的模型文件:"
        ls -la "$MODEL_DIR/gemma4/*.gguf" 2>/dev/null || echo "未找到 GGUF 文件"
        exit 1
    fi
    MODEL_PATH="$GEMMA_MODEL"
    MODEL_NAME="$GGUF_MODEL"
fi

echo -e "${GREEN}模型配置:${NC}"
echo "  类型: $MODEL_TYPE"
echo "  模型: $MODEL_NAME"
echo "  路径: $MODEL_PATH"
echo "  端口: $PORT"
echo "  后端: $BACKEND"
echo ""

# 启动服务
echo -e "${YELLOW}正在启动服务...${NC}"

if [ "$BACKEND" == "vllm" ]; then
    if [ "$MODEL_TYPE" == "qwen" ]; then
        # vLLM 启动 Qwen
        echo "使用 vLLM 启动 Qwen 模型..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --port $PORT \
            --host 0.0.0.0 \
            --trust-remote-code \
            --dtype auto
    elif [ "$MODEL_TYPE" == "gemma4" ]; then
        # vLLM 启动 Gemma4 (需要先转换 GGUF 到 HuggingFace 格式)
        echo -e "${YELLOW}注意: vLLM 不直接支持 GGUF 格式${NC}"
        echo "请使用 llama.cpp 后端启动 Gemma4 GGUF 模型"
        echo "命令: $0 --gemma4 --backend llama.cpp"
        exit 1
    fi
elif [ "$BACKEND" == "llama.cpp" ]; then
    # llama.cpp 启动
    if [ "$MODEL_TYPE" == "qwen" ]; then
        echo -e "${YELLOW}注意: llama.cpp 需要 GGUF 格式的 Qwen 模型${NC}"
        echo "请先转换模型或使用 vLLM 后端"
        exit 1
    elif [ "$MODEL_TYPE" == "gemma4" ]; then
        echo "使用 llama.cpp 启动 Gemma4 GGUF 模型..."
        llama-server \
            --model "$MODEL_PATH" \
            --port $PORT \
            --host 0.0.0.0 \
            --ctx-size 4096 \
            --n-gpu-layers -1
    fi
else
    echo -e "${RED}未知后端类型: $BACKEND${NC}"
    echo "支持的后端: vllm, llama.cpp"
    exit 1
fi