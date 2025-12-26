#!/bin/bash

# LongBench v2 离线推理示例脚本
# 支持默认设置或通过参数自定义

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# 环境变量设置

export ENABLE_SPARSE=true
export VLLM_HASH_ATTENTION=1

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TIMESTAMP

# 参数变量初始化
MODEL_PATH=""
STORAGE_BACKENDS=""
UCM_SPARSE_CONFIG=""
DATASET_FILE=""
TEMPLATE=""
SAVE_DIR=""
DOMAIN=""
MAX_TOKENS=""
BATCH_SIZE=""
MAX_CONTEXT_LENGTH=""
COT_FLAG=""
MAX_SAMPLES=""
RESUME_FLAG=0

show_help() {
    echo "Usage: bash $0 [options]"
    echo "Options:"
    echo "  --model PATH              Path to model"
    echo "  --storage PATH            Path to KV cache storage directory"
    echo "  --config PATH             Path to UCM sparse config"
    echo "  --dataset PATH            Path to dataset file (optional, will load from HuggingFace if not set)"
    echo "  --cuda_devices STR        CUDA visible devices (e.g., '2,3')"
    echo "  --template PATH          Path to prompt template file"
    echo "  --save_dir PATH          Directory to save results"
    echo "  --domain STR             Domain to filter (e.g., 'Single-Document QA')"
    echo "  --max_tokens INT         Maximum tokens to generate"
    echo "  --batch_size INT         Batch size for inference"
    echo "  --max_context_length INT Maximum context length (optional)"
    echo "  --cot                    Enable Chain-of-Thought evaluation"
    echo "  --max_samples INT        Maximum number of samples to process (optional)"
    echo "  --resume                 Resume from existing output file"
    echo "  -h, --help               Show help"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --storage)
            STORAGE_BACKENDS="$2"
            shift 2
            ;;
        --config)
            UCM_SPARSE_CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILE="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_context_length)
            MAX_CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --cot)
            COT_FLAG=1
            shift
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# 设置默认值
MODEL_PATH="${MODEL_PATH:-/home/models/Qwen3-32B}"
STORAGE_BACKENDS="${STORAGE_BACKENDS:-${CODE_ROOT}/ucm_kv_cache}"
UCM_SPARSE_CONFIG="${UCM_SPARSE_CONFIG:-${CODE_ROOT}/eval/ucm_sparse_config_esa.json}"

UCM_CONFIG_NAME=$(basename "$UCM_SPARSE_CONFIG") 
UCM_CONFIG_NAME_NO_EXT="${UCM_CONFIG_NAME%.*}" 

DATASET_FILE="${DATASET_FILE:-${CODE_ROOT}/eval/data/longbench-v2/data.json}"
TEMPLATE="${TEMPLATE:-${CODE_ROOT}/eval/prompts/0shot.txt}"
SAVE_DIR="${SAVE_DIR:-${CODE_ROOT}/eval/ucm_sparse_predictions/LongBench_v2/${UCM_CONFIG_NAME_NO_EXT}}"
DOMAIN="${DOMAIN:-Single-Document QA}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
BATCH_SIZE="${BATCH_SIZE:-20}"
COT_FLAG="${COT_FLAG:-1}"

# 导出环境变量
export MODEL_PATH
export STORAGE_BACKENDS
export UCM_SPARSE_CONFIG
if [[ -n "$DATASET_FILE" ]]; then
    export DATASET_FILE
fi


echo "------------- Final Arguments -------------"
echo "MODEL_PATH          = $MODEL_PATH"
echo "STORAGE_BACKENDS    = $STORAGE_BACKENDS"
echo "UCM_SPARSE_CONFIG   = $UCM_SPARSE_CONFIG"
echo "DATASET_FILE        = $DATASET_FILE"
echo "TEMPLATE            = $TEMPLATE"
echo "SAVE_DIR            = $SAVE_DIR"
echo "DOMAIN              = $DOMAIN"
echo "MAX_TOKENS          = $MAX_TOKENS"
echo "BATCH_SIZE          = $BATCH_SIZE"
if [[ -n "$MAX_CONTEXT_LENGTH" ]]; then
    echo "MAX_CONTEXT_LENGTH  = $MAX_CONTEXT_LENGTH"
fi
echo "COT_FLAG            = $COT_FLAG"
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "MAX_SAMPLES         = $MAX_SAMPLES"
fi
echo "RESUME_FLAG         = $RESUME_FLAG"
echo "--------------------------------------------"

# 构建 Python 命令参数
PYTHON_ARGS=(
    --template "$TEMPLATE"
    --save_dir "$SAVE_DIR"
    --domain "$DOMAIN"
    --max_tokens "$MAX_TOKENS"
    --batch_size "$BATCH_SIZE"
)

if [[ "$COT_FLAG" == "1" ]]; then
    PYTHON_ARGS+=(--cot)
fi

if [[ -n "$MAX_CONTEXT_LENGTH" ]]; then
    PYTHON_ARGS+=(--max_context_length "$MAX_CONTEXT_LENGTH")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    PYTHON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "$RESUME_FLAG" == "1" ]]; then
    PYTHON_ARGS+=(--resume)
fi

# 运行推理
echo "Running inference..."
python "${CODE_ROOT}/eval/offline_inference_longbench_v2_acc.py" \
    "${PYTHON_ARGS[@]}"

# 分析结果
# 查找最新的结果文件
if [[ -d "$SAVE_DIR" ]]; then
    # 查找最新的 jsonl 文件（使用 ls -t 按时间排序）
    LATEST_FILE=$(ls -t "$SAVE_DIR"/*.jsonl 2>/dev/null | head -1)
    
    if [[ -n "$LATEST_FILE" && -f "$LATEST_FILE" ]]; then
        echo "Analyzing results from: $LATEST_FILE"
        python "${CODE_ROOT}/eval/analyze_longbench_v2_results.py" "$LATEST_FILE"
    else
        echo "Warning: No result file found in $SAVE_DIR"
    fi
else
    echo "Warning: Save directory $SAVE_DIR does not exist"
fi

# 清理 KV cache
rm -rf "${STORAGE_BACKENDS}"

# echo "Done!"
