#!/bin/bash

# LongBench v2 在线推理示例脚本
# 支持默认设置或通过参数自定义

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TIMESTAMP

# 参数变量初始化
MODEL=""
LLM_URL=""
DATASET_FILE=""
TEMPLATE=""
SAVE_DIR=""
DOMAIN=""
MAX_TOKENS=""
MAX_CONTEXT_LENGTH=""
COT_FLAG=1
MAX_SAMPLES=""
RESUME_FLAG=0
LOCAL_TOKENIZER=""
MAX_MODEL_LEN=""
TIMEOUT=""
TEMPERATURE=""
NO_TRUNCATE_FLAG=0
CONCURRENCY=1

show_help() {
    echo "Usage: bash $0 [options]"
    echo "Options:"
    echo "  --model NAME              Model name for API (e.g., Qwen2.5-14B-Instruct)"
    echo "  --llm_url URL             LLM service base URL (e.g., http://127.0.0.1:7800/v1)"
    echo "  --dataset PATH            Path to dataset file (optional, will load from HuggingFace if not set)"
    echo "  --template PATH           Path to prompt template file"
    echo "  --save_dir PATH           Directory to save results"
    echo "  --domain STR              Domain to filter (e.g., 'Single-Document QA')"
    echo "  --max_tokens INT          Maximum tokens to generate"
    echo "  --max_context_length INT  Maximum context length (optional)"
    echo "  --cot                     Enable Chain-of-Thought evaluation"
    echo "  --max_samples INT         Maximum number of samples to process (optional)"
    echo "  --resume                  Resume from existing output file"
    echo "  --local_tokenizer PATH    Local tokenizer path (optional)"
    echo "  --max_model_len INT       Model's maximum context length for truncation (default: 32768)"
    echo "  --timeout INT             Connection timeout in seconds (default: 30)"
    echo "  --temperature FLOAT      Sampling temperature (default: 0.1)"
    echo "  --no_truncate             Disable prompt truncation"
    echo "  --concurrency INT         Number of concurrent requests (default: 1)"
    echo "  -h, --help                Show help"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --llm_url)
            LLM_URL="$2"
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
        --local_tokenizer)
            LOCAL_TOKENIZER="$2"
            shift 2
            ;;
        --max_model_len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --no_truncate)
            NO_TRUNCATE_FLAG=1
            shift
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
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
MODEL="${MODEL:-Qwen3-32B}"
LLM_URL="${LLM_URL:-http://127.0.0.1:7800/v1}"
DATASET_FILE="${DATASET_FILE:-${CODE_ROOT}/eval/data/longbench-v2/data.json}"
TEMPLATE="${TEMPLATE:-${CODE_ROOT}/eval/prompts/0shot.txt}"
SAVE_DIR="${SAVE_DIR:-${CODE_ROOT}/eval/ucm_sparse_predictions/LongBench_v2}"
DOMAIN="${DOMAIN:-Long Structured Data Understanding}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
COT_FLAG="${COT_FLAG:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TIMEOUT="${TIMEOUT:-30}"
TEMPERATURE="${TEMPERATURE:-0.1}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-/home/models/Qwen3-32B}"
CONCURRENCY="${CONCURRENCY:-1}"

echo "------------- Final Arguments -------------"
echo "MODEL              = $MODEL"
echo "LLM_URL            = $LLM_URL"
echo "DATASET_FILE       = $DATASET_FILE"
echo "TEMPLATE           = $TEMPLATE"
echo "SAVE_DIR           = $SAVE_DIR"
echo "DOMAIN             = $DOMAIN"
echo "MAX_TOKENS         = $MAX_TOKENS"
if [[ -n "$MAX_CONTEXT_LENGTH" ]]; then
    echo "MAX_CONTEXT_LENGTH = $MAX_CONTEXT_LENGTH"
fi
echo "COT_FLAG           = $COT_FLAG"
if [[ -n "$MAX_SAMPLES" ]]; then
    echo "MAX_SAMPLES        = $MAX_SAMPLES"
fi
echo "RESUME_FLAG        = $RESUME_FLAG"
if [[ -n "$LOCAL_TOKENIZER" ]]; then
    echo "LOCAL_TOKENIZER    = $LOCAL_TOKENIZER"
fi
echo "MAX_MODEL_LEN      = $MAX_MODEL_LEN"
echo "TIMEOUT            = $TIMEOUT"
echo "TEMPERATURE        = $TEMPERATURE"
echo "CONCURRENCY        = $CONCURRENCY"
if [[ "$NO_TRUNCATE_FLAG" == "1" ]]; then
    echo "NO_TRUNCATE        = true"
fi
echo "--------------------------------------------"

# 检查 LLM 服务是否可用
echo "检查 LLM 服务连接: $LLM_URL"
TEST_BODY='{"model":"Qwen3-32B","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
if ! curl -s --max-time 5 -X POST "${LLM_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$TEST_BODY" > /dev/null 2>&1; then
    echo "警告: 无法连接到 LLM 服务 $LLM_URL"
    echo "请确保服务正在运行"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 构建 Python 命令参数
PYTHON_ARGS=(
    --model "$MODEL"
    --llm_url "$LLM_URL"
    --save_dir "$SAVE_DIR"
    --domain "$DOMAIN"
    --max_tokens "$MAX_TOKENS"
    --max_model_len "$MAX_MODEL_LEN"
    --timeout "$TIMEOUT"
    --temperature "$TEMPERATURE"
    --concurrency "$CONCURRENCY"
)

if [[ -n "$TEMPLATE" ]]; then
    PYTHON_ARGS+=(--template "$TEMPLATE")
fi

if [[ -n "$DATASET_FILE" ]]; then
    PYTHON_ARGS+=(--dataset "$DATASET_FILE")
fi

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

if [[ -n "$LOCAL_TOKENIZER" ]]; then
    PYTHON_ARGS+=(--local_tokenizer "$LOCAL_TOKENIZER")
fi

if [[ "$NO_TRUNCATE_FLAG" == "1" ]]; then
    PYTHON_ARGS+=(--no_truncate)
fi

# 运行推理
echo "Running inference..."
python3 "${CODE_ROOT}/eval/online_inference_longbench_v2_acc.py" \
    "${PYTHON_ARGS[@]}"

# 分析结果
# 查找最新的结果文件
if [[ -d "$SAVE_DIR" ]]; then
    # 查找最新的 jsonl 文件（使用 ls -t 按时间排序）
    LATEST_FILE=$(ls -t "$SAVE_DIR"/*.jsonl 2>/dev/null | head -1)
    
    if [[ -n "$LATEST_FILE" && -f "$LATEST_FILE" ]]; then
        echo "Analyzing results from: $LATEST_FILE"
        python3 "${CODE_ROOT}/eval/analyze_longbench_v2_results.py" "$LATEST_FILE"
    else
        echo "Warning: No result file found in $SAVE_DIR"
    fi
else
    echo "Warning: Save directory $SAVE_DIR does not exist"
fi

echo "Done!"

