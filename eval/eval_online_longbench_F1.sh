#!/bin/bash

# Check and auto-install required Python packages
REQUIRED_PACKAGES=("fuzzywuzzy" "jieba" "rouge" "requests" "tqdm")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! python3 -c "import $pkg" 2>/dev/null; then
    echo "❌ $pkg not found, installing..."
    pip install "$pkg" --upgrade 2>/dev/null && echo "✅ $pkg installed successfully" || echo "❌ Failed to install $pkg (run 'pip3 install $pkg' manually)"
  fi
done

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TIMESTAMP

# 参数变量初始化
MODEL=""
LLM_URL=""
TEST_DATA_DIR=""
SAVE_DIR=""
STRIP_THINK=1
LOCAL_TOKENIZER=""
MAX_LEN=32768
MAX_TOKENS=2048
TIMEOUT=30
CONCURRENCY=1
RESUME_FLAG=""

show_help() {
    echo "Usage: bash eval_online_inference_F1.sh [options]"
    echo "Options:"
    echo "  --model NAME              Model name for API (e.g., Qwen3-32B)"
    echo "  --llm_url URL             LLM service base URL (e.g., http://127.0.0.1:7800/v1)"
    echo "  --data PATH               Path to test data directory"
    echo "  --save_dir PATH           Directory to save results"
    echo "  --strip_think 0|1         Whether to apply --strip_think in eval"
    echo "  --local_tokenizer PATH    Local tokenizer path (optional)"
    echo "  --max_len INT             Maximum input length (default: 32768)"
    echo "  --max_tokens INT          Maximum generation tokens (default: 2048)"
    echo "  --timeout INT             Connection timeout in seconds (default: 30)"
    echo "  --concurrency INT         Number of concurrent requests (default: 1)"
    echo "  --resume                  Resume from existing result file"
    echo "  -h, --help                Show help"
    exit 0
}

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
        --data)
            TEST_DATA_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --strip_think)
            STRIP_THINK="$2"
            shift 2
            ;;
        --local_tokenizer)
            LOCAL_TOKENIZER="$2"
            shift 2
            ;;
        --max_len)
            MAX_LEN="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --resume)
            RESUME_FLAG="--resume"
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
MODEL="${MODEL:-Qwen3-32B}"
LLM_URL="${LLM_URL:-http://127.0.0.1:7800/v1}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${CODE_ROOT}/eval/data/longbench}"

# 与 offline 版本保持一致的保存路径
SAVE_PATH="${CODE_ROOT}/eval/ucm_sparse_predictions"
DATASET="LongBench"
DATASET_SAVE_DIR="${SAVE_PATH}/${DATASET}"
SAVE_DIR="${SAVE_DIR:-${DATASET_SAVE_DIR}}"
STRIP_THINK="${STRIP_THINK:-1}"
CONCURRENCY="${CONCURRENCY:-20}"
LOCAL_TOKENIZER="${LOCAL_TOKENIZER:-/home/models/Qwen3-32B}"

echo "------------- Final Arguments -------------"
echo "MODEL            = $MODEL"
echo "LLM_URL          = $LLM_URL"
echo "TEST_DATA_DIR    = $TEST_DATA_DIR"
echo "SAVE_DIR         = $SAVE_DIR"
echo "STRIP_THINK      = $STRIP_THINK"
echo "LOCAL_TOKENIZER  = $LOCAL_TOKENIZER"
echo "MAX_LEN          = $MAX_LEN"
echo "MAX_TOKENS       = $MAX_TOKENS"
echo "TIMEOUT          = $TIMEOUT"
echo "CONCURRENCY      = $CONCURRENCY"
echo "RESUME           = $RESUME_FLAG"
echo "--------------------------------------------"


# 创建保存目录（与 offline 版本保持一致）
mkdir -p "$DATASET_SAVE_DIR" || { echo "Failed to create save directory!"; exit 1; }

# -------------------------- LongBench --------------------------
TARGET_FILES=(
"${TEST_DATA_DIR}/multifieldqa_zh.jsonl"
# "${TEST_DATA_DIR}/dureader.jsonl"
)
# ---------------------------------------------------------------
EXISTING_FILES=()
declare -A seen_files 
for file in "${TARGET_FILES[@]}"; do
    if [[ -f "$file" && -z "${seen_files[$file]}" ]]; then
        seen_files["$file"]=1
        EXISTING_FILES+=("$file")
    fi
done
if [[ ${#EXISTING_FILES[@]} -eq 0 ]]; then
    echo "❌ No valid data files found for LongBench!"
    exit 1
fi

echo -e "\nFound ${#EXISTING_FILES[@]} data files for LongBench:"
for file in "${EXISTING_FILES[@]}"; do
    echo "  - $(basename "$file")"
done

for DATASET_FILE in "${EXISTING_FILES[@]}"; do
    filename=$(basename "$DATASET_FILE")
    file_name_no_ext="${filename%.*}"
    
    MODEL_NAME=$(basename "$MODEL")
    RES_FILE="${SAVE_DIR}/${MODEL_NAME}_${file_name_no_ext}_online_${TIMESTAMP}.jsonl"
    
    echo -e "\n======================================"
    echo "Executed model: $MODEL"
    echo "Dataset file: $DATASET_FILE"
    echo "Result file: $RES_FILE"
    echo "======================================"

    INFERENCE_ARGS=(
        "--model" "$MODEL"
        "--dataset" "$DATASET_FILE"
        "--llm_url" "$LLM_URL"
        "--save_dir" "$SAVE_DIR"
        "--max_len" "$MAX_LEN"
        "--max_tokens" "$MAX_TOKENS"
        "--timeout" "$TIMEOUT"
        "--concurrency" "$CONCURRENCY"
    )
    
    # 传递结果文件名给 Python 脚本（通过环境变量）
    export RES_FILE
    
    if [[ -n "$LOCAL_TOKENIZER" ]]; then
        INFERENCE_ARGS+=("--local_tokenizer" "$LOCAL_TOKENIZER")
    fi
    
    if [[ -n "$RESUME_FLAG" ]]; then
        INFERENCE_ARGS+=("$RESUME_FLAG")
    fi
    
    # 运行在线推理
    python3 "${CODE_ROOT}/eval/online_inference_longbench_F1.py" "${INFERENCE_ARGS[@]}"

    if [[ ! -f "$RES_FILE" ]]; then
        echo "Warning: test finished but result file not found: $RES_FILE"
        continue
    fi

    echo -e "\nCalculating F1 score..."
    F1_FILE="${RES_FILE}.f1.txt"
    > /tmp/scores

    extra_args=""
    if [[ "$STRIP_THINK" == "1" ]]; then
        extra_args="--strip_think"
    fi
    if [[ "$STRIP_THINK" == "1" ]]; then
        python3 "${CODE_ROOT}/eval/eval.py" \
            --answer "$RES_FILE" \
            --dataset "$file_name_no_ext" \
            --strip_think \
            2>&1 | grep -E "50 score:|All score:" > /tmp/scores
    else
        python3 "${CODE_ROOT}/eval/eval.py" \
            --answer "$RES_FILE" \
            --dataset "$file_name_no_ext" \
            2>&1 | grep -E "50 score:|All score:" > /tmp/scores
    fi

    if [[ -s /tmp/scores ]]; then
        echo "Result file: $RES_FILE" > "$F1_FILE"
        cat /tmp/scores >> "$F1_FILE"
        echo "" >> "$F1_FILE"
        echo "F1 score saved to: $F1_FILE"
        echo -e "\n\n======================================"
        echo "Model: $MODEL"
        echo "LLM URL: $LLM_URL"
        echo ""
        cat "$F1_FILE"
        echo "======================================"
    else
        echo "Warning: No valid F1 score generated!  Please run eval.py directly to check where the error is."
        touch "$F1_FILE"
    fi
  
done

echo -e "\n\n======================================"
echo "All Files Processed (Online Inference)!"
echo "======================================"

