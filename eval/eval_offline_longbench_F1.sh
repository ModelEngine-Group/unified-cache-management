#!/bin/bash

# Check and auto-install required Python packages
REQUIRED_PACKAGES=("fuzzywuzzy" "jieba" "rouge" )
for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! python3 -c "import $pkg" 2>/dev/null; then
    echo "❌ $pkg not found, installing..."
    pip install "$pkg" --upgrade 2>/dev/null && echo "✅ $pkg installed successfully" || echo "❌ Failed to install $pkg (run 'pip3 install $pkg' manually)"
  fi
done

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
export ENABLE_SPARSE=true
export VLLM_HASH_ATTENTION=1

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TIMESTAMP

MODEL_PATH=""
UCM_SPARSE_CONFIG=""
TEST_DATA_DIR=""
STRIP_THINK=0
BATCH_SIZE=20

show_help() {
    echo "Usage: bash run.sh [options]"
    echo "Options:"
    echo "  --model PATH            Path to model"
    echo "  --config PATH           Path to UCM sparse config"
    echo "  --data PATH             Path to test data directory"
    echo "  --strip_think 0|1       Whether to apply --strip_think"
    echo "  --batch INT             Batch size"
    echo "  --resume                Resume from existing result file"
    echo "  -h, --help              Show help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            UCM_SPARSE_CONFIG="$2"
            shift 2
            ;;
        --data)
            TEST_DATA_DIR="$2"
            shift 2
            ;;
        --strip_think)
            STRIP_THINK="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
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

# Set default values
MODEL_PATH="${MODEL_PATH:-/home/models/Qwen2.5-14B-Instruct/}"
UCM_SPARSE_CONFIG="${UCM_SPARSE_CONFIG:-${CODE_ROOT}/eval/ucm_sparse_config_esa.json}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${CODE_ROOT}/eval/data/longbench}"
BATCH_SIZE="${BATCH_SIZE:-20}"

echo "------------- Final Arguments -------------"
echo "MODEL_PATH       = $MODEL_PATH"
echo "UCM_SPARSE_CONFIG= $UCM_SPARSE_CONFIG"
echo "TEST_DATA_DIR    = $TEST_DATA_DIR"
echo "STRIP_THINK      = $STRIP_THINK"
echo "BATCH_SIZE       = $BATCH_SIZE"
echo "RESUME_FLAG      = $RESUME_FLAG"
echo "--------------------------------------------"




# set vllm version
export VLLM_VERSION="0.9.2"
export VLLM_USE_V1="1"

# Model 
export MODEL_PATH
MODEL_NAME=$(basename "$MODEL_PATH")

# Dataset and storage path
STORAGE_BACKENDS="${CODE_ROOT}/ucm_kv_cache/${MODEL_NAME}"
export STORAGE_BACKENDS

SAVE_PATH="${CODE_ROOT}/eval/ucm_sparse_predictions"
DATASET="LongBench" 
DATASET_SAVE_DIR="${SAVE_PATH}/${DATASET}"

mkdir -p "$STORAGE_BACKENDS" "$DATASET_SAVE_DIR" || { echo "Failed to create dirs!"; exit 1; }



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
    echo "❌ No valid data files found for '$DATASET'!"
    exit 1
fi

echo -e "\nFound ${#EXISTING_FILES[@]} data files for $DATASET:"
for file in "${EXISTING_FILES[@]}"; do
    rel_path="${file#${BASE_DATA_DIR}/${DATASET}/}"
    echo "  - $rel_path"
done

UCM_CONFIG_NAME=$(basename "$UCM_SPARSE_CONFIG") 
UCM_CONFIG_NAME_NO_EXT="${UCM_CONFIG_NAME%.*}" 

for DATASET_FLIE in "${EXISTING_FILES[@]}"; do
    filename=$(basename "$DATASET_FLIE")
    file_name_no_ext="${filename%.*}"
    export DATASET_FLIE
    
    RES_FILE="${DATASET_SAVE_DIR}/${MODEL_NAME}_${file_name_no_ext}_${UCM_CONFIG_NAME_NO_EXT}_offline_bs${BATCH_SIZE}_${TIMESTAMP}.jsonl"
    export RES_FILE

    export UCM_SPARSE_CONFIG
    export BATCH_SIZE
    export RESUME_FLAG
    echo -e "\n======================================"
    echo "Executed model: $MODEL_NAME"
    echo "Using Config: $UCM_SPARSE_CONFIG"
    echo "======================================"

    python3 "${CODE_ROOT}/eval/offline_inference_longbench_F1.py" \

    if [[ ! -f "$RES_FILE" ]]; then
        echo "Warning: test finished but result file not found!"
        continue
    fi

    echo -e "\nCalculating F1 score..."
    F1_FILE="${RES_FILE}.f1.txt"
    > /tmp/scores

    extra_args=""
    if [[ "$STRIP_THINK" == "1" ]]; then
        extra_args="--strip_think"
    fi
    python3 "${CODE_ROOT}/eval/eval.py" \
        --answer "$RES_FILE" \
        --dataset "$file_name_no_ext" \
        $extra_args \
        2>&1 | grep -E "50 score:|All score:" > /tmp/scores

    if [[ -s /tmp/scores ]]; then
        echo "Result file: $RES_FILE" > "$F1_FILE"
        cat /tmp/scores >> "$F1_FILE"
        echo "" >> "$F1_FILE"
        echo "F1 score saved to: $F1_FILE"
        echo -e "\n\n======================================"
        echo ""
        cat "$UCM_SPARSE_CONFIG"
        echo
        cat "$F1_FILE"
        echo "======================================"
    else
        echo "Warning: No valid F1 score generated!  Please run eval.py directly to check where the error is."
        touch "$F1_FILE"
    fi
  
done
rm -rf ${STORAGE_BACKENDS}