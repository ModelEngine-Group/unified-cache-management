#!/bin/bash

load_config() {
    local config_file
    config_file="$(dirname "${BASH_SOURCE[0]}")/config.properties"
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file '$config_file' not found!" >&2
        exit 1
    fi

    while IFS='=' read -r key value; do
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        [[ -z "$key" || "$key" == \#* ]] && continue

        export "$key"="$value"
    done < <(grep -v '^\s*#' "$config_file" | grep -v '^\s*$')
}

start_server() {
    [[ -z "$MODEL" ]] && { echo "ERROR: MODEL not set in config.properties" >&2; exit 1; }

    if [[ "$UCM_ENABLE" == "1" ]]; then
        [[ -z "$UCM_CONFIG_YAML_PATH" ]] && {
            echo "ERROR: UCM_CONFIG_YAML_PATH not set but UCM_ENABLE=1" >&2
            exit 1
        }
        LOG_FILE="vllm_ucm.log"
    else
        LOG_FILE="vllm.log"
    fi

    echo ""
    echo "===== vLLM Server Configuration ====="
    echo "MODEL                    = $MODEL"
    echo "SERVED_MODEL_NAME        = ${SERVED_MODEL_NAME:-<default>}"
    echo "TP_SIZE                  = $TP_SIZE"
    echo "DP_SIZE                  = $DP_SIZE"
    echo "PP_SIZE                  = $PP_SIZE"
    echo "ENABLE_EXPERT_PARALLEL   = $ENABLE_EXPERT_PARALLEL"
    echo "MAX_MODEL_LEN            = $MAX_MODEL_LEN"
    echo "MAX_NUM_BATCHED_TOKENS   = $MAX_NUM_BATCH_TOKENS"
    echo "MAX_NUM_SEQS             = $MAX_NUM_SEQS"
    echo "BLOCK_SIZE               = $BLOCK_SIZE"
    echo "GPU_MEMORY_UTILIZATION   = $GPU_MEMORY_UTILIZATION"
    echo "QUANTIZATION             = $QUANTIZATION"
    echo "SERVER_HOST              = $SERVER_HOST"
    echo "SERVER_PORT              = $SERVER_PORT"
    echo "DISTRIBUTED_BACKEND      = $DISTRIBUTED_EXECUTOR_BACKEND"
    echo "ENABLE_PREFIX_CACHING    = $ENABLE_PREFIX_CACHING"
    echo "ASYNC_SCHEDULING         = $ASYNC_SCHEDULING"
    echo "GRAPH_MODE               = $GRAPH_MODE"
    if [[ "$UCM_ENABLE" == "1" ]]; then
        echo "UCM_CONFIG_FILE          = $UCM_CONFIG_YAML_PATH"
    fi
    echo "LOG_FILE                 = $LOG_FILE"
    echo "====================================="
    echo ""

    CMD=(
        vllm serve "$MODEL"
        --max-model-len "$MAX_MODEL_LEN"
        --tensor-parallel-size "$TP_SIZE"
        --data-parallel-size "$DP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --trust-remote-code
        --max-num-batched-tokens "$MAX_NUM_BATCH_TOKENS"
        --max-num-seqs "$MAX_NUM_SEQS"
        --block-size "$BLOCK_SIZE"
        --host "$SERVER_HOST"
        --port "$SERVER_PORT"
        --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND"
    )

    if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then CMD+=("--enable-expert-parallel"); fi

    if [[ "$ENABLE_PREFIX_CACHING" == "0" ]]; then CMD+=("--no-enable-prefix-caching"); fi

    if [[ "$ASYNC_SCHEDULING" == "1" ]]; then CMD+=("--async-scheduling"); fi

    [[ -n "$SERVED_MODEL_NAME" ]] && CMD+=("--served-model-name" "$SERVED_MODEL_NAME")
    
    [[ "$QUANTIZATION" != "None" ]] && CMD+=("--quantization" "$QUANTIZATION")

    if [[ "$UCM_ENABLE" == "1" ]]; then
        KV_CONFIG_JSON="{
            \"kv_connector\":\"UCMConnector\",
            \"kv_connector_module_path\":\"ucm.integration.vllm.ucm_connector\",
            \"kv_role\":\"kv_both\",
            \"kv_connector_extra_config\":{\"UCM_CONFIG_FILE\":\"$UCM_CONFIG_YAML_PATH\"}
        }"
        CMD+=("--kv-transfer-config" "$KV_CONFIG_JSON")
    fi

    if [[ -n "$GRAPH_MODE" ]]; then 
        COMPILATION_CONFIG='{"cudagraph_mode":"'"$GRAPH_MODE"'"}'
        CMD+=("--compilation-config" "$COMPILATION_CONFIG")
    fi

    echo "Executing command: ${CMD[*]}"
    echo ""

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
}

load_config
start_server