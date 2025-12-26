#!/bin/bash

load_config() {
    local config_file
    config_file="$(dirname "${BASH_SOURCE[0]}")/config.properties"
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file '$config_file' not found!" >&2
        exit 1
    fi

    while IFS= read -r line; do
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [[ -z "$line" || "$line" == \#* ]] && continue

        if [[ "$line" == export\ * ]]; then
            rest="${line#export }"
            eval "export $rest"
        else
            if [[ "$line" == *=* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                eval "$key=\$value"
            else
                echo "WARNING: Invalid config line (no '=' found): $line" >&2
            fi
        fi
    done < "$config_file"
}

start_server() {
    [[ -z "$model" ]] && { echo "ERROR: model not set in config.properties" >&2; exit 1; }

    if [[ "$ucm_enable" == "true" ]]; then
        [[ -z "$ucm_config_yaml_path" ]] && {
            echo "ERROR: ucm_config_yaml_path not set but ucm_enable=1" >&2
            exit 1
        }
        LOG_FILE="vllm_ucm.log"
    else
        LOG_FILE="vllm.log"
    fi

    echo ""
    echo "===== vllm server configuration ====="
    echo "model                    = $model"
    echo "served_model_name        = ${served_model_name:-<default>}"
    echo "tp_size                  = $tp_size"
    echo "dp_size                  = $dp_size"
    echo "pp_size                  = $pp_size"
    echo "enable_expert_parallel   = $enable_expert_parallel"
    echo "max_model_len            = $max_model_len"
    echo "max_num_batched_tokens   = $max_num_batch_tokens"
    echo "max_num_seqs             = $max_num_seqs"
    echo "block_size               = $block_size"
    echo "gpu_memory_utilization   = $gpu_memory_utilization"
    echo "quantization             = $quantization"
    echo "server_host              = $server_host"
    echo "server_port              = $server_port"
    echo "distributed_backend      = $distributed_executor_backend"
    echo "enable_prefix_caching    = $enable_prefix_caching"
    echo "async_scheduling         = $async_scheduling"
    echo "graph_mode               = $graph_mode"
    if [[ "$ucm_enable" == "true" ]]; then
        echo "ucm_config_file          = $ucm_config_yaml_path"
    fi
    echo "log_file                 = $LOG_FILE"
    echo "====================================="
    echo ""

    CMD=(
        vllm serve "$model"
        --max-model-len "$max_model_len"
        --tensor-parallel-size "$tp_size"
        --data-parallel-size "$dp_size"
        --pipeline-parallel-size "$pp_size"
        --gpu-memory-utilization "$gpu_memory_utilization"
        --trust-remote-code
        --max-num-batched-tokens "$max_num_batch_tokens"
        --max-num-seqs "$max_num_seqs"
        --block-size "$block_size"
        --host "$server_host"
        --port "$server_port"
        --distributed-executor-backend "$distributed_executor_backend"
    )

    if [[ "$enable_expert_parallel" == "true" ]]; then CMD+=("--enable-expert-parallel"); fi

    if [[ "$enable_prefix_caching" == "false" ]]; then CMD+=("--no-enable-prefix-caching"); fi

    if [[ "$async_scheduling" == "true" ]]; then CMD+=("--async-scheduling"); fi

    [[ -n "$served_model_name" ]] && CMD+=("--served-model-name" "$served_model_name")
    
    [[ "$quantization" != "NONE" ]] && CMD+=("--quantization" "$quantization")

    if [[ "$ucm_enable" == "true" ]]; then
        KV_CONFIG_JSON="{
            \"kv_connector\":\"UCMConnector\",
            \"kv_connector_module_path\":\"ucm.integration.vllm.ucm_connector\",
            \"kv_role\":\"kv_both\",
            \"kv_connector_extra_config\":{\"UCM_CONFIG_FILE\":\"$ucm_config_yaml_path\"}
        }"
        CMD+=("--kv-transfer-config" "$KV_CONFIG_JSON")
    fi

    if [[ -n "$graph_mode" ]]; then 
        COMPILATION_CONFIG='{"cudagraph_mode":"'"$graph_mode"'"}'
        CMD+=("--compilation-config" "$COMPILATION_CONFIG")
    fi

    echo "Executing command: ${CMD[*]}"
    echo ""

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
}

load_config
start_server