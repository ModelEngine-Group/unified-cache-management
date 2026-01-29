#!/bin/bash
echo $CONFIG_FILE
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

start_server() {
    [[ -z "$model" ]] && { echo "ERROR: model not set in config.properties" >&2; exit 1; }

    if [[ "$ucm_enable" == "true" ]]; then
        [[ -z "$ucm_config_yaml_path" ]] && {
            echo "ERROR: ucm_config_yaml_path not set but ucm_enable=true" >&2
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
    echo "max_num_batched_tokens   = $max_num_batched_tokens"
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
        --host "$server_host"
        --port "$server_port"
        --distributed-executor-backend "$distributed_executor_backend"
    )

    # --- Optional numeric/string params ---
    if [[ -n "$block_size" ]]; then CMD+=("--block-size" "$block_size"); fi
    if [[ -n $max_num_batched_tokens ]]; then CMD+=("--max-num-batched-tokens" "$max_num_batched_tokens"); fi
    if [[ -n $max_num_seqs ]]; then CMD+=("--max-num-seqs" "$max_num_seqs"); fi
    if [[ -n "$seed" ]]; then CMD+=("--seed" "$seed"); fi
    if [[ -n "$served_model_name" ]]; then CMD+=("--served-model-name" "$served_model_name"); fi
    if [[ -n "$quantization" ]] && [[ "$quantization" != "NONE" ]]; then CMD+=("--quantization" "$quantization"); fi
    if [[ -n "$graph_mode" ]]; then 
        COMPILATION_CONFIG='{"cudagraph_mode":"'"$graph_mode"'"}'
        CMD+=("--compilation-config" "$COMPILATION_CONFIG")
    fi
    
    # --- Boolean flags ---
    if [[ "$distributed_executor_backend" == "ray" ]] && [[ "$dp_size" -gt 1 ]]; then 
        CMD+=("--data-parallel-backend" "ray")
        CMD+=("--data-parallel-size-local" "$((dp_size / node_num))");
    fi
    if [[ "$async_scheduling" == "true" ]]; then CMD+=("--async-scheduling"); fi
    if [[ "$enable_expert_parallel" == "true" ]]; then CMD+=("--enable-expert-parallel"); fi
    if [[ "$enable_prefix_caching" == "false" ]]; then CMD+=("--no-enable-prefix-caching"); fi

    # --- Advanced configs (JSON) ---
    if [[ "$enable_speculative_decoding" == "true" ]]; then
        SPECULATIVE_CONFIG='{"model":"'"$speculative_decode_model"'", "num_speculative_tokens": "'"$num_speculative_tokens"'", "method":"'"$speculative_decode_method"'"}'
        CMD+=("--speculative-config" "$SPECULATIVE_CONFIG")
    fi

    ADDITIONAL_CONFIG="{"
    SEP=""
    if [[ -n "$enable_ascend_scheduler" ]]; then
        ADDITIONAL_CONFIG+="${SEP}\"ascend_scheduler_config\":{\"enabled\":$enable_ascend_scheduler}"
        SEP=","
    fi
    if [[ -n "$enable_torchair_graph" ]]; then
        ADDITIONAL_CONFIG+="${SEP}\"torchair_graph_config\":{\"enabled\":$enable_torchair_graph}"
        SEP=","
    fi
    ADDITIONAL_CONFIG+="}"
    if [[ "$ADDITIONAL_CONFIG" != "{}" ]]; then CMD+=("--additional-config" "$ADDITIONAL_CONFIG"); fi

    if [[ "$ucm_enable" == "true" ]]; then
        KV_CONFIG_JSON="{
            \"kv_connector\":\"UCMConnector\",
            \"kv_connector_module_path\":\"ucm.integration.vllm.ucm_connector\",
            \"kv_role\":\"kv_both\",
            \"kv_connector_extra_config\":{\"UCM_CONFIG_FILE\":\"$ucm_config_yaml_path\"}
        }"
        CMD+=("--kv-transfer-config" "$KV_CONFIG_JSON")
    fi

    echo "Executing command: ${CMD[*]}"
    echo ""

    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
}

load_config
start_server