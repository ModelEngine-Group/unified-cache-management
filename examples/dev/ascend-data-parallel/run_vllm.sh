#!/bin/bash

if [[ -z "$NODE" ]]; then
    echo "ERROR: Please set NODE=N before running. N should be 0 for master node; 1,2,3... for workers. Note the IPs and environment variables in the script should be modified accordingly. "
    echo "Usage: NODE=0 ./run_vllm.sh"
    exit 1
fi

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

ensure_ifconfig_installed() {
    if command -v ifconfig >/dev/null 2>&1; then
        return 0
    fi

    echo "ifconfig not found. Attempting to install net-tools..."

    if command -v apt-get >/dev/null 2>&1; then
        echo "Detected apt-get (Debian/Ubuntu). Installing net-tools..."
        sudo apt-get update && sudo apt-get install -y net-tools
    elif command -v yum >/dev/null 2>&1; then
        echo "Detected yum (RHEL/CentOS). Installing net-tools..."
        sudo yum install -y net-tools
    elif command -v dnf >/dev/null 2>&1; then
        echo "Detected dnf (Fedora). Installing net-tools..."
        sudo dnf install -y net-tools
    else
        echo "ERROR: No supported package manager (apt/yum/dnf) found."
        echo "Please install 'net-tools' manually or use a system with 'ip' command."
        exit 1
    fi

    if ! command -v ifconfig >/dev/null 2>&1; then
        echo "ERROR: Failed to install ifconfig. Please check permissions or network."
        exit 1
    fi

    echo "✅ ifconfig is now available."
}

get_interface_by_ip() {
    local target_ip="$1"
    ifconfig | awk -v target="$target_ip" '
        /^[[:alnum:]]/ {
            iface = $1
            sub(/:$/, "", iface)  
        }
        /inet / {
            for (i = 1; i <= NF; i++) {
                gsub(/addr:/, "", $i)
                if ($i == target) {
                    print iface
                    exit
                }
            }
        }
    '
}

start_server() {
    # Ascend environment variables
    if [[ "$NODE" == "0" ]]; then
        export TARGET_IP="$master_ip"
    else
        export TARGET_IP="$worker_ip"
    fi

    IFACE=$(get_interface_by_ip "$TARGET_IP")

    if [[ -z "$IFACE" ]]; then
        echo "WARNING: Could not find interface with IP $TARGET_IP via ifconfig. Falling back to 'eth0'."
        IFACE="eth0"
    else
        echo "✅ Detected interface: $IFACE (bound to IP $TARGET_IP)"
    fi

    export HCCL_IF_IP="$TARGET_IP"
    export HCCL_SOCKET_IFNAME="$IFACE"
    export GLOO_SOCKET_IFNAME="$IFACE"
    export TP_SOCKET_IFNAME="$IFACE"

    # vLLM parameters 
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
    echo "node                     = $NODE"
    echo "master_ip                = $master_ip"
    echo "local_ip                 = $TARGET_IP"
    echo "network_interface        = $IFACE"
    echo "model                    = $model"
    echo "served_model_name        = ${served_model_name:-<default>}"
    echo "tp_size                  = $tp_size"
    echo "dp_size                  = $dp_size"
    echo "pp_size                  = $pp_size"
    echo "dp_size_local            = $dp_size_local"
    echo "dp_start_rank            = $((dp_size_local * NODE))"
    echo "dp_address               = $master_ip"
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
        --data-parallel-size-local "$dp_size_local"
        --data-parallel-start-rank "$((dp_size_local * NODE))"
        --data-parallel-address "$master_ip"
        --data-parallel-rpc-port "$dp_rpc_port"
        --seed "$seed"
        --pipeline-parallel-size "$pp_size"
        --gpu-memory-utilization "$gpu_memory_utilization"
        --trust-remote-code
        --max-num-batched-tokens "$max_num_batch_tokens"
        --max-num-seqs "$max_num_seqs"
        --block-size "$block_size"
        --host "$server_host"
        --port "$server_port"
    )
    if [[ "$NODE" != "0" ]]; then CMD+=("--headless"); fi

    if [[ "$enable_expert_parallel" == "true" ]]; then CMD+=("--enable-expert-parallel"); fi

    if [[ "$enable_prefix_caching" == "false" ]]; then CMD+=("--no-enable-prefix-caching"); fi

    if [[ "$async_scheduling" == "true" ]]; then CMD+=("--async-scheduling"); fi

    [[ -n "$served_model_name" ]] && CMD+=("--served-model-name" "$served_model_name")
    
    [[ "$quantization" != "NONE" ]] && CMD+=("--quantization" "$quantization")

    if [[ -n "$graph_mode" ]]; then 
        COMPILATION_CONFIG='{"cudagraph_mode": "'"$graph_mode"'"}'
        CMD+=("--compilation-config" "$COMPILATION_CONFIG")
    fi

    if [[ -n "$method" ]]; then
        SPECULATIVE_CONFIG='{"num_speculative_tokens": 1, "method":"'"$method"'"}'
        CMD+=("--compilation-config" "$SPECULATIVE_CONFIG")
    fi

    ADDITIONAL_CONFIG='{"ascend_scheduler_config":{"enabled":'"$enable_ascend_scheduler"'},"torchair_graph_config":{"enabled":'"$enable_torchair_graph"'}}'
    CMD+=("--additional-config" "$ADDITIONAL_CONFIG")

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