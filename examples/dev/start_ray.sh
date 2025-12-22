#!/bin/bash

if [[ -z "$NODE" ]]; then
    echo "ERROR: Please set NODE=N before running. N should be 0 for head node; 1,2,3... for workers. Note the IPs and environment variables in the script should be modified accordingly. "
    echo "Usage: NODE=0 ./start_ray.sh"
    exit 1
fi

load_config() {
    config_file="$(dirname "${BASH_SOURCE[0]}")/config.properties"
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file '$config_file' not found!"
        exit 1
    fi

    while IFS='=' read -r key value; do
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        if [[ -z "$key" ]] || [[ "$key" == \#* ]]; then
            continue
        fi

        export "$key"="$value"
    done < <(grep -v '^\s*#' "$config_file" | grep -v '^\s*$')
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

set_node_env(){
    if [[ "$NODE" == "0" ]]; then
        export TARGET_IP="$MASTER_IP"
    else
        export TARGET_IP="$WORKER_IP"
    fi

    IFACE=$(get_interface_by_ip "$TARGET_IP")

    if [[ -z "$IFACE" ]]; then
        echo "WARNING: Could not find interface with IP $TARGET_IP via ifconfig. Falling back to 'eth0'."
        IFACE="eth0"
    else
        echo "✅ Detected interface: $IFACE (bound to IP $TARGET_IP)"
    fi

    export HCCL_IF_IP="$TARGET_IP"
    export NCCL_SOCKET_IFNAME="$IFACE"
    export GLOO_SOCKET_IFNAME="$IFACE"
    export TP_SOCKET_IFNAME="$IFACE"
    export NUM_GPUS=$(($TP_SIZE / $NODE_NUM))

    echo ""
    echo "===== Ray Startup Configuration ======"
    echo "NODE                     = $NODE"
    echo "LOCAL_IP                 = $TARGET_IP"
    if [[ "$NODE" != "0" ]]; then
        echo "MASTER_IP                = $MASTER_IP"
    fi
    echo "NETWORK_INTERFACE        = $IFACE"
    echo "NUM_GPUS (per node)      = $NUM_GPUS"
    echo "CUDA_VISIBLE_DEVICES     = $CUDA_VISIBLE_DEVICES"
    echo "ASCEND_RT_VISIBLE_DEVICES= $ASCEND_RT_VISIBLE_DEVICES"
    echo "======================================"
    echo ""
}

load_config
set_node_env

if [[ "$NODE" == "0" ]]; then
    echo "Starting Ray head node on NODE 0, MASTER_IP: $TARGET_IP"
    ray start --head --num-gpus=$NUM_GPUS --node-ip-address="$TARGET_IP" --port=6379
else
    echo "Starting Ray worker node on NODE $NODE, WORKER_IP=$TARGET_IP, connecting to master at $MASTER_IP"
    ray start --address="$MASTER_IP:6379" --num-gpus=$NUM_GPUS --node-ip-address="$TARGET_IP"
fi