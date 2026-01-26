#!/bin/bash

if [[ -z "$NODE" ]]; then
    echo "ERROR: Please set NODE=N before running. N should be 0 for head node; 1,2,3... for workers. Note the IPs and environment variables in the script should be modified accordingly. "
    echo "Usage: NODE=0 ./start_ray.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

set_node_env(){
    if [[ "$NODE" == "0" ]]; then
        export TARGET_IP="$master_ip"
    else
        export TARGET_IP="$worker_ip"
    fi

    IFACE=$(get_interface_by_ip "$TARGET_IP")

    if [[ -z "$IFACE" ]]; then
        echo ""
        echo "ERROR: Could not find interface with IP $TARGET_IP via ifconfig. Falling back to 'eth0'."
        IFACE="eth0"
    else
        echo "✅ Detected interface: $IFACE (bound to IP $TARGET_IP)"
    fi

    export HCCL_IF_IP="$TARGET_IP"
    export NCCL_SOCKET_IFNAME="$IFACE"
    export GLOO_SOCKET_IFNAME="$IFACE"
    export TP_SOCKET_IFNAME="$IFACE"
    export NUM_GPUS=$((tp_size * dp_size * pp_size / node_num))

    echo ""
    echo "===== ray startup configuration ======"
    echo "node                     = $NODE"
    echo "master_ip                = $master_ip"
    echo "local_ip                 = $TARGET_IP"
    echo "network_interface        = $IFACE"
    echo "num_gpus/npus (per node)      = $NUM_GPUS"
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
    echo "Starting Ray worker node on NODE $NODE, WORKER_IP=$TARGET_IP, connecting to master at $master_ip"
    ray start --address="$master_ip:6379" --num-gpus=$NUM_GPUS --node-ip-address="$TARGET_IP"
fi