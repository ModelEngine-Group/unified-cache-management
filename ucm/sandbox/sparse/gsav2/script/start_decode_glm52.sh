#!/bin/bash
set -e
set -u
LOG_DIR="./vllm_log"

if [ ! -d "$LOG_DIR" ]; then
    echo "目录不存在，创建中..."
    mkdir "$LOG_DIR"
    echo "创建完成"
else
    echo "目录已存在"
fi

echo "$(date): 开始启动 vLLM 服务..."

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"/* 2>/dev/null
else
    mkdir -p "$LOG_DIR"
fi

KVPre_DIR="gsa_log/GLM/KVpre"
if [ -d "$KVPre_DIR" ]; then
    rm -rf "$KVPre_DIR"/* 2>/dev/null
else
    mkdir -p "$KVPre_DIR"
fi

UCM_DIR="gsa_log/GLM/pyLog"
if [ -d "$UCM_DIR" ]; then
    rm -rf "$UCM_DIR"/* 2>/dev/null
else
    mkdir -p "$UCM_DIR"
fi

UCM_DIR="gsa_log/GLM/KVpre"
if [ -d "$UCM_DIR" ]; then
    rm -rf "$UCM_DIR"/* 2>/dev/null
else
    mkdir -p "$UCM_DIR"
fi

# ================================
# 🔧 基础参数配置
# ================================
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="./mooncake.json"
export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:$CODE_ROOT/ucm:$CODE_ROOT/ucm/pmr/pmrSearch/lib:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=0
export debug_save_search_result=false
export PYTHONHASHSEED=123456
export SPDLOG_LEVEL=warn
export VLLM_LOG_PATH='./vllm_log/'

export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_VERSION=0.23.0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HCCL_CONNECT_TIMEOUT=1800

export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=60000
export ASCEND_TRANSFER_TIMEOUT=60000
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=960
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ENABLE_UCM_PATCH=1

export PYTHONHASHSEED=0
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_ENABLE_USE_FABRIC_MEM=1  # A3 必须开启


# 模型配置
MODEL_PATH="/mnt/home_old/models/GLM-5.2-w4a8c8/"
MODEL_NAME="glm-5"

# vLLM 服务配置
MAX_MODEL_LEN=131072
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=16
GPU_MEMORY_UTILIZATION=0.93
SERVER_PORT=8063
BLOCK_SIZE=128


# 检查模型目录是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在！ ($MODEL_PATH)"
    exit 1
fi

additional_config='{"pa_shape_list": [48,64,72,80], "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "enable_mlapo": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}'
compilation_config='{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 96]}'
ucm_config='{"kv_connector": "MultiConnector", "kv_role": "kv_consumer", "kv_connector_extra_config": {"connectors": [
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_consumer",
                    "kv_port": 23697,
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": 16},
                        "decode": {"dp_size": 1, "tp_size": 16}
                    }
                },
                {
                    "kv_connector": "UCMConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
                    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "./ucm_config_decode.yaml"}
                }
            ]}}'

speculative_config='{"method": "mtp", "num_speculative_tokens": 3}'
LOG_FILE="${VLLM_LOG_PATH}/pmr.log"
script -q -f "$LOG_FILE" -c "
    vllm serve \"$MODEL_PATH\" \
        --served-model-name \"$MODEL_NAME\" \
        --trust-remote-code \
        --distributed-executor-backend mp \
        --data-parallel-size \"$DATA_PARALLEL_SIZE\" \
        --tensor-parallel-size \"$TENSOR_PARALLEL_SIZE\" \
        --max-num-seqs 16 \
        --max-model-len \"$MAX_MODEL_LEN\" \
        --max-num-batched-tokens 64 \
        --additional-config '$additional_config' \
        --port \"$SERVER_PORT\" \
        --block-size \"$BLOCK_SIZE\" \
        --gpu-memory-utilization \"$GPU_MEMORY_UTILIZATION\" \
        --kv-transfer-config '$ucm_config'\
        --enable-expert-parallel\
        --quantization ascend\
        --enable-prefix-caching \
        --no-async-scheduling \
        --compilation-config '$compilation_config' \
        --speculative-config '$speculative_config' \
        >& ./gsa_log/dsa_server_$(date '+%Y%m%d_%H%M%S').log
        "

# 日志处理模块
function back_up_logs() {
  # 创建日志归档目录
  BACKUP_DIR="${VLLM_LOG_PATH}/log_backup"
  mkdir -p "$BACKUP_DIR"
  BACKUP_FILE="./backup_files.sh"

  # 查找并终止旧进程
  backup_pid=$(pgrep -f "$(basename "$BACKUP_FILE")")
  if [ -n "$backup_pid" ]; then
      kill -9 "$backup_pid" && sleep 1
      echo "[INFO] 已终止旧备份进程 PID: $backup_pid"
  fi

  # 启动新进程
  chmod +x $BACKUP_FILE
  nohup "$BACKUP_FILE" >>"$BACKUP_DIR/log_backup.log" 2>&1 &
  new_pid=$!
  echo "[INFO] 启动新备份进程 PID: $new_pid"
}

back_up_logs