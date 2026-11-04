echo "$(date): 开始启动 vLLM 服务..."

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
# ================================
# 🔧 基础参数配置
# ================================
export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:$CODE_ROOT/ucm:$CODE_ROOT/ucm/pmr/pmrSearch/lib:$PYTHONPATH"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_VERSION=0.23.0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HCCL_CONNECT_TIMEOUT=1800
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ENABLE_UCM_PATCH=1
export ENABLE_GSA_PATCH=1

source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:$CODE_ROOT/ucm:$CODE_ROOT/ucm/pmr/pmrSearch/lib:$PYTHONPATH"
LOG_DIR="./gsa_log/GLM/KVpre"
if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"/* 2>/dev/null
else
    mkdir -p "$LOG_DIR"
fi

UCM_DIR="gsa_log/GLM/pyLog"
if [ -d "$UCM_DIR" ]; then
    rm -rf "$UCM_DIR"/* 2>/dev/null
else
    mkdir -p "$UCM_DIR"
fi

vllm serve /home/l00844759/models/GLM-5.2-w4a8c8/ \
    --host 0.0.0.0 \
    --port 8055 \
    --max-model-len 131072 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-5 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --quantization ascend \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --no-async-scheduling \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 96]}' \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}, "enable_mlapo": true, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true}' \
    --kv-transfer-config '{"kv_connector": "UCMConnector", "kv_connector_module_path": "ucm.integration.vllm.ucm_connector", "kv_role": "kv_both", "kv_connector_extra_config": {"UCM_CONFIG_FILE": "./ucm/sandbox/sparse/gsav2/script/ucm_config_example.yaml"}}' \
    >& ./gsa_log/dsa_acc_server_$(date '+%Y%m%d_%H%M%S').log