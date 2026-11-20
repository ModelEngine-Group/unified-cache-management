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
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_VERSION=0.23.0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export HCCL_CONNECT_TIMEOUT=1800
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ENABLE_UCM_PATCH=1
export ENABLE_GSA_PATCH=1
export UCM_SPARSE_GSA_ENABLE=1

source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
export PYTHONPATH="/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:$CODE_ROOT/ucm:$CODE_ROOT/ucm/pmr/pmrSearch/lib:$PYTHONPATH"

# ---------------------------------------------------------------------------
# GSA 自定义算子库环境（aclnnKvQuantSparseFlashAttentionGsa 等符号由
# libcust_opapi.so 提供，进程内 dlopen 只认启动时的 LD_LIBRARY_PATH，必须
# 在 vllm serve 启动前注入；路径前置，优先于其他 vendor 的同名 libcust_opapi.so，
# 否则首个 decode step 报 "aclnn... not in libopapi.so"）。
# 安装：bash ucm/sandbox/sparse/gsav2/ops/ascend/install.sh（装 GSA-custom_ops*.run
# 到 opp/vendors/gsa 并生成 setenv.bash）。
# ---------------------------------------------------------------------------
GSA_VENDOR_BASE=""
for _base in /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/gsa \
             "${HOME}/Ascend/ascend-toolkit/latest/opp/vendors/gsa"; do
    if [ -d "${_base}/op_api/lib" ]; then GSA_VENDOR_BASE="$_base"; break; fi
done
if [ -n "$GSA_VENDOR_BASE" ]; then
    if [ -f "${GSA_VENDOR_BASE}/setenv.bash" ]; then
        source "${GSA_VENDOR_BASE}/setenv.bash"
    else
        export LD_LIBRARY_PATH="${GSA_VENDOR_BASE}/op_api/lib:${LD_LIBRARY_PATH}"
    fi
    echo "[GSA] custom ops lib: ${GSA_VENDOR_BASE}/op_api/lib"
else
    echo "[GSA][WARN] opp/vendors/gsa 未找到: GSA-custom_ops 未安装, C8 sparse attention 将不可用"
    echo "[GSA][WARN] 请先执行: bash ucm/sandbox/sparse/gsav2/ops/ascend/install.sh 再重启服务"
fi
unset _base

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

python ucm/sandbox/sparse/gsav2/script/serve_gsa_offline.py >& ./gsa_log/gsa_offline_$(date +%Y%m%d_%H%M%S).log