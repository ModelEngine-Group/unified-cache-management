#!/usr/bin/env bash
# ============================================================================
# build-kv-link.sh — 建链断链测试统一编译脚本（CPU / NPU 双模式）
#
# 用途：按 --mode 编译对应链路的全部组件（kv client + testTool），两模式共用
# 本脚本，差异只在编译参数处（运行时环境 / provider 宏 / CANN 路径 / 构建类型）：
#   cpu（缺省）CPU-over-RoCE 链路：
#     1) kv client：BUILD_KV_CLIENT_PROVIDER_ROCE=ON（RoCE provider 编入
#        libkv_transport.so，需要 libibverbs），RUNTIME_ENVIRONMENT=simu，Debug
#     2) testTool：链接上一步 build-kv 目录下的预构建 kv client 库，Debug
#   npu：HCCP KV-over-RoCE 链路（NPU 数据面，详细 ABI 契约与算子库要求见
#       kv_semantics/docs/guide/hccp-kv-over-roce-provider.md）：
#     1) kv client：BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON，
#        RUNTIME_ENVIRONMENT=ascend，KV_CLIENT_CANN_ROOT=$ASCEND_HOME_PATH，Release
#     2) testTool：经 CMAKE_CXX_FLAGS 注入 TESTTOOL_ENABLE_ASCEND 与 CANN 头文件
#        路径（真 NPU 内存分配——HCCP 的 RegisterMemory 只收设备内存），Release
#
# 适用代码：含对应 provider 构建开关的分支（脚本前置检测，不含即报错退出，
# 防止在不含该代码的分支上误跑而静默编出无对应 provider 的产物）。
#
# 用法（在 testTool 目录下执行；从任意目录发起亦可，路径以脚本自身位置推算）：
#   bash scripts/build-kv-link.sh                # 等价 --mode cpu
#   bash scripts/build-kv-link.sh --mode cpu     # CPU-over-RoCE 链路
#   bash scripts/build-kv-link.sh --mode npu     # HCCP KV-over-RoCE 链路（NPU 机器 + CANN）
#
# 前置依赖（缺失时脚本前置检测报错退出，按提示安装/准备后重跑）：
#   共同：cmake(>=3.18) / g++(C++17) / python3（metrics 代码生成）
#   cpu 额外：libibverbs-dev（RoCE）
#   npu 额外：CANN（ascend-toolkit）已安装并 source set_env.sh（$ASCEND_HOME_PATH 有值）
#
# 两模式共用 build-kv 与 testTool build 目录（CMake 检测到参数变化自动重新配置）；
# 切换模式后如遇缓存导致的配置异常，删除 build-kv 与 build 目录后重跑即可。
# ============================================================================

# ---- 解释器检测（须在 set 之前：Ubuntu 的 sh 是 dash，不支持 -o pipefail 与
# BASH_SOURCE；用 sh 执行时在此给出明确提示，而不是晦涩的 "Illegal option" 报错）----
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[FAIL] 本脚本需用 bash 执行（当前解释器不支持所需特性，如 Ubuntu 的 sh）"
    echo "       请改用：bash $0"
    exit 1
fi

set -euo pipefail

# ---- 路径推算：以脚本自身位置为基准（scripts 位于 kv_semantics/testTool/scripts，
# 从任意目录发起执行均可，不依赖当前工作目录）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTTOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"             # kv_semantics/testTool
KV_SEMANTICS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # kv_semantics

# ---- 模式解析：--mode cpu（缺省）/ --mode npu，支持 "--mode xxx" 与 "--mode=xxx"
# 两种写法；非法取值即报错退出 ----
MODE="cpu"
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            if [ $# -lt 2 ]; then
                echo "[FAIL] --mode 缺少取值（cpu / npu）"
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#--mode=}"
            shift
            ;;
        *)
            echo "[FAIL] 无法识别的参数：$1（仅支持 --mode cpu|npu）"
            exit 1
            ;;
    esac
done
case "${MODE}" in
    cpu|npu) ;;
    *)
        echo "[FAIL] 无效模式：${MODE}（仅支持 cpu / npu）"
        exit 1
        ;;
esac

# ---- 依赖检测：缺失即报错退出（不代装，避免 root/sudo 假设），按提示安装后重跑 ----
MISSING=""
for cmd in cmake g++ python3; do
    command -v "${cmd}" > /dev/null 2>&1 || MISSING="${MISSING} ${cmd}"
done
if [ "${MODE}" = "cpu" ]; then
    # libibverbs 按头文件探测（跨发行版通用；CMake 配置阶段同样要求 infiniband/verbs.h）
    [ -f /usr/include/infiniband/verbs.h ] || MISSING="${MISSING} libibverbs-dev"
fi
if [ -n "${MISSING}" ]; then
    echo "[FAIL] 缺少依赖：${MISSING}"
    echo "       Ubuntu/Debian 安装示例：sudo apt-get install -y cmake g++ python3 libibverbs-dev"
    exit 1
fi

# ---- 分支能力检测：当前 checkout 的 kv_semantics 必须含本模式 provider 的构建开关 ----
if [ "${MODE}" = "cpu" ]; then
    if ! grep -q "BUILD_KV_CLIENT_PROVIDER_ROCE" "${KV_SEMANTICS_ROOT}/CMakeLists.txt"; then
        echo "[FAIL] 当前代码不含 RoCE provider 构建开关（${KV_SEMANTICS_ROOT}/CMakeLists.txt）"
        echo "       请切换到含 RoCE provider 的分支后重跑"
        exit 1
    fi
else
    if ! grep -q "BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE" "${KV_SEMANTICS_ROOT}/CMakeLists.txt"; then
        echo "[FAIL] 当前代码不含 HCCP provider 构建开关（${KV_SEMANTICS_ROOT}/CMakeLists.txt）"
        echo "       请切换到含 HCCP provider 的分支（feature_26A_kvclient / PR #1698 合入分支）后重跑"
        exit 1
    fi
fi

# ---- NPU 专属前置检测：CANN 环境就绪（$ASCEND_HOME_PATH 有值，且其下按 kv client
# CMake 同样的候选布局能找到 acl 头文件；libascendcl 的六种库布局探测交给 CMake
# 配置期做，其报错读法见 2 号 SOP §6。此处同时确定头文件目录，供步骤 2 的
# -I 注入使用——两种布局取实际命中者）----
if [ "${MODE}" = "npu" ]; then
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        echo "[FAIL] 未检测到 CANN 环境（\$ASCEND_HOME_PATH 为空）"
        echo "       请先安装 ascend-toolkit 并执行 source <CANN>/set_env.sh 后重跑"
        exit 1
    fi
    CANN_INC_DIR="${ASCEND_HOME_PATH}/include"
    if [ ! -f "${CANN_INC_DIR}/acl/acl.h" ]; then
        if [ -f "${ASCEND_HOME_PATH}/aarch64-linux/include/acl/acl.h" ]; then
            CANN_INC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/include"
        else
            echo "[FAIL] \${ASCEND_HOME_PATH} 下未找到 acl/acl.h：${ASCEND_HOME_PATH}"
            echo "       请确认 CANN 安装完整（include 与 aarch64-linux/include 两种布局均未命中）"
            exit 1
        fi
    fi
fi

# ---- 步骤 1/2：构建 kv client（build-kv 目录名为 testTool 构建脚本的既定引用路径
# ../build-kv/src，不可改名；--target kv_client 只构建所需目标——两种 provider 的
# 源文件均已编入 libkv_transport.so 的源列表，随 kv_client 目标的依赖一并构建）----
if [ "${MODE}" = "cpu" ]; then
    echo "===== [1/2] 构建 kv client（CPU-over-RoCE：RUNTIME_ENVIRONMENT=simu，BUILD_KV_CLIENT_PROVIDER_ROCE=ON）====="
    cmake -S "${KV_SEMANTICS_ROOT}" -B "${KV_SEMANTICS_ROOT}/build-kv" \
        -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug -DBUILD_KV_CLIENT_PROVIDER_ROCE=ON
    cmake --build "${KV_SEMANTICS_ROOT}/build-kv" --target kv_client -j "$(nproc)"
else
    echo "===== [1/2] 构建 kv client（HCCP KV-over-RoCE：RUNTIME_ENVIRONMENT=ascend，BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON）====="
    cmake -S "${KV_SEMANTICS_ROOT}" -B "${KV_SEMANTICS_ROOT}/build-kv" \
        -DRUNTIME_ENVIRONMENT=ascend -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON \
        -DKV_CLIENT_CANN_ROOT="${ASCEND_HOME_PATH}"
    cmake --build "${KV_SEMANTICS_ROOT}/build-kv" --target kv_client -j "$(nproc)"
fi

# ---- 步骤 2/2：构建 testTool（链接 build-kv 预构建产物；首次配置需联网拉取 fmt/spdlog）。
# npu 模式经 CMAKE_CXX_FLAGS 注入 TESTTOOL_ENABLE_ASCEND 与 CANN 头文件路径——
# testTool 的 CMakeLists 目前无该宏的正式开关入口（待后续补 option），沿用注入方式；
# 真 NPU 内存分配为 HCCP 链路建链验证的必备条件（RegisterMemory 只收设备内存）----
if [ "${MODE}" = "cpu" ]; then
    echo "===== [2/2] 构建 testTool ====="
    cmake -S "${TESTTOOL_ROOT}" -B "${TESTTOOL_ROOT}/build" -DCMAKE_BUILD_TYPE=Debug
    cmake --build "${TESTTOOL_ROOT}/build" -j "$(nproc)"
else
    echo "===== [2/2] 构建 testTool（TESTTOOL_ENABLE_ASCEND：真 NPU 内存分配）====="
    cmake -S "${TESTTOOL_ROOT}" -B "${TESTTOOL_ROOT}/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-DTESTTOOL_ENABLE_ASCEND -I${CANN_INC_DIR}"
    cmake --build "${TESTTOOL_ROOT}/build" -j "$(nproc)"
fi

# ---- 结果提示：产物路径与下一步运行引导 ----
echo "===== 构建完成（mode=${MODE}）====="
echo "kv client 库：${KV_SEMANTICS_ROOT}/build-kv/src/libkv_client.so（及 libkv_transport.so）"
echo "testTool    ：${TESTTOOL_ROOT}/build/testTool"
echo
if [ "${MODE}" = "cpu" ]; then
    echo "下一步（CPU 建链断链验证）："
    echo "  1. kv client 配置传输段改 CPU-over-RoCE：transport.provider_type=CPU；"
    echo "     RoCE 旋钮键名 transport.roce_*（完整模板已内置 conf/kv_client_real.conf，"
    echo "     取消注释按环境改值即可；对端 KV store 侧同名键前缀为 asu_roce_*），"
    echo "     参考 ${KV_SEMANTICS_ROOT}/docs/guide/roce-provider.md；对端 KV store 需先就绪"
    echo "  2. 建链验证（在 ${TESTTOOL_ROOT} 目录下执行）："
    echo "     ./build/testTool conf/test_config_onlyconnect.conf <kv client conf>"
    echo "     建链 + region 注册完成后 Ctrl+C 断链退出（日志确认 Ctrl+C received, exiting）"
else
    echo "下一步（NPU 建链断链验证，详见 sop/2. NPU建链断链-SOP.md）："
    echo "  1. kv client 配置：transport.provider_type=HCCP_KV_OVER_ROCE；"
    echo "     transport.hccp_kv_roce_lib_path 指向算子库主体绝对路径；对端 IP 填在"
    echo "     asu_info.N 的 local.comm_id 字段（对端 KV store 以 HCCP 后端就绪）"
    echo "  2. 算子库三个 so 必须同目录：libhccp_kv_over_roce.so + librdma_kv_send_kernel.so"
    echo "     + librdma_kv_wait_kernel.so（算子库主体在同目录加载两个内核，分目录放置无法找到）"
    echo "  3. 建链验证（在 ${TESTTOOL_ROOT} 目录下执行，test_config_onlyconnect.conf"
    echo "     的 buffertype 设 npu）："
    echo "     KV_LOG_LEVEL=debug ./build/testTool conf/test_config_onlyconnect.conf <kv client conf>"
fi
