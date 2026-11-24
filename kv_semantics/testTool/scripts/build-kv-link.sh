#!/usr/bin/env bash
# ============================================================================
# build-kv-link.sh — 建链断链测试统一编译脚本（CPU / NPU 双模式）
#
# 用途：按 --mode 编译对应链路的全部组件（kv client + testTool），两模式共用
# 本脚本，差异只在编译参数处（运行时环境 / provider 宏 / CANN 路径）：
#   cpu（缺省）CPU-over-RoCE 链路：
#     1) kv client：BUILD_KV_CLIENT_PROVIDER_ROCE=ON 与 BUILD_KV_CLIENT_PROVIDER_FAKE=ON
#        双 provider 编入 libkv_transport.so（RoCE 需要依赖 libibverbs；FAKE 为 CMake
#        默认 ON，此处显式声明），RUNTIME_ENVIRONMENT=simu——FAKE 模拟与 RoCE 真实
#        建链共用一份产物，testTool 侧改 conf 的 provider_type 即切换，无需重编
#     2) testTool：链接上一步 build-kv 目录下的预构建 kv client 库
#   npu：HCCP KV-over-RoCE 链路（NPU 数据面，详细 ABI 契约与算子库要求见
#       kv_semantics/docs/guide/hccp-kv-over-roce-provider.md）：
#     1) kv client：BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON，且
#        BUILD_KV_CLIENT_PROVIDER_ROCE=ON 与 BUILD_KV_CLIENT_PROVIDER_FAKE=ON
#        一并编入（三种 provider 共存一份产物，conf 改 transport.provider_type
#        即在 FAKE 模拟 / CPU-over-RoCE / HCCP 三种链路间切换，无需重编），
#        RUNTIME_ENVIRONMENT=ascend，KV_CLIENT_CANN_ROOT=$ASCEND_HOME_PATH
#     2) testTool：TESTTOOL_ENABLE_ASCEND=ON（CMakeLists 正式开关——编译宏、CANN
#        头文件路径、libascendcl 显式链接三联动；真 NPU 内存分配——HCCP 的
#        RegisterMemory 只收设备内存）
#
# 构建类型：--build-type release（缺省）/ --build-type debug。缺省 Release 使
# 产物随时可直接用于性能测试；需要调试（断点 / 单步）时显式声明 debug。
# 两模式的 kv client 与 testTool 均按该值传 CMAKE_BUILD_TYPE（testTool 侧不传
# BUILD_TYPE 时 CMakeLists 同样走 Release 选项，两种入口行为一致）。
#
# 适用代码：含对应 provider 构建开关的分支（脚本前置检测，不含即报错退出，
# 防止在不含该代码的分支上误跑而静默编出无对应 provider 的产物）。
#
# 用法（在 testTool 目录下执行；从任意目录发起亦可，路径以脚本自身位置推算）：
#   bash scripts/build-kv-link.sh                                # CPU 链路 + Release（缺省）
#   bash scripts/build-kv-link.sh --mode npu                     # HCCP KV-over-RoCE 链路 + Release
#   bash scripts/build-kv-link.sh --mode cpu --build-type debug  # CPU 链路 + Debug（调试构建）
#
# 前置依赖（缺失时脚本前置检测报错退出，按提示安装/准备后重跑）：
#   共同：cmake(>=3.18) / g++(C++17) / python3（metrics 代码生成）
#   cpu 额外：libibverbs-dev（RoCE）
#   npu 额外：CANN toolkit 已安装并 source set_env.sh（$ASCEND_HOME_PATH 有值）；
#             libibverbs-dev（三 provider 共存产物编入 RoCE）
#
# 两模式共用 build-kv 与 testTool build 目录，且各自显式声明全部构建开关（provider 与
# TESTTOOL_ENABLE_ASCEND 均含关闭项）——CMake 缓存变量"不传即沿用旧值"，显式声明使两模式
# 来回切换幂等无残留，无需清理；若因 FetchContent 依赖源损坏等其他缓存问题报错，
# 执行 bash scripts/clean-build.sh 清理后重跑即可。
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

# ---- 参数解析：--mode cpu（缺省）/ --mode npu；--build-type release（缺省）/ debug。
# 两个参数均支持 "--opt xxx" 与 "--opt=xxx" 两种写法；非法取值即报错退出 ----
MODE="cpu"
BUILD_TYPE="release"
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
        --build-type)
            if [ $# -lt 2 ]; then
                echo "[FAIL] --build-type 缺少取值（release / debug）"
                exit 1
            fi
            BUILD_TYPE="$2"
            shift 2
            ;;
        --build-type=*)
            BUILD_TYPE="${1#--build-type=}"
            shift
            ;;
        *)
            echo "[FAIL] 无法识别的参数：$1（仅支持 --mode cpu|npu 与 --build-type release|debug）"
            exit 1
            ;;
    esac
done
# 取值归一化小写后校验（大小写混传同样接受，如 RELEASE / Debug）
MODE="$(printf '%s' "${MODE}" | tr '[:upper:]' '[:lower:]')"
BUILD_TYPE="$(printf '%s' "${BUILD_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${MODE}" in
    cpu|npu) ;;
    *)
        echo "[FAIL] 无效模式：${MODE}（仅支持 cpu / npu）"
        exit 1
        ;;
esac
case "${BUILD_TYPE}" in
    release|debug) ;;
    *)
        echo "[FAIL] 无效构建类型：${BUILD_TYPE}（仅支持 release / debug）"
        exit 1
        ;;
esac

# ---- 依赖检测：缺失即报错退出（不代装，避免 root/sudo 假设），按提示安装后重跑 ----
MISSING=""
for cmd in cmake g++ python3; do
    command -v "${cmd}" > /dev/null 2>&1 || MISSING="${MISSING} ${cmd}"
done
if [ "${MODE}" = "cpu" ] || [ "${MODE}" = "npu" ]; then
    # libibverbs 按头文件探测（跨发行版通用；CMake 配置阶段同样要求 infiniband/verbs.h）
    # —— cpu 模式 RoCE 与 npu 模式三 provider（HCCP/RoCE/FAKE 共存）均编入 RoCE，均需要
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
    # 三 provider 共存后 npu 模式同时编入 RoCE：CMake 对未定义 option 的传参不报错，
    # 缺开关的分支会静默编出双 provider 而非三 provider，故同样前置检测
    if ! grep -q "BUILD_KV_CLIENT_PROVIDER_ROCE" "${KV_SEMANTICS_ROOT}/CMakeLists.txt"; then
        echo "[FAIL] 当前代码不含 RoCE provider 构建开关（${KV_SEMANTICS_ROOT}/CMakeLists.txt）"
        echo "       请切换到同时含 HCCP 与 RoCE provider 的分支后重跑"
        exit 1
    fi
fi

# ---- NPU 专属前置检测（编译前预检查）：CANN 环境就绪（$ASCEND_HOME_PATH 有值，且其下
# 按 kv client CMake 同样的候选布局能找到 acl 头文件；libascendcl 的六种库布局探测
# 交给 CMake 配置期做，其报错读法见 2 号 SOP §6。检测结果 echo 到控制台供确认，
# 缺失时报错提示给出可直接复制的设置步骤——source CANN 官方脚本优先，手动 export
# 为应急方式。此处同时确定头文件目录，供步骤 2 的 -I 注入使用——两种布局取实际命中者）----
if [ "${MODE}" = "npu" ]; then
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        echo "[FAIL] 未检测到 CANN 环境（\$ASCEND_HOME_PATH 为空）"
        echo "       请先安装 CANN toolkit，再在当前终端会话中设置环境变量后重跑（二选一）："
        echo "         1) source CANN 官方脚本（推荐——同时设置 LD_LIBRARY_PATH 等全套变量；"
        echo "            路径随 CANN 版本布局不同，两代均列出，取实际存在者）："
        echo "              CANN 8.x 及更早：source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo "              CANN 9.0 及更新：source /usr/local/Ascend/cann/set_env.sh"
        echo "         2) 手动 export（应急方式，仅设置本变量——运行期会因缺 LD_LIBRARY_PATH"
        echo "            找不到 libascendcl.so，优先用方式 1）："
        echo "              export ASCEND_HOME_PATH=<CANN 安装目录>"
        echo "              （如 CANN 9.0.0：/usr/local/Ascend/cann-9.0.0；8.x：/usr/local/Ascend/ascend-toolkit/latest）"
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
    echo "[预检查] CANN 环境就绪：ASCEND_HOME_PATH=${ASCEND_HOME_PATH}，头文件目录：${CANN_INC_DIR}"
fi

# ---- 步骤 1/2：构建 kv client（build-kv 目录名为 testTool 构建脚本的既定引用路径
# ../build-kv/src，不可改名；--target kv_client 只构建所需目标——两种 provider 的
# 源文件均已编入 libkv_transport.so 的源列表，随 kv_client 目标的依赖一并构建；
# 构建类型取 --build-type 解析值，缺省 release；
# 两模式各自显式声明全部 provider 开关（含关闭项）——CMake 缓存变量"不传即沿用
# 旧值"，不显式关闭则上次另一模式留下的 ON 残留到本次配置引发矛盾报错
# （如 HCCP=ON 残留 + simu → FATAL_ERROR），显式开关使两模式切换幂等无残留）----
if [ "${MODE}" = "cpu" ]; then
    echo "===== [1/2] 构建 kv client（CPU-over-RoCE + FAKE 双 provider：RUNTIME_ENVIRONMENT=simu，BUILD_KV_CLIENT_PROVIDER_ROCE=ON，BUILD_KV_CLIENT_PROVIDER_FAKE=ON，build-type=${BUILD_TYPE}）====="
    cmake -S "${KV_SEMANTICS_ROOT}" -B "${KV_SEMANTICS_ROOT}/build-kv" \
        -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBUILD_KV_CLIENT_PROVIDER_ROCE=ON -DBUILD_KV_CLIENT_PROVIDER_FAKE=ON \
        -DBUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=OFF
    cmake --build "${KV_SEMANTICS_ROOT}/build-kv" --target kv_client -j "$(nproc)"
else
    echo "===== [1/2] 构建 kv client（HCCP KV-over-RoCE + RoCE + FAKE 三 provider：RUNTIME_ENVIRONMENT=ascend，BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON，BUILD_KV_CLIENT_PROVIDER_ROCE=ON，BUILD_KV_CLIENT_PROVIDER_FAKE=ON，build-type=${BUILD_TYPE}）====="
    cmake -S "${KV_SEMANTICS_ROOT}" -B "${KV_SEMANTICS_ROOT}/build-kv" \
        -DRUNTIME_ENVIRONMENT=ascend -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE=ON \
        -DBUILD_KV_CLIENT_PROVIDER_ROCE=ON -DBUILD_KV_CLIENT_PROVIDER_FAKE=ON \
        -DKV_CLIENT_CANN_ROOT="${ASCEND_HOME_PATH}"
    cmake --build "${KV_SEMANTICS_ROOT}/build-kv" --target kv_client -j "$(nproc)"
fi

# ---- 步骤 2/2：构建 testTool（链接 build-kv 预构建产物；首次配置需联网拉取 fmt/spdlog）。
# npu 模式走 CMakeLists 正式开关 TESTTOOL_ENABLE_ASCEND=ON（编译宏、CANN 头文件路径、
# libascendcl 显式链接三联动，根目录经 TESTTOOL_ASCEND_ROOT 显式传入——不依赖
# 环境变量穿透 CMake 缓存，配置显式可追溯）；真 NPU 内存分配为 HCCP 链路建链
# 验证的必备条件（RegisterMemory 只收设备内存）----
if [ "${MODE}" = "cpu" ]; then
    echo "===== [2/2] 构建 testTool（build-type=${BUILD_TYPE}）====="
    cmake -S "${TESTTOOL_ROOT}" -B "${TESTTOOL_ROOT}/build" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DTESTTOOL_ENABLE_ASCEND=OFF
    cmake --build "${TESTTOOL_ROOT}/build" -j "$(nproc)"
else
    echo "===== [2/2] 构建 testTool（TESTTOOL_ENABLE_ASCEND：真 NPU 内存分配，build-type=${BUILD_TYPE}）====="
    cmake -S "${TESTTOOL_ROOT}" -B "${TESTTOOL_ROOT}/build" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DTESTTOOL_ENABLE_ASCEND=ON -DTESTTOOL_ASCEND_ROOT="${ASCEND_HOME_PATH}"
    cmake --build "${TESTTOOL_ROOT}/build" -j "$(nproc)"
fi

# ---- 结果提示：产物路径与下一步运行引导 ----
echo "===== 构建完成（mode=${MODE}，build-type=${BUILD_TYPE}）====="
echo "kv client 库：${KV_SEMANTICS_ROOT}/build-kv/src/libkv_client.so（及 libkv_transport.so）"
echo "testTool    ：${TESTTOOL_ROOT}/build/testTool"
echo
if [ "${MODE}" = "cpu" ]; then
    echo "下一步（CPU 建链断链验证；FAKE 模拟与 RoCE 真链路均已编入，conf 改 provider_type 即切换）："
    echo "  1. 复制 conf/kv_client.conf 与 conf/kv_client_view.conf 为副本：transport.provider_type 改 CPU；"
    echo "     RoCE 旋钮键名 transport.roce_*（模板段已内置，取消注释按环境改值即可，"
    echo "     必填 roce_gid_index；对端 KV store 侧同名键前缀为 asu_roce_*），"
    echo "     参考 ${KV_SEMANTICS_ROOT}/docs/guide/roce-provider.md；对端 KV store 需先就绪"
    echo "  2. 复制 conf/test_config.conf 为副本改 onlyconnect=on，建链验证（在 ${TESTTOOL_ROOT} 目录下执行）："
    echo "     ./build/testTool <test conf 副本> <kv client 副本>"
    echo "     建链 + region 注册完成后 Ctrl+C 断链退出（日志确认 Ctrl+C received, exiting）"
else
    echo "下一步（NPU 建链断链验证，详见 sop/2.npu-connect-disconnect-sop.md；FAKE 模拟与 RoCE 真链路也已编入，"
    echo "       conf 改 transport.provider_type 即在 FAKE / CPU-over-RoCE / HCCP 三种链路间切换，无需重编）："
    echo "  1. 复制 conf/kv_client.conf 为副本：transport.provider_type 改 HCCP_KV_OVER_ROCE；"
    echo "     transport.hccp_kv_roce_lib_path 指向算子库主体绝对路径（模板段已内置）；对端 IP 填在"
    echo "     asu_info.N 的 local.comm_id 字段（对端 KV store 以 HCCP 后端就绪）"
    echo "  2. 算子库三个 so 必须同目录：libhccp_kv_over_roce.so + librdma_kv_send_kernel.so"
    echo "     + librdma_kv_wait_kernel.so（算子库主体在同目录加载两个内核，分目录放置无法找到）"
    echo "  3. 建链验证（在 ${TESTTOOL_ROOT} 目录下执行）：复制 conf/test_config.conf 为副本，"
    echo "     onlyconnect=on 且 buffertype 设 npu，然后执行"
    echo "     KV_LOG_LEVEL=debug ./build/testTool <test conf 副本> <kv client 副本>"
fi
