#!/usr/bin/env bash
# ============================================================================
# build-cpu-roce.sh — CPU 建链断链测试一键编译脚本
#
# 用途：编译 CPU-over-RoCE 待测链路的全部组件（kv client + testTool）：
#   1) kv client：BUILD_KV_CLIENT_PROVIDER_ROCE=ON（RoCE provider 编入
#      libkv_transport.so，需要 libibverbs）
#   2) testTool：链接上一步 build-kv 目录下的预构建 kv client 库
#
# 适用代码：含 RoCE provider 的分支——当前按 cpu-kv-over-roce-provider 分支的
# 代码结构编写（CMake 开关 BUILD_KV_CLIENT_PROVIDER_ROCE + src/trans_provider/roce/），
# 后续联调分支就绪后同样适用（脚本编译当前 checkout 的代码，本身不切分支）。
#
# 用法（在 demo/scripts 目录下执行）：
#   bash build-cpu-roce.sh
#
# 前置依赖（缺失时脚本前置检测报错退出，按提示安装后重跑）：
#   cmake(>=3.18) / g++(C++17) / python3（metrics 代码生成） / libibverbs-dev（RoCE）
# ============================================================================

# ---- 解释器检测（须在 set 之前：Ubuntu 的 sh 是 dash，不支持 -o pipefail 与
# BASH_SOURCE；用 sh 执行时在此给出明确提示，而不是晦涩的 "Illegal option" 报错）----
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[FAIL] 本脚本需用 bash 执行（当前解释器不支持所需特性，如 Ubuntu 的 sh）"
    echo "       请改用：bash $0"
    exit 1
fi

set -euo pipefail

# ---- 路径推算：以脚本自身位置为基准（scripts 位于 kv_semantics/testTool/demo/scripts，
# 从任意目录发起执行均可，不依赖当前工作目录）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTTOOL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"          # kv_semantics/testTool
KV_SEMANTICS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # kv_semantics

# ---- 依赖检测：缺失即报错退出（不代装，避免 root/sudo 假设），按提示安装后重跑 ----
MISSING=""
for cmd in cmake g++ python3; do
    command -v "${cmd}" > /dev/null 2>&1 || MISSING="${MISSING} ${cmd}"
done
# libibverbs 按头文件探测（跨发行版通用；CMake 配置阶段同样要求 infiniband/verbs.h）
[ -f /usr/include/infiniband/verbs.h ] || MISSING="${MISSING} libibverbs-dev"
if [ -n "${MISSING}" ]; then
    echo "[FAIL] 缺少依赖：${MISSING}"
    echo "       Ubuntu/Debian 安装示例：sudo apt-get install -y cmake g++ python3 libibverbs-dev"
    exit 1
fi

# ---- 分支能力检测：当前 checkout 的 kv_semantics 必须含 RoCE provider 构建开关，
# 防止在不含该代码的分支上误跑（静默编出无 RoCE 的产物而不自知）----
if ! grep -q "BUILD_KV_CLIENT_PROVIDER_ROCE" "${KV_SEMANTICS_ROOT}/CMakeLists.txt"; then
    echo "[FAIL] 当前代码不含 RoCE provider 构建开关（${KV_SEMANTICS_ROOT}/CMakeLists.txt）"
    echo "       请切换到待测分支（cpu-kv-over-roce-provider 或后续联调分支）后重跑"
    exit 1
fi

# ---- 步骤 1/2：构建 kv client（RoCE provider 开启；build-kv 目录名为 testTool 构建
# 脚本的既定引用路径 ../build-kv/src，不可改名；--target kv_client 只构建所需目标）----
echo "===== [1/2] 构建 kv client（BUILD_KV_CLIENT_PROVIDER_ROCE=ON）====="
cmake -S "${KV_SEMANTICS_ROOT}" -B "${KV_SEMANTICS_ROOT}/build-kv" \
    -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug -DBUILD_KV_CLIENT_PROVIDER_ROCE=ON
cmake --build "${KV_SEMANTICS_ROOT}/build-kv" --target kv_client -j "$(nproc)"

# ---- 步骤 2/2：构建 testTool（链接 build-kv 预构建产物；首次配置需联网拉取 fmt/spdlog）。
echo "===== [2/2] 构建 testTool ====="
cmake -S "${TESTTOOL_ROOT}" -B "${TESTTOOL_ROOT}/build" -DCMAKE_BUILD_TYPE=Debug
cmake --build "${TESTTOOL_ROOT}/build" -j "$(nproc)"

# ---- 结果提示：产物路径与下一步运行引导 ----
echo "===== 构建完成 ====="
echo "kv client 库：${KV_SEMANTICS_ROOT}/build-kv/src/libkv_client.so（及 libkv_transport.so）"
echo "testTool    ：${TESTTOOL_ROOT}/build/testTool"
echo
echo "下一步（CPU 建链断链验证）："
echo "  1. kv client 配置传输段改 CPU-over-RoCE：transport.provider_type=CPU；"
echo "     RoCE 旋钮键名 transport.roce_*（完整模板已内置 demo/conf/kv_client_real.conf，"
echo "     取消注释按环境改值即可；对端 KV store 侧同名键前缀为 asu_roce_*），"
echo "     参考 ${KV_SEMANTICS_ROOT}/docs/guide/roce-provider.md；对端 KV store 需先就绪"
echo "  2. 建链验证（在 ${TESTTOOL_ROOT} 目录下执行）："
echo "     ./build/testTool demo/conf/test_config_onlyconnect.conf <kv client conf>"
echo "     建链 + region 注册完成后 Ctrl+C 断链退出（日志确认 Ctrl+C received, exiting）"
