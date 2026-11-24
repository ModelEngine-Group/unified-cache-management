#!/usr/bin/env bash
# ============================================================================
# clean-build.sh — 清理编译缓存与构建产物
#
# 用途：删除构建产生的全部目录（可安全重跑，清理后用 build-kv-link.sh 重建）：
#   kv_semantics/build-kv       kv client 构建目录（build-kv-link.sh 步骤 1 产物）
#   testTool/build              testTool 构建目录（含 _deps 下 fmt / spdlog 依赖缓存）
#   testTool/build-arm          ARM 构建目录（历史遗留布局，存在即一并清理）
#
# 适用场景：切换构建模式后配置缓存异常、FetchContent 依赖源损坏（如报
# "spdlog/async.h No such file"）、磁盘空间回收、彻底重建。
# 运行产物（log/、asu-fake-backend-store/ 等）不属于编译缓存，本脚本不清理。
#
# 用法（在 testTool 目录下执行；从任意目录发起亦可，路径以脚本自身位置推算）：
#   bash scripts/clean-build.sh
#
# 清理后重建：bash scripts/build-kv-link.sh（缺省 CPU 模式；NPU 链路加 --mode npu）。
# 注意：testTool 侧首次重建需联网重新拉取 fmt / spdlog（FetchContent），无外网
# 机器改用离线方式（见 sop/0 构建节）。
# ============================================================================

# ---- 解释器检测（须在 set 之前：Ubuntu 的 sh 是 dash，不支持 BASH_SOURCE；
# 用 sh 执行时给出明确提示，与 build-kv-link.sh 同款）----
if [ -z "${BASH_VERSION:-}" ]; then
    echo "[FAIL] 本脚本需用 bash 执行（当前解释器不支持所需特性，如 Ubuntu 的 sh）"
    echo "       请改用：bash $0"
    exit 1
fi

set -euo pipefail

# ---- 路径推算：以脚本自身位置为基准（与 build-kv-link.sh 同款，从任意目录发起
# 均可，不依赖当前工作目录）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTTOOL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"             # kv_semantics/testTool
KV_SEMANTICS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"      # kv_semantics

# ---- 逐目录清理：存在则先取占用量再删除，不存在则跳过（幂等，可安全重跑）----
for dir in "${KV_SEMANTICS_ROOT}/build-kv" "${TESTTOOL_ROOT}/build" "${TESTTOOL_ROOT}/build-arm"; do
    if [ -d "${dir}" ]; then
        size="$(du -sh "${dir}" 2>/dev/null | cut -f1)"
        rm -rf "${dir}"
        echo "已清理：${dir}（释放 ${size}）"
    else
        echo "跳过（不存在）：${dir}"
    fi
done

# ---- 收尾提示：重建入口与联网注意事项 ----
echo
echo "===== 清理完成 ====="
echo "重建执行：bash scripts/build-kv-link.sh（缺省 CPU 模式；NPU 链路加 --mode npu）"
echo "注意：testTool 侧首次重建需联网重新拉取 fmt / spdlog（FetchContent）"
