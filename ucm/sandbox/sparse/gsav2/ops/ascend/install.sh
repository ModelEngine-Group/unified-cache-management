#!/usr/bin/env bash
# GSA AscendC kernel 算子安装脚本
# 仅安装 .run 到 vendors/gsa（提供 aclnn 接口头文件 + libcust_opapi.so）
# whl 安装由顶层 ops/build.sh 统一处理
#
# .run 由 build_ops.sh 生成，位于 ascend/output/（-s 模式）或 ascend/build/
#
# 用法:
#   bash install.sh      # 从 build/ 找 .run 安装
#   bash install.sh -s   # 从 output/ 找 .run 安装（source 模式，build.sh 用此方式）

set -e

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
WITH_SOURCE="false"

while getopts "s" arg; do
    case "$arg" in
        s) WITH_SOURCE="true" ;;
        ?)
            echo "Usage: $0 [-s]"
            echo "  -s: Install from output/ (source mode, .run collected by build_ops.sh)"
            exit 1
            ;;
    esac
done

echo "[GSA] install.sh ROOT_DIR: $ROOT_DIR, WITH_SOURCE: $WITH_SOURCE"

# .run 文件查找路径
# - 默认: build/（CPack 直接生成位置）
# - -s 模式: output/（build_ops.sh collect_output 复制位置）
RUN_FILE_DIR="$ROOT_DIR/build"
if [ "$WITH_SOURCE" = "true" ]; then
    RUN_FILE_DIR="$ROOT_DIR/output"
fi

# 查找 .run 文件
RUN_FILE=$(ls ${RUN_FILE_DIR}/GSA-custom_ops*.run 2>/dev/null | head -1)
if [ -z "$RUN_FILE" ]; then
    echo "[GSA] ERROR: GSA-custom_ops*.run not found in $RUN_FILE_DIR"
    echo "[GSA] Please run 'bash build_ops.sh -n kv_quant_sparse_flash_attention_gsa -c <soc>' first"
    exit 1
fi

# 安装 .run 到 vendors/gsa
echo "[GSA] Installing kernel ops: $(basename $RUN_FILE)"
chmod +x "$RUN_FILE"
"$RUN_FILE"

# 确定 vendors 安装路径（root / 非root 自适应）
USER_ID=$(id -u)
if [ "${USER_ID}" = "0" ]; then
    VENDOR_BASE="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/gsa"
else
    VENDOR_BASE="${HOME}/Ascend/ascend-toolkit/latest/opp/vendors/gsa"
fi

# 设置 LD_LIBRARY_PATH 环境变量
VENDOR_LIB_DIR="${VENDOR_BASE}/op_api/lib"
line="export LD_LIBRARY_PATH=${VENDOR_LIB_DIR}:\${LD_LIBRARY_PATH}"
env_file="${VENDOR_BASE}/setenv.bash"

mkdir -p "$(dirname "$env_file")" 2>/dev/null || true
[ ! -f "$env_file" ] && touch "$env_file" 2>/dev/null || true

if [ -n "$line" ] && ! grep -Fqx "$line" ~/.bashrc 2>/dev/null; then
    echo "$line" >> ~/.bashrc
fi
if [ -n "$line" ] && [ -w "$env_file" ] && ! grep -Fqx "$line" "$env_file" 2>/dev/null; then
    echo "$line" >> "$env_file"
fi
[ -f "$env_file" ] && source "$env_file" 2>/dev/null || true

echo "[GSA] Kernel ops installed to ${VENDOR_BASE}"
echo "[GSA]   aclnn headers: ${VENDOR_BASE}/op_api/include"
echo "[GSA]   aclnn lib:     ${VENDOR_LIB_DIR}/libcust_opapi.so"
echo "[GSA] LD_LIBRARY_PATH updated in ~/.bashrc and ${env_file}"
