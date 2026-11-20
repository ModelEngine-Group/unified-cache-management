#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# GSA ops 统一构建脚本
#
# 统一流程:
#   Step 1: 编译 AscendC 自定义算子 kernel (ascend/build_ops.sh) → ascend/output/GSA-custom_ops*.run
#   Step 2: 安装 .run 到 vendors/gsa (ascend/install.sh) → aclnn 接口头文件 + libcust_opapi.so
#   Step 3: 编译 gsa_ops whl (setup_wheel.py) → dist/gsa_ops*.whl
#   Step 4: 安装 gsa_ops whl (pip install) → torch.ops._C_gsa 可用
#
# 用法:
#   bash build.sh              # 默认 A3，仅编译不安装
#   bash build.sh a2           # 指定 A2
#   bash build.sh a5           # 指定 A5
#   bash build.sh a3 install   # A3，编译并安装（.run + whl）
#   bash build.sh whl install  # 跳过 kernel 编译，仅编译安装 whl（假设 .run 已安装）

set -e

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
echo "[GSA] build.sh ROOT_DIR: $ROOT_DIR"

IS_INSTALL="false"
SOC_VERSION="ascend910_93"   # 默认 A3
BUILD_KERNEL="true"

for arg in "$@"; do
    case "$arg" in
        a2)  SOC_VERSION="ascend910b" ;;
        a3)  SOC_VERSION="ascend910_93" ;;
        a5)  SOC_VERSION="ascend950" ;;
        install) IS_INSTALL="true" ;;
        whl)  BUILD_KERNEL="false" ;;   # 仅编译 whl，跳过 kernel
        *) echo "[GSA] Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "[GSA] SOC_VERSION=$SOC_VERSION IS_INSTALL=$IS_INSTALL BUILD_KERNEL=$BUILD_KERNEL"

# ---------------------------------------------------------------------------
# Step 1: 编译 AscendC 自定义算子 kernel
# ---------------------------------------------------------------------------
if [ "$BUILD_KERNEL" = "true" ]; then
    ASCEND_DIR="$ROOT_DIR/ascend"
    GSA_OP="kv_quant_sparse_flash_attention_gsa"

    case "$SOC_VERSION" in
        ascend910b)    SOC_ARG="ascend910b" ;;
        ascend910_93)  SOC_ARG="ascend910_93" ;;
        ascend950)     SOC_ARG="ascend950" ;;
        *) echo "[GSA] Unsupported SOC: $SOC_VERSION"; exit 1 ;;
    esac

    echo "[GSA] ===== Step 1: Build AscendC kernel ops ($GSA_OP for $SOC_ARG) ====="
    cd "$ASCEND_DIR"
    echo "[GSA] cwd: $PWD"
    rm -rf build output
    bash build_ops.sh -n "$GSA_OP" -c "$SOC_ARG"

    # ---------------------------------------------------------------------------
    # Step 2: 安装 .run 到 vendors/gsa
    # ---------------------------------------------------------------------------
    if [ "$IS_INSTALL" = "true" ]; then
        echo "[GSA] ===== Step 2: Install .run to vendors/gsa ====="
        bash install.sh -s
    else
        echo "[GSA] ===== Step 2: Skipped (use 'install' arg to install .run) ====="
        echo "[GSA]   .run package: $ASCEND_DIR/output/GSA-custom_ops*.run"
        echo "[GSA]   Run 'bash ascend/install.sh' manually to install kernel ops"
    fi
else
    echo "[GSA] ===== Step 1-2: Skipped (whl-only mode, assuming .run already installed) ====="
fi

# ---------------------------------------------------------------------------
# Step 3: 编译 gsa_ops whl（NpuExtension 编译 torch_binding.cpp + meta）
# 注意: EXEC_NPU_CMD 用 dlsym 动态查找 aclnn 函数，编译时不需要 aclnn 头文件
#       但运行时需要 vendors/gsa/libcust_opapi.so（即 Step 2 已安装）
# ---------------------------------------------------------------------------
echo "[GSA] ===== Step 3: Build gsa_ops wheel ====="
cd "$ROOT_DIR"
echo "[GSA] cwd: $PWD"
mkdir -p build dist
rm -rf build/* dist/*
python3 setup_wheel.py build bdist_wheel

# ---------------------------------------------------------------------------
# Step 4: 安装 gsa_ops whl
# ---------------------------------------------------------------------------
if [ "$IS_INSTALL" = "true" ]; then
    echo "[GSA] ===== Step 4: Install gsa_ops wheel ====="
    pip3 install dist/gsa_ops*.whl --force-reinstall
fi

echo "[GSA] ===== Done. gsa_ops wheel: $ROOT_DIR/dist/gsa_ops*.whl ====="
