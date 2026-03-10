#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
# Modified from 
# https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build_aclnn.sh

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

echo "ROOT_DIR: $ROOT_DIR"

IS_INSTALL="false"
SOC_VERSION="ascend910b"

for arg in "$@"; do
    if [[ "$arg" == "a3" ]]; then
        SOC_VERSION="ascend910_93"
    elif [[ "$arg" == "install" ]]; then
        IS_INSTALL="true"
    fi
done

echo "SOC_VERSION: $SOC_VERSION"


if [[ "$SOC_VERSION" =~ ^ascend310 ]]; then
    # ASCEND310P series
    # currently, no custom aclnn ops for ASCEND310 series
    # CUSTOM_OPS=""
    # SOC_ARG="ascend310p"
    exit 0
elif [[ "$SOC_VERSION" =~ ^ascend910b ]]; then
    # ASCEND910B (A2) series
    # dependency: catlass
    CUSTOM_OPS="hamming_dist_top_k;reshape_and_cache_bnsd;"
    SOC_ARG="ascend910b"
elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]]; then
    # ASCEND910C (A3) series
    # dependency: catlass
    # dependency: cann-toolkit file moe_distribute_base.h
    HCCL_STRUCT_FILE_PATH=$(find -L "${ASCEND_TOOLKIT_HOME}" -name "moe_distribute_base.h" 2>/dev/null | head -n1)
    if [ -z "$HCCL_STRUCT_FILE_PATH" ]; then
        echo "cannot find moe_distribute_base.h file in CANN env"
        exit 1
    fi
    # for dispatch_gmm_combine_decode
    yes | cp "${HCCL_STRUCT_FILE_PATH}" "${ROOT_DIR}/csrc/utils/inc/kernel"
    # for dispatch_ffn_combine
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR="$SCRIPT_DIR/dispatch_ffn_combine/op_kernel/utils/"
    TARGET_FILE="$TARGET_DIR/$(basename "$HCCL_STRUCT_FILE_PATH")"
    # for dispatch_ffn_combine_bf16
    SCRIPT_DIR_BF16=$(cd "$(dirname "$0")" && pwd)
    TARGET_DIR_BF16="$SCRIPT_DIR_BF16/dispatch_ffn_combine_bf16/op_kernel/utils/"
    TARGET_FILE_BF16="$TARGET_DIR_BF16/$(basename "$HCCL_STRUCT_FILE_PATH")"

    echo "*************************************"
    echo $HCCL_STRUCT_FILE_PATH
    echo "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR"
    cp "$HCCL_STRUCT_FILE_PATH" "$TARGET_DIR_BF16"

    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE"
    sed -i 's/struct HcclOpResParam {/struct HcclOpResParamCustom {/g' "$TARGET_FILE_BF16"
    sed -i 's/struct HcclRankRelationResV2 {/struct HcclRankRelationResV2Custom {/g' "$TARGET_FILE_BF16"

    CUSTOM_OPS_ARRAY=(
        "hamming_dist_top_k"
        "reshape_and_cache_bnsd"
    )
    CUSTOM_OPS=$(IFS=';'; echo "${CUSTOM_OPS_ARRAY[*]}")
    SOC_ARG="ascend910_93"
else
    # others
    # currently, no custom aclnn ops for other series
    exit 0
fi


# build custom ops
cd $ROOT_DIR
echo "Current directory: $PWD, begin to build custom ops (ascendc version)..."
rm -rf build output
echo "Building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build_ops.sh -n "$CUSTOM_OPS" -c "$SOC_ARG"

# build python wheel
echo "Begin to build python wheel..."
bash build_wheel.sh

# install custom ops and python package
echo "Installing ucm custom ops. This may take a while, please wait..."
if [[ "$IS_INSTALL" == "true" ]]; then
    bash install.sh -s
fi
