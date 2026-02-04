#!/bin/bash

ROOT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

echo "ROOT_DIR: $ROOT_DIR"

# Get value for a key from npu-smi output lines (key : value format)
get_value_from_lines() {
    local key="$1"
    awk -v key="$key" '$0 ~ key { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }'
}

get_chip_type() {
    if ! command -v npu-smi &>/dev/null; then
        echo "npu-smi command not found, if this is an npu envir, please check if npu driver is installed correctly." >&2
        echo ""
        return
    fi
    local npu_info chip_info npu_id chip_name chip_type npu_name
    npu_info=$(npu-smi info -l 2>/dev/null) || { echo "Get chip info failed" >&2; return 1; }
    npu_id=$(echo "$npu_info" | get_value_from_lines "NPU ID")
    chip_info=$(npu-smi info -t board -i "$npu_id" -c 0 2>/dev/null) || { echo "Get chip info failed" >&2; return 1; }
    chip_name=$(echo "$chip_info" | get_value_from_lines "Chip Name")
    chip_type=$(echo "$chip_info" | get_value_from_lines "Chip Type")
    npu_name=$(echo "$chip_info" | get_value_from_lines "NPU Name")

    if [[ "$chip_name" == *310* ]]; then
        # 310P case
        [[ -n "$chip_type" ]] || { echo "Expected chip_type for 310" >&2; exit 1; }
        echo "${chip_type}${chip_name}" | tr 'A-Z' 'a-z'
    elif [[ "$chip_name" == *910* ]]; then
        if [[ -n "$chip_type" ]]; then
            # A2 case
            [[ -z "$npu_name" ]] || { echo "Unexpected npu_name for A2" >&2; exit 1; }
            echo "${chip_type}${chip_name}" | tr 'A-Z' 'a-z'
        else
            # A3 case
            [[ -n "$npu_name" ]] || { echo "Expected npu_name for A3" >&2; exit 1; }
            echo "${chip_name}_${npu_name}" | tr 'A-Z' 'a-z'
        fi
    else
        # TODO(zzzzwwjj): Currently, A5's chip name has not determined yet.
        echo "Unable to recognize chip name: ${chip_name}, please manually set env SOC_VERSION" >&2
        exit 1
    fi
}

SOC_VERSION=$(get_chip_type) || exit 1

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
echo "current directory: $PWD, begin to build custom ops (ascendc version)..."
rm -rf build output
echo "building custom ops $CUSTOM_OPS for $SOC_VERSION"
bash build.sh -n "$CUSTOM_OPS" -c "$SOC_ARG"

# install custom ops inside csrc/ascend, the path is $ROOT_DIR/_ucm_custom_ops
echo "installing custom ops in $ROOT_DIR/_ucm_custom_ops"
mkdir -p $ROOT_DIR/_ucm_custom_ops
./output/*.run --install-path=$ROOT_DIR/_ucm_custom_ops

# install custom_ops in the default path which is /usr/local/Ascend/latest/opp/vendors
# echo "installing custom ops in /usr/local/Ascend/latest/opp/vendors"
#./output/*.run

# update environment variables ASCEND_CUSTOM_OPP_PATH and LD_LIBRARY_PATH
# such that the compiled ascend custom ops can be used in the current shell
set_env_path=$ROOT_DIR/_ucm_custom_ops/vendors/ucm/bin/set_env.bash
source $set_env_path

# update environment variables ASCEND_CUSTOM_OPP_PATH and LD_LIBRARY_PATH in ~/.bashrc
# such that the compiled ascend custom ops can be used in the future shells (add at most once)
for pattern in "ASCEND_CUSTOM_OPP_PATH" "LD_LIBRARY_PATH"; do
  line=$(grep "$pattern" "$set_env_path")
  if [ -n "$line" ] && ! grep -Fqx "$line" ~/.bashrc 2>/dev/null; then
    echo "$line" >> ~/.bashrc
  fi
done

# install ucm_custom_ops python package
echo "installing ucm_custom_ops python package. This may take a while, please wait...)"
bash install_python_package.sh