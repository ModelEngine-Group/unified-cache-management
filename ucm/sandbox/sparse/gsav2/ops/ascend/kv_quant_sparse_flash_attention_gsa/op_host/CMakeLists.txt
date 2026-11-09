# op_host CMakeLists for KvQuantSparseFlashAttentionGsa
# 参考 gsa_on_device/hamming_dist_top_k/op_host/CMakeLists.txt 风格

add_ops_compile_options(
        OP_NAME KvQuantSparseFlashAttentionGsa
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(kv_quant_sparse_flash_attention_gsa PARENT_SCOPE)

# def.cpp 喂给 op_host_aclnn → op_build 生成 aclnn 接口
target_sources(op_host_aclnn PRIVATE
        kv_quant_sparse_flash_attention_gsa_def.cpp
)

# tiling 实现
target_sources(optiling PRIVATE
        kv_quant_sparse_flash_attention_tiling.cpp
)

# proto (InferShape / InferDataType)
target_sources(opsproto PRIVATE
        kv_quant_sparse_flash_attention_gsa_proto.cpp
)

# 旧版 infershape (兼容 BUILD_OPEN_PROJECT=OFF 路径，保留)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(opsproto PRIVATE
        kv_quant_sparse_flash_attention_infershape.cpp
    )
    target_sources(opmaster_ct PRIVATE
        kv_quant_sparse_flash_attention_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
