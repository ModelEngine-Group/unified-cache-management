add_ops_compile_options(
        OP_NAME HammingDistTopK
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(hamming_dist_top_k PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        hamming_dist_top_k_def.cpp
)

target_sources(optiling PRIVATE
        hamming_dist_top_k_tiling.cpp
        hamming_dist_top_k.cpp
        hamming_dist_top_k_split.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        hamming_dist_top_k_tiling.cpp
        hamming_dist_top_k.cpp
        hamming_dist_top_k_split.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        hamming_dist_top_k_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
