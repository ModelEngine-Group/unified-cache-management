add_ops_compile_options(
        OP_NAME ReshapeAndCacheBnsd
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(reshape_and_cache_bnsd PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        reshape_and_cache_bnsd.cpp
)

target_sources(optiling PRIVATE
        reshape_and_cache_bnsd_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        reshape_and_cache_bnsd_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        reshape_and_cache_bnsd_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
