# Ascend-only benchmark: the bench drives aclrt itself -- runtime init,
# host and device memory -- so no acl/acl_rt.h exists to build against
# under the other runtime environments.
if(RUNTIME_ENVIRONMENT STREQUAL "ascend" OR
   RUNTIME_ENVIRONMENT STREQUAL "ascend-a3" OR
   RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    # The CANN paths of ucm_require_cann are PRIVATE to ucm_transfer_memory;
    # the bench includes acl/acl_rt.h itself and finds the same runtime again
    # under the same hints.
    find_path(BENCH_ACL_RT_INCLUDE_DIR acl/acl_rt.h
        HINTS ENV ASCEND_HOME_PATH
        PATHS /usr/local/Ascend/ascend-toolkit/latest
        PATH_SUFFIXES include
    )
    find_library(BENCH_ACL_RT_LIBRARY acl_rt
        HINTS ENV ASCEND_HOME_PATH
        PATHS /usr/local/Ascend/ascend-toolkit/latest
        PATH_SUFFIXES lib64
    )
    if(NOT BENCH_ACL_RT_INCLUDE_DIR OR NOT BENCH_ACL_RT_LIBRARY)
        message(FATAL_ERROR "CANN runtime not found: no acl/acl_rt.h + libacl_rt.so under "
            "ASCEND_HOME_PATH or /usr/local/Ascend/ascend-toolkit/latest; install CANN or set "
            "ASCEND_HOME_PATH")
    endif()

    add_executable(bench_ascend_dma bench_ascend_dma.cc)
    target_include_directories(bench_ascend_dma PRIVATE ${BENCH_ACL_RT_INCLUDE_DIR})
    target_link_libraries(bench_ascend_dma
        PRIVATE ucm_build_options ucm_include ucm_common ucm_transfer_memory ${BENCH_ACL_RT_LIBRARY})
endif()
