function(ucm_require_cann TARGET)
    find_path(ACL_RT_INCLUDE_DIR acl/acl_rt.h
        HINTS ENV ASCEND_HOME_PATH
        PATHS /usr/local/Ascend/ascend-toolkit/latest
        PATH_SUFFIXES include
    )
    find_library(ACL_RT_LIBRARY acl_rt
        HINTS ENV ASCEND_HOME_PATH
        PATHS /usr/local/Ascend/ascend-toolkit/latest
        PATH_SUFFIXES lib64
    )
    if(NOT ACL_RT_INCLUDE_DIR OR NOT ACL_RT_LIBRARY)
        message(FATAL_ERROR "CANN runtime not found: no acl/acl_rt.h + libacl_rt.so under "
            "ASCEND_HOME_PATH or /usr/local/Ascend/ascend-toolkit/latest; install CANN or set "
            "ASCEND_HOME_PATH")
    endif()
    target_include_directories(${TARGET} PRIVATE ${ACL_RT_INCLUDE_DIR})
    target_link_libraries(${TARGET} PUBLIC ${ACL_RT_LIBRARY})
endfunction()

if(RUNTIME_ENVIRONMENT STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(ucm_transfer_memory PUBLIC CUDA::cudart)
    target_sources(ucm_transfer_memory PRIVATE
        cuda_naive_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    ucm_require_cann(ucm_transfer_memory)
    target_sources(ucm_transfer_memory PRIVATE
        ascend_naive_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend-a3")
    ucm_require_cann(ucm_transfer_memory)
    target_sources(ucm_transfer_memory PRIVATE
        ascend_naive_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    ucm_require_cann(ucm_transfer_memory)
    target_sources(ucm_transfer_memory PRIVATE
        ascend_naive_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "musa")
    target_sources(ucm_transfer_memory PRIVATE
        musa_naive_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "simu")
    target_sources(ucm_transfer_memory PRIVATE
        simu_naive_dma_engine.cc
    )
else()
    message(FATAL_ERROR "Unsupported RUNTIME_ENVIRONMENT, got: '${RUNTIME_ENVIRONMENT}'")
endif()
