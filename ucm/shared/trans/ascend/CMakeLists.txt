set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
add_library(Ascend::ascendcl UNKNOWN IMPORTED)
set_target_properties(Ascend::ascendcl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
    IMPORTED_LOCATION "${ASCEND_ROOT}/lib64/libascendcl.so"
)
add_library(trans STATIC
    ascend_device.cc
    ascend_buffer.cc
    ascend_event.cc
    ascend_stream.cc
)
if(ASCEND_SUPPORTS_REGISTER_PIN)
    target_compile_definitions(trans PRIVATE ASCEND_SUPPORTS_REGISTER_PIN=1)
endif()
if(UCM_RUNTIME_ASCEND_SDMA_DIRECT)
    set(ASCEND_ARCH_DIR "${ASCEND_ROOT}/aarch64-linux" CACHE PATH
        "Path to Ascend architecture-specific directory")
    set(ASCEND_PKG_INCLUDE_DIR "${ASCEND_ARCH_DIR}/pkg_inc" CACHE PATH
        "Path to Ascend package include directory")
    set(ASCEND_FFTS_INCLUDE_DIR "${ASCEND_ARCH_DIR}/pkg_inc/runtime" CACHE PATH
        "Path to Ascend FFTS include directory")
    set(ASCEND_TOOLCHAIN_INCLUDE_DIR "${ASCEND_ARCH_DIR}/pkg_inc/toolchain" CACHE PATH
        "Path to Ascend toolchain include directory")
    set(ASCEND_PROFILING_INCLUDE_DIR "${ASCEND_ARCH_DIR}/pkg_inc/profiling" CACHE PATH
        "Path to Ascend profiling include directory")
    set(ASCEND_RUNTIME_LIBRARY "${ASCEND_ARCH_DIR}/lib64/libruntime.so" CACHE FILEPATH
        "Path to Ascend runtime library")
    if(NOT EXISTS "${ASCEND_RUNTIME_LIBRARY}")
        message(FATAL_ERROR
            "RUNTIME_ENVIRONMENT=ascend-a3 requires libruntime: ${ASCEND_RUNTIME_LIBRARY}")
    endif()
    if(NOT EXISTS "${ASCEND_FFTS_INCLUDE_DIR}/runtime/rt_ffts_plus.h"
        AND NOT EXISTS "${ASCEND_FFTS_INCLUDE_DIR}/rt_external_ffts.h")
        message(FATAL_ERROR
            "RUNTIME_ENVIRONMENT=ascend-a3 requires FFTS headers in: ${ASCEND_FFTS_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ASCEND_TOOLCHAIN_INCLUDE_DIR}/prof_api.h")
        message(FATAL_ERROR
            "RUNTIME_ENVIRONMENT=ascend-a3 requires Ascend toolchain headers in: ${ASCEND_TOOLCHAIN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${ASCEND_PROFILING_INCLUDE_DIR}/prof_common.h")
        message(FATAL_ERROR
            "RUNTIME_ENVIRONMENT=ascend-a3 requires Ascend profiling headers in: ${ASCEND_PROFILING_INCLUDE_DIR}")
    endif()
    target_sources(trans PRIVATE
        sdma_direct/ascend_sdma_direct_stream.cc
        sdma_direct/ascend_sdma_direct_copier.cc
        ffts/ffts_sdma_dispatcher.cc
    )
    target_include_directories(trans PUBLIC
        "${ASCEND_FFTS_INCLUDE_DIR}"
        "${ASCEND_PKG_INCLUDE_DIR}"
        "${ASCEND_TOOLCHAIN_INCLUDE_DIR}"
        "${ASCEND_PROFILING_INCLUDE_DIR}"
    )
    target_link_libraries(trans PUBLIC "${ASCEND_RUNTIME_LIBRARY}")
endif()
target_link_libraries(trans PUBLIC
    fmt
    Ascend::ascendcl
)
