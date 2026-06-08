set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
add_library(Ascend::ascendcl UNKNOWN IMPORTED)
set_target_properties(Ascend::ascendcl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
    IMPORTED_LOCATION "${ASCEND_ROOT}/lib64/libascendcl.so"
)
add_library(trans STATIC
    ascend_device.cc
    ascend_buffer.cc
    ascend_stream.cc
)
if(ASCEND_SUPPORTS_REGISTER_PIN)
    target_compile_definitions(trans PRIVATE ASCEND_SUPPORTS_REGISTER_PIN=1)
endif()
if(UCM_ENABLE_ASCEND_IO_AGGREGATION)
    find_library(ASCEND_RUNTIME_LIBRARY runtime
        PATHS
            "${ASCEND_ROOT}/lib64"
            "${ASCEND_ROOT}/lib"
            "${ASCEND_ROOT}/runtime/lib64"
            "${ASCEND_ROOT}/runtime/lib"
            "${ASCEND_ROOT}/aarch64-linux/lib64"
            "${ASCEND_ROOT}/aarch64-linux/lib"
        NO_DEFAULT_PATH
    )
    find_path(ASCEND_FFTS_INCLUDE_DIR
        NAMES runtime/rt_ffts_plus.h rt_external_ffts.h
        PATHS
            "${ASCEND_ROOT}/include"
            "${ASCEND_ROOT}/pkg_inc"
            "${ASCEND_ROOT}/pkg_inc/runtime"
            "${ASCEND_ROOT}/aarch64-linux/pkg_inc"
            "${ASCEND_ROOT}/aarch64-linux/pkg_inc/runtime"
        NO_DEFAULT_PATH
    )
    if(NOT ASCEND_RUNTIME_LIBRARY OR NOT ASCEND_FFTS_INCLUDE_DIR)
        message(FATAL_ERROR "UCM_ENABLE_ASCEND_IO_AGGREGATION requires FFTS headers and libruntime.")
    endif()
    get_filename_component(ASCEND_FFTS_INCLUDE_PARENT "${ASCEND_FFTS_INCLUDE_DIR}" DIRECTORY)
    set(ASCEND_FFTS_INCLUDE_DIRS "${ASCEND_FFTS_INCLUDE_DIR}")
    if(ASCEND_FFTS_INCLUDE_DIR MATCHES "/runtime$" AND EXISTS "${ASCEND_FFTS_INCLUDE_PARENT}")
        list(APPEND ASCEND_FFTS_INCLUDE_DIRS "${ASCEND_FFTS_INCLUDE_PARENT}")
    endif()
    list(APPEND ASCEND_FFTS_INCLUDE_DIRS
        "${ASCEND_ROOT}/pkg_inc"
        "${ASCEND_ROOT}/pkg_inc/toolchain"
        "${ASCEND_ROOT}/pkg_inc/profiling"
        "${ASCEND_ROOT}/aarch64-linux/pkg_inc"
        "${ASCEND_ROOT}/aarch64-linux/pkg_inc/toolchain"
        "${ASCEND_ROOT}/aarch64-linux/pkg_inc/profiling"
    )
    list(REMOVE_DUPLICATES ASCEND_FFTS_INCLUDE_DIRS)
    target_sources(trans PRIVATE
        ascend_shard_io_aggregator.cc
        ffts_d2d_dispatcher.cc
    )
    target_include_directories(trans PUBLIC ${ASCEND_FFTS_INCLUDE_DIRS})
    target_compile_definitions(trans PUBLIC UCM_ENABLE_ASCEND_IO_AGGREGATION=1)
    target_link_libraries(trans PUBLIC ${ASCEND_RUNTIME_LIBRARY})
else()
    target_compile_definitions(trans PUBLIC UCM_ENABLE_ASCEND_IO_AGGREGATION=0)
endif()
target_link_libraries(trans PUBLIC
    fmt
    Ascend::ascendcl
)
