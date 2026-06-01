# Copyright (c) 2026 Huawei Technologies Co., Ltd.

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(Ascend_ROOT $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(Ascend_ROOT $ENV{ASCEND_CUSTOM_PATH}/latest)
else()
    set(Ascend_ROOT /usr/local/Ascend/ascend-toolkit/latest)
endif()

find_path(FFTS_ASCEND_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS
        ${Ascend_ROOT}/include
        ${Ascend_ROOT}/aarch64-linux/include
        ${Ascend_ROOT}/arm64-linux/include
    NO_DEFAULT_PATH
)

find_path(FFTS_RUNTIME_INCLUDE_DIR
    NAMES runtime/rt_ffts_plus.h
    HINTS
        ${Ascend_ROOT}/pkg_inc/runtime
        ${Ascend_ROOT}/aarch64-linux/pkg_inc/runtime
        ${Ascend_ROOT}/arm64-linux/pkg_inc/runtime
    NO_DEFAULT_PATH
)

find_path(FFTS_TOOLCHAIN_INCLUDE_DIR
    NAMES toolchain/prof_api.h
    HINTS
        ${Ascend_ROOT}/pkg_inc
        ${Ascend_ROOT}/aarch64-linux/pkg_inc
        ${Ascend_ROOT}/arm64-linux/pkg_inc
        ${Ascend_ROOT}/include/experiment/msprof
        ${Ascend_ROOT}/aarch64-linux/include/experiment/msprof
        ${Ascend_ROOT}/arm64-linux/include/experiment/msprof
    NO_DEFAULT_PATH
)

find_path(FFTS_PROFILING_INCLUDE_DIR
    NAMES prof_common.h
    HINTS
        ${Ascend_ROOT}/pkg_inc/profiling
        ${Ascend_ROOT}/aarch64-linux/pkg_inc/profiling
        ${Ascend_ROOT}/arm64-linux/pkg_inc/profiling
        ${Ascend_ROOT}/include/experiment/msprof/toolchain
        ${Ascend_ROOT}/aarch64-linux/include/experiment/msprof/toolchain
        ${Ascend_ROOT}/arm64-linux/include/experiment/msprof/toolchain
    NO_DEFAULT_PATH
)

find_library(FFTS_ASCENDCL_LIBRARY
    NAMES ascendcl
    HINTS
        ${Ascend_ROOT}/lib64
        ${Ascend_ROOT}/aarch64-linux/lib64
        ${Ascend_ROOT}/arm64-linux/lib64
    NO_DEFAULT_PATH
)

find_library(FFTS_RUNTIME_LIBRARY
    NAMES runtime
    HINTS
        ${Ascend_ROOT}/lib64
        ${Ascend_ROOT}/runtime/lib64
        ${Ascend_ROOT}/aarch64-linux/lib64
        ${Ascend_ROOT}/arm64-linux/lib64
    NO_DEFAULT_PATH
)

if(NOT FFTS_ASCEND_INCLUDE_DIR OR NOT FFTS_RUNTIME_INCLUDE_DIR OR NOT FFTS_TOOLCHAIN_INCLUDE_DIR OR
   NOT FFTS_PROFILING_INCLUDE_DIR OR NOT FFTS_ASCENDCL_LIBRARY OR NOT FFTS_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Ascend CANN headers/libraries required for ucm_transport_ffts were not found")
endif()

find_package(Threads REQUIRED)

add_library(ucm_transport_ffts STATIC
    src/ffts_transport.cpp
    src/ffts_engine.cpp
)

target_compile_features(ucm_transport_ffts PUBLIC cxx_std_17)

target_include_directories(ucm_transport_ffts
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(ucm_transport_ffts SYSTEM
    PRIVATE
        ${FFTS_ASCEND_INCLUDE_DIR}
        ${FFTS_RUNTIME_INCLUDE_DIR}
        ${FFTS_TOOLCHAIN_INCLUDE_DIR}
        ${FFTS_PROFILING_INCLUDE_DIR}
)

target_link_libraries(ucm_transport_ffts
    PUBLIC
        infra_status
    PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
        m
        ${FFTS_ASCENDCL_LIBRARY}
        ${FFTS_RUNTIME_LIBRARY}
)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
