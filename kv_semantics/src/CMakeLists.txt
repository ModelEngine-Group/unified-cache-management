find_package(Threads REQUIRED)

add_subdirectory(common/runtime)

file(GLOB KV_COMMON_SOURCES CONFIGURE_DEPENDS common/utils/*.cc)
file(GLOB KV_TRANSPORT_SOURCES CONFIGURE_DEPENDS
    trans/*.cc
    trans/buffer/*.cc
    trans/conn/*.cc
    trans/proto/*.cc
    trans/task/*.cc
    trans/utils/*.cc
    trans_provider/*.cc
    trans_provider/aicpu/*.cc
    trans_provider/aiv/*.cc
    trans_provider/fake/*.cc
    common/logger/*.cc
)
list(APPEND KV_TRANSPORT_SOURCES ${KV_COMMON_SOURCES})
add_library(kv_transport SHARED ${KV_TRANSPORT_SOURCES})
target_include_directories(kv_transport
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/trans
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_provider
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ds
        ${CMAKE_CURRENT_SOURCE_DIR}/common/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/common/logger
        ${CMAKE_CURRENT_SOURCE_DIR}/trans
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_provider
)
target_compile_definitions(kv_transport PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(kv_transport
    PUBLIC
        pthread fmt spdlog zlibstatic
    PRIVATE
        kv_buffer_runtime
)

if(BUILD_KV_CLIENT_PROVIDER_AICPU)
    if(NOT DEFINED ASU_HCOMM_ROOT)
        if(DEFINED ENV{ASU_HCOMM_ROOT})
            set(ASU_HCOMM_ROOT "$ENV{ASU_HCOMM_ROOT}" CACHE PATH "Install prefix of hcomm")
        elseif(DEFINED ENV{HCOMM_ROOT})
            set(ASU_HCOMM_ROOT "$ENV{HCOMM_ROOT}" CACHE PATH "Install prefix of hcomm")
        else()
            set(ASU_HCOMM_ROOT "${ASCEND_ROOT}" CACHE PATH "Install prefix of hcomm")
        endif()
    endif()

    find_path(ASU_HCOMM_INCLUDE_DIR
        NAMES hcomm/hcomm_res.h hcomm_res.h
        HINTS
            ${ASU_HCOMM_ROOT}/include
            ${ASU_HCOMM_ROOT}/include/hcomm
            ${ASU_HCOMM_ROOT}/aarch64-linux/include
            ${ASU_HCOMM_ROOT}/aarch64-linux/include/hcomm
            ${ASU_HCOMM_ROOT}/arm64-linux/include
            ${ASU_HCOMM_ROOT}/arm64-linux/include/hcomm
        NO_DEFAULT_PATH
    )
    if(NOT ASU_HCOMM_INCLUDE_DIR)
        message(FATAL_ERROR "Cannot find hcomm_res.h under ASU_HCOMM_ROOT=${ASU_HCOMM_ROOT}")
    endif()

    find_library(ASU_HCOMM_LIB
        NAMES hcomm
        HINTS
            ${ASU_HCOMM_ROOT}/lib
            ${ASU_HCOMM_ROOT}/lib64
            ${ASU_HCOMM_ROOT}/aarch64-linux/lib
            ${ASU_HCOMM_ROOT}/aarch64-linux/lib64
            ${ASU_HCOMM_ROOT}/arm64-linux/lib
            ${ASU_HCOMM_ROOT}/arm64-linux/lib64
        NO_DEFAULT_PATH
    )
    if(NOT ASU_HCOMM_LIB)
        message(FATAL_ERROR "Cannot find libhcomm under ASU_HCOMM_ROOT=${ASU_HCOMM_ROOT}")
    endif()

    message(STATUS "ASU HCOMM include: ${ASU_HCOMM_INCLUDE_DIR}")
    message(STATUS "ASU HCOMM lib: ${ASU_HCOMM_LIB}")
    target_include_directories(kv_transport PRIVATE ${ASU_HCOMM_INCLUDE_DIR})
    target_link_libraries(kv_transport PRIVATE ${ASU_HCOMM_LIB} ${CMAKE_DL_LIBS})
    target_compile_definitions(kv_transport PRIVATE UCM_ASU_ENABLE_AICPU_PROVIDER=1)
endif()

if(BUILD_KV_CLIENT_PROVIDER_FAKE)
    target_compile_definitions(kv_transport PRIVATE UCM_ASU_ENABLE_FAKE_PROVIDER=1)
endif()

if(BUILD_KV_CLIENT_PROVIDER_AIV)
    if(NOT DEFINED KV_CLIENT_AIV_PROVIDER_ROOT)
        if(DEFINED ENV{KV_CLIENT_AIV_PROVIDER_ROOT})
            set(KV_CLIENT_AIV_PROVIDER_ROOT "$ENV{KV_CLIENT_AIV_PROVIDER_ROOT}" CACHE PATH "Install prefix of libumc.a")
        else()
            set(KV_CLIENT_AIV_PROVIDER_ROOT "" CACHE PATH "Install prefix of libumc.a")
        endif()
    endif()

    find_library(ASU_AIV_PROVIDER_LIB
        NAMES libumc.a umc
        HINTS
            ${KV_CLIENT_AIV_PROVIDER_ROOT}/lib
            ${KV_CLIENT_AIV_PROVIDER_ROOT}/lib64
        NO_DEFAULT_PATH
    )
    if(NOT ASU_AIV_PROVIDER_LIB)
        message(FATAL_ERROR "Cannot find libumc.a under KV_CLIENT_AIV_PROVIDER_ROOT=${KV_CLIENT_AIV_PROVIDER_ROOT}")
    endif()

    message(STATUS "KV AIV provider lib: ${ASU_AIV_PROVIDER_LIB}")
    target_link_libraries(kv_transport PRIVATE ${ASU_AIV_PROVIDER_LIB} ${CMAKE_DL_LIBS})
    target_compile_definitions(kv_transport PRIVATE UCM_ASU_ENABLE_AIV_PROVIDER=1)
endif()

file(GLOB_RECURSE KV_CLIENT_SOURCES CONFIGURE_DEPENDS client/*.cc)
add_library(kv_client SHARED ${KV_CLIENT_SOURCES})
target_include_directories(kv_client
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(kv_client
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ds
        ${CMAKE_CURRENT_SOURCE_DIR}/common/logger
        ${CMAKE_CURRENT_SOURCE_DIR}/common/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/client
)
target_link_libraries(kv_client PUBLIC kv_transport pthread)
target_compile_definitions(kv_client PRIVATE SPDLOG_FMT_EXTERNAL)

set(INSTALL_REL_PATH "kv_semantics/src")
set_target_properties(kv_client PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(kv_transport PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS kv_client kv_transport LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
install(TARGETS kv_client kv_transport LIBRARY DESTINATION ucm/store/asu COMPONENT ucm)
