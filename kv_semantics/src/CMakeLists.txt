find_package(Threads REQUIRED)

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
)
file(GLOB LOGGER_SOURCES CONFIGURE_DEPENDS
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/*.cc
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc/*.cc
)
list(APPEND KV_TRANSPORT_SOURCES ${KV_COMMON_SOURCES})
list(APPEND KV_TRANSPORT_SOURCES ${LOGGER_SOURCES})
add_library(kv_transport SHARED ${KV_TRANSPORT_SOURCES})
target_include_directories(kv_transport
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/trans
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_provider
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/common/ds
        ${CMAKE_CURRENT_SOURCE_DIR}/trans
        ${CMAKE_CURRENT_SOURCE_DIR}/trans_provider
        ${UCM_ROOT_DIR}/ucm/shared/infra
        ${UCM_ROOT_DIR}/ucm/shared/infra/logger
        ${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc
        ${UCM_ROOT_DIR}/ucm/shared/infra/status
        ${UCM_ROOT_DIR}/ucm/shared
)
target_compile_definitions(kv_transport PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(kv_transport
    PUBLIC
        pthread fmt spdlog zlibstatic trans
)

if(BUILD_KV_CLIENT_PROVIDER_AICPU)
    target_compile_definitions(kv_transport PRIVATE UCM_ASU_ENABLE_AICPU_PROVIDER=1)
endif()

if(BUILD_KV_CLIENT_PROVIDER_FAKE)
    target_compile_definitions(kv_transport PRIVATE UCM_ASU_ENABLE_FAKE_PROVIDER=1)
endif()

if(BUILD_KV_CLIENT_PROVIDER_AIV)
    if(NOT DEFINED ASU_AIV_PROVIDER_ROOT)
        if(DEFINED ENV{ASU_AIV_PROVIDER_ROOT})
            set(ASU_AIV_PROVIDER_ROOT "$ENV{ASU_AIV_PROVIDER_ROOT}" CACHE PATH "Install prefix of libumc.a")
        else()
            set(ASU_AIV_PROVIDER_ROOT "" CACHE PATH "Install prefix of libumc.a")
        endif()
    endif()

    find_library(ASU_AIV_PROVIDER_LIB
        NAMES libumc.a umc
        HINTS
            ${ASU_AIV_PROVIDER_ROOT}/lib
            ${ASU_AIV_PROVIDER_ROOT}/lib64
        NO_DEFAULT_PATH
    )
    if(NOT ASU_AIV_PROVIDER_LIB)
        message(FATAL_ERROR "Cannot find libumc.a under ASU_AIV_PROVIDER_ROOT=${ASU_AIV_PROVIDER_ROOT}")
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
        ${CMAKE_CURRENT_SOURCE_DIR}/client
        ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_link_libraries(kv_client PUBLIC kv_transport pthread)
target_compile_definitions(kv_client PRIVATE SPDLOG_FMT_EXTERNAL)

if(BUILD_UNIT_TESTS)
    target_compile_definitions(kv_transport PRIVATE ASU_BUILD_TESTS)
endif()

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kv_client PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(kv_transport PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS kv_client kv_transport LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
install(TARGETS kv_client kv_transport LIBRARY DESTINATION ucm/store/asu COMPONENT ucm)