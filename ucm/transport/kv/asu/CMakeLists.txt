if(NOT (RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "simu"))
    message(FATAL_ERROR
        "BUILD_UCM_ASU requires RUNTIME_ENVIRONMENT=ascend or simu. Current value: ${RUNTIME_ENVIRONMENT}")
endif()

option(BUILD_UCM_ASU_PROVIDER_AICPU "Build ASU AICPU trans provider" OFF)
option(BUILD_UCM_ASU_PROVIDER_FAKE "Build ASU fake trans provider" ON)
option(BUILD_UCM_ASU_PROVIDER_AIV "Build ASU AIV trans provider (requires external libumc.a)" OFF)

file(GLOB ASU_COMMON_SOURCES CONFIGURE_DEPENDS common/*.cpp)
file(GLOB ASU_TRANSPORT_SOURCES CONFIGURE_DEPENDS trans/src/*.cpp)
file(GLOB LOGGER_SOURCES CONFIGURE_DEPENDS
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/*.cc
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc/*.cc
)
list(APPEND ASU_TRANSPORT_SOURCES ${ASU_COMMON_SOURCES})
list(APPEND ASU_TRANSPORT_SOURCES ${LOGGER_SOURCES})
add_library(asu_transport SHARED ${ASU_TRANSPORT_SOURCES})
target_include_directories(asu_transport
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
        ${UCM_ROOT_DIR}/ucm/shared/infra/logger
        ${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc
        ${UCM_ROOT_DIR}/ucm/shared/infra/status
        ${UCM_ROOT_DIR}/ucm/shared
)
target_compile_definitions(asu_transport PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(asu_transport
    PUBLIC
        pthread fmt spdlog zlibstatic trans
)

if(BUILD_UCM_ASU_PROVIDER_AICPU)
    target_compile_definitions(asu_transport PRIVATE UCM_ASU_ENABLE_AICPU_PROVIDER=1)
endif()

if(BUILD_UCM_ASU_PROVIDER_FAKE)
    target_compile_definitions(asu_transport PRIVATE UCM_ASU_ENABLE_FAKE_PROVIDER=1)
endif()

if(BUILD_UCM_ASU_PROVIDER_AIV)
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

    message(STATUS "ASU AIV provider lib: ${ASU_AIV_PROVIDER_LIB}")
    target_link_libraries(asu_transport PRIVATE ${ASU_AIV_PROVIDER_LIB} ${CMAKE_DL_LIBS})
    target_compile_definitions(asu_transport PRIVATE UCM_ASU_ENABLE_AIV_PROVIDER=1)
endif()

file(GLOB ASU_CLIENT_SOURCES CONFIGURE_DEPENDS client/src/*.cpp)
add_library(asu_client SHARED ${ASU_CLIENT_SOURCES})
target_include_directories(asu_client
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/client/include
)
target_include_directories(asu_client
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/client/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_link_libraries(asu_client PUBLIC asu_transport router pthread)
target_compile_definitions(asu_client PRIVATE SPDLOG_FMT_EXTERNAL)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(asu_client PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(asu_transport PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS asu_client asu_transport LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
install(TARGETS asu_client asu_transport LIBRARY DESTINATION ucm/store/asu COMPONENT ucm)

if(BUILD_UNIT_TESTS)
    target_compile_definitions(asu_transport PRIVATE ASU_BUILD_TESTS)
    include(GoogleTest)
    file(GLOB ASU_TEST_SOURCE_FILES CONFIGURE_DEPENDS
        test/client/*.cpp
        test/transport/*.cpp
    )
    add_executable(asu.test ${ASU_TEST_SOURCE_FILES})
    target_compile_definitions(asu.test PRIVATE ASU_BUILD_TESTS)
    target_include_directories(asu.test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/client/src
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
    )
    target_link_libraries(asu.test PRIVATE
        asu_client
        gtest
    )
    gtest_discover_tests(asu.test)
endif()
