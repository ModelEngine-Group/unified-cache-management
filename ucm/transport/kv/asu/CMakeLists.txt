if(NOT RUNTIME_ENVIRONMENT STREQUAL "ascend")
    message(FATAL_ERROR "BUILD_UCM_ASU requires RUNTIME_ENVIRONMENT=ascend. Current value: ${RUNTIME_ENVIRONMENT}")
endif()

if(NOT DEFINED ASCEND_ROOT)
    set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
endif()

find_path(ASU_ASCEND_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS
        ${ASCEND_ROOT}/include
        ${ASCEND_ROOT}/aarch64-linux/include
        ${ASCEND_ROOT}/arm64-linux/include
    NO_DEFAULT_PATH
)
if(NOT ASU_ASCEND_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find acl/acl.h under ASCEND_ROOT=${ASCEND_ROOT}")
endif()

find_library(ASU_ASCENDCL_LIB
    NAMES ascendcl
    HINTS
        ${ASCEND_ROOT}/lib64
        ${ASCEND_ROOT}/aarch64-linux/lib64
    NO_DEFAULT_PATH
)
if(NOT ASU_ASCENDCL_LIB)
    message(FATAL_ERROR "Cannot find libascendcl under ASCEND_ROOT=${ASCEND_ROOT}")
endif()

message(STATUS "ASU ASCEND include: ${ASU_ASCEND_INCLUDE_DIR}")
message(STATUS "ASU ASCENDCL lib: ${ASU_ASCENDCL_LIB}")

add_library(asu_ascend_deps INTERFACE)
target_include_directories(asu_ascend_deps INTERFACE ${ASU_ASCEND_INCLUDE_DIR})
target_link_libraries(asu_ascend_deps INTERFACE ${ASU_ASCENDCL_LIB})

file(GLOB ASU_COMMON_SOURCES CONFIGURE_DEPENDS common/*.cpp)
file(GLOB ASU_TRANSPORT_SOURCES CONFIGURE_DEPENDS trans/src/*.cpp)
file(GLOB LOGGER_SOURCES CONFIGURE_DEPENDS 
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/*.cc
    ${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc/*.cc
)
list(APPEND ASU_TRANSPORT_SOURCES ${ASU_COMMON_SOURCES})
list(APPEND ASU_TRANSPORT_SOURCES ${LOGGER_SOURCES})
list(APPEND ASU_TRANSPORT_SOURCES ${UCM_ROOT_DIR}/ucm/shared/trans/ascend/ascend_buffer.cc)
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
        pthread fmt spdlog z
    PRIVATE
        asu_ascend_deps
)

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
        ${UCM_ROOT_DIR}/ucm/transport/kv/common/include
)
target_link_libraries(asu_client PUBLIC asu_transport kv_common pthread)
target_compile_definitions(asu_client PRIVATE SPDLOG_FMT_EXTERNAL)

if(BUILD_UNIT_TESTS)
    target_compile_definitions(asu_transport PRIVATE ASU_BUILD_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE ASU_TEST_SOURCE_FILES CONFIGURE_DEPENDS test/case/*.cc)
    file(GLOB_RECURSE ASU_TRANSPORT_TEST_SOURCE_FILES CONFIGURE_DEPENDS trans/test/*.cpp)
    list(APPEND ASU_TEST_SOURCE_FILES ${ASU_TRANSPORT_TEST_SOURCE_FILES})
    file(GLOB_RECURSE ASU_CLIENT_TEST_SOURCE_FILES CONFIGURE_DEPENDS client/test/*.cpp)
    list(APPEND ASU_TEST_SOURCE_FILES ${ASU_CLIENT_TEST_SOURCE_FILES})
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
        gtest_main gtest
    )
    gtest_discover_tests(asu.test)
endif()
