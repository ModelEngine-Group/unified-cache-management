if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # 独立编译模式：作为顶层 project 处理
    cmake_minimum_required(VERSION 3.18)
    project(kv_semantics LANGUAGES CXX VERSION 1.0.0)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(RUNTIME_ENVIRONMENT "ascend" CACHE STRING "runtime: simu or ascend.")
    option(BUILD_UNIT_TESTS "build kv_semantics unit tests." OFF)
    option(DOWNLOAD_DEPENDENCE "download deps via FetchContent." ON)

    set(FLAGS_PUBLIC "-Wall -Werror -fPIC -Wl,-z,relro,-z,now")
    set(FLAGS_DEBUG "-O0 -g")
    set(FLAGS_RELEASE "-s -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2")
    string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
    if(CMAKE_BUILD_TYPE_LOWER STREQUAL "debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_DEBUG}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_RELEASE}")
    endif()

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor/dep-fmt.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor/dep-spdlog.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor/dep-zlib.cmake)
    if(BUILD_UNIT_TESTS)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vendor/dep-googletest.cmake)
        enable_testing()
    endif()
endif()

if(NOT (RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "simu"))
    message(FATAL_ERROR
        "BUILD_KV_CLIENT requires RUNTIME_ENVIRONMENT=ascend or simu. Current value: ${RUNTIME_ENVIRONMENT}")
endif()

option(BUILD_KV_CLIENT_PROVIDER_AICPU "Build KV AICPU trans provider" OFF)
option(BUILD_KV_CLIENT_PROVIDER_FAKE "Build KV fake trans provider" ON)
option(BUILD_KV_CLIENT_PROVIDER_AIV "Build KV AIV trans provider (requires external libumc.a)" OFF)

add_subdirectory(src)

if(BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(kv_test)