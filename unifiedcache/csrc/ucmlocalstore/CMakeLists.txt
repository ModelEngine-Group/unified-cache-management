cmake_minimum_required(VERSION 3.12)
project(ucmlocalstore VERSION 25.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_RPATH TRUE)

set(BUILD_OPTS "-Wall -Werror -fPIC -Wl,-z,relro,-z,now")
set(BUILD_DEBUG_OPTS "-O0 -g")
set(BUILD_RELEASE_OPTS "-O3 -D_FORTIFY_SOURCE=2")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_OPTS} ${BUILD_DEBUG_OPTS}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_OPTS} ${BUILD_RELEASE_OPTS}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/lib)

add_subdirectory(cc)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()