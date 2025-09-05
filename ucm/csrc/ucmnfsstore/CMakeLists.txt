cmake_minimum_required(VERSION 3.12)
project(ucmnfsstore VERSION 25.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "build all test suits." OFF)
option(COVERAGE_ENABLED "coverage enabled." OFF)
option(DOWNLOAD_DEPENDENCE "download dependence by cmake." ON)
set(RUNTIME_ENVIRONMENT "ascend" CACHE STRING "runtime: simu, ascend or cuda.")

set(UCMNFSSTORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include(${UCMNFSSTORE_ROOT}/cmake/variable.cmake)
include(${UCMNFSSTORE_ROOT}/cmake/flags.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${UCMNFSSTORE_ROOT}/output/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${UCMNFSSTORE_ROOT}/output/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${UCMNFSSTORE_ROOT}/output/lib)

if(DOWNLOAD_DEPENDENCE)
    add_subdirectory(vendor)
endif()
add_subdirectory(cc)
add_subdirectory(cpy)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
