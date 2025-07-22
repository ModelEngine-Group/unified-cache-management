cmake_minimum_required(VERSION 3.12)
project(ucmnfsstore VERSION 25.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "build all test suits." OFF)
option(DOWNLOAD_DEPENDENCE "download dependence by cmake." ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/flags.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/output/lib)

if(DOWNLOAD_DEPENDENCE)
    add_subdirectory(vendor)
endif()
add_subdirectory(cc)
add_subdirectory(cpy)
if(BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif(BUILD_TESTS)
