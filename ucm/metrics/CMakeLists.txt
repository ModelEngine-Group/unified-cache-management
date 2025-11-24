cmake_minimum_required(VERSION 3.16)
project(ucm_metrics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(metrics STATIC
    cc/stats_monitor.cc
)

set_property(TARGET metrics PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(metrics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
)

pybind11_add_module(ucmmonitor cpy/metrics.py.cc)
target_link_libraries(ucmmonitor PRIVATE metrics)

set_target_properties(ucmmonitor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)