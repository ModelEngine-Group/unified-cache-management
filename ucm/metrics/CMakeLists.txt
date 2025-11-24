cmake_minimum_required(VERSION 3.18)
project(ucm_metrics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(pybind11 REQUIRED)

pybind11_add_module(ucmmonitor
    cpy/metrics.py.cc
    cc/stats/ucm_stats.cc
    cc/stats_registry.cc
    cc/stats_monitor.cc
)

target_include_directories(ucmmonitor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cc)

set_target_properties(ucmmonitor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)