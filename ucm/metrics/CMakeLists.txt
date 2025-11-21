#-------------- 基础要求 --------------
cmake_minimum_required(VERSION 3.16)
project(ucm_metrics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-------------- 找 pybind11 --------------
# 1) 系统 apt 安装：apt install pybind11-dev
# 2) 或 pip 安装：pip install pybind11
find_package(pybind11 REQUIRED)

#-------------- 自动注册函数（你原来的不动）--------------
function(generate_auto_register target_name stats_dir)
    file(GLOB_RECURSE STATS_HEADERS "${stats_dir}/*.h")
    set(REGISTER_CODE "")
    foreach(header ${STATS_HEADERS})
        file(STRINGS "${header}" REGISTER_LINE REGEX "REGISTER_STATS\\([A-Za-z0-9_]+\\)")
        if(REGISTER_LINE)
            string(REGEX MATCH "REGISTER_STATS\\(([A-Za-z0-9_]+)\\)" _ ${REGISTER_LINE})
            set(class_name ${CMAKE_MATCH_1})
            set(REGISTER_CODE "${REGISTER_CODE}
#include \"${header}\"
template class StatsRegistrar<${class_name}>;
static StatsRegistrar<${class_name}> ${class_name}_registrar;")
        endif()
    endforeach()
    set(AUTO_REGISTER_FILE "${CMAKE_BINARY_DIR}/auto_stats_registrar.cpp")
    file(WRITE "${AUTO_REGISTER_FILE}" "${REGISTER_CODE}")
    target_sources(${target_name} PRIVATE "${AUTO_REGISTER_FILE}")
endfunction()

#-------------- 静态库：metrics --------------
add_library(metrics STATIC
    cc/stats_monitor.cc
)

# 强制 PIC，以便后续链接到 .so
set_property(TARGET metrics PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(metrics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc        # 让 #include "stats/xxx.h" 能找到
)

# 扫描 cc/stats 目录自动生成注册代码
generate_auto_register(metrics ${CMAKE_CURRENT_SOURCE_DIR}/cc/stats)

#-------------- pybind11 模块：ucmmonitor --------------
pybind11_add_module(ucmmonitor cpy/metrics.py.cc)
target_link_libraries(ucmmonitor PRIVATE metrics)

# 可选：把 .so 生成到项目根目录，方便 import
set_target_properties(ucmmonitor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)