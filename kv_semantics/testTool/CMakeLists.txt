# ============================================================================
# testTool — kv_semantics 测试工具
# 构建（Docker 开发容器内，工作目录为 testTool/）：
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
#   cmake --build build -j
#   ./build/testTool
# ============================================================================

# ---------- 工程基础信息：声明 CMake 最低版本与工程名称、语言、版本号 ----------
cmake_minimum_required(VERSION 3.18)
project(testTool LANGUAGES CXX VERSION 0.1.0)

# ---------- 全局编译标准：C++17，禁用编译器方言扩展，保证跨编译器行为一致 ----------
# CMAKE_EXPORT_COMPILE_COMMANDS：导出 compile_commands.json，供 clangd / IDE 代码分析使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- 构建选项：第三方依赖获取方式（ON=联网 FetchContent 拉取；OFF=使用本目录下预置源码目录） ----------
option(DOWNLOAD_DEPENDENCE "download deps via FetchContent." ON)

# ---------- 编译选项：与主工程 kv_semantics 保持一致 ----------
# FLAGS_PUBLIC：告警与安全加固（-Wall -Werror 视告警为错误、-fPIC、链接期 relro/now）
# FLAGS_DEBUG：调试构建（-O0 不优化 + -g 带调试符号）
# FLAGS_RELEASE：发布构建（-O2 优化、-s 去符号、栈保护与 _FORTIFY_SOURCE）
set(FLAGS_PUBLIC "-Wall -Werror -fPIC -Wl,-z,relro,-z,now")
set(FLAGS_DEBUG "-O0 -g")
set(FLAGS_RELEASE "-s -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2")
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_DEBUG}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_RELEASE}")
endif()

# ---------- 第三方依赖：复用父工程 cmake/vendor 依赖脚本，版本与主工程 kv_semantics 锁定一致 ----------
# dep-fmt.cmake    ：fmt 格式化库（spdlog 的外部格式化后端）
# dep-spdlog.cmake ：spdlog 日志库（对应 src/log 模块）
# 可移植性约束：代码需适配 CPU/GPU/NPU 与 ARM/x86，只允许引入纯 C++、
# 跨架构可移植的库（fmt/spdlog 均满足）；新增依赖前必须确认 arm64/x86_64 均可编译。
set(TESTTOOL_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/vendor)
list(APPEND CMAKE_MODULE_PATH ${TESTTOOL_VENDOR_DIR})
include(${TESTTOOL_VENDOR_DIR}/dep-fmt.cmake)
include(${TESTTOOL_VENDOR_DIR}/dep-spdlog.cmake)

# ---------- 系统库：线程支持（src/worker 并发模块需要 pthread） ----------
find_package(Threads REQUIRED)

# ---------- 源码收集：递归 GLOB src/ 下所有 .cc/.cpp 源文件 ----------
# CONFIGURE_DEPENDS：新增/删除源文件后，下次构建自动重新配置，无需手改本文件
file(GLOB_RECURSE TESTTOOL_SOURCES CONFIGURE_DEPENDS
    src/*.cc
    src/*.cpp
)

# ---------- 可执行目标：testTool 主程序（源码来自上一步收集结果） ----------
add_executable(testTool ${TESTTOOL_SOURCES})

# ---------- 头文件搜索路径：src/ 为模块根目录 ----------
# 支持以 "log/xxx.h"、"utils/xxx.h" 模块前缀形式进行跨模块引用，保持 include 语义清晰
target_include_directories(testTool
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---------- 编译宏：SPDLOG_FMT_EXTERNAL 指定 spdlog 使用外部 fmt（与主工程用法一致） ----------
target_compile_definitions(testTool PRIVATE SPDLOG_FMT_EXTERNAL)

# ---------- 链接库：fmt（格式化）、spdlog（日志）、Threads（线程） ----------
target_link_libraries(testTool
    PRIVATE
        fmt
        spdlog
        Threads::Threads
)

# ---------- kv client 依赖：头文件取自 ../include，库取自 kv client 顶层预构建产物 ----------
# 前提：先独立构建 kv client 库（FAKE provider 默认编入，产物为 libkv_client.so / libkv_transport.so）：
#   cmake -S .. -B ../build-kv -DRUNTIME_ENVIRONMENT=simu -DCMAKE_BUILD_TYPE=Debug
#   cmake --build ../build-kv --target kv_client -j
target_include_directories(testTool
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_directories(testTool
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../build-kv/src
)
target_link_libraries(testTool
    PRIVATE
        kv_client
)
