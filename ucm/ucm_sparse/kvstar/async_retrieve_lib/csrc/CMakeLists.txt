cmake_minimum_required(VERSION 3.18)
project(kvstar_retrieve_clib)

# -- 包含 FetchContent 模块，用于在配置时下载源码 --
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

# --- 获取并编译 {fmt} 库 ---
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        10.2.1
)
# 将FMT_TEST设置为OFF
set(FMT_TEST OFF CACHE BOOL "Disable testing for fmt")
FetchContent_MakeAvailable(fmt)

# --- 获取 spdlog 源码 ---
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1
)
# 将SPDLOG_BUILD_TESTS设置为OFF
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable testing for spdlog")
# 设置spdlog使用指定的fmt
set(SPDLOG_USE_EXTERNAL_FMT ON CACHE BOOL "Use external fmt library")
FetchContent_MakeAvailable(spdlog)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# 依次加载两个子模块
add_subdirectory(core)
add_subdirectory(py_intf)