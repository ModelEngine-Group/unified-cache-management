cmake_minimum_required(VERSION 3.18)
project(kvstar_retrieve_clib)

# -- 包含 FetchContent 模块，用于在配置时下载源码 --
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")


# --- 添加 F16C 支持 ---       (针对x86，为了支持fp16计算)
# 检查是否为 x86 架构并支持 F16C
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|AMD64")
    include(CheckCXXCompilerFlag)

    # 检查编译器是否支持 F16C 指令集
    check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_F16C AND COMPILER_SUPPORTS_AVX2)
        # 添加全局编译选项
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c -mavx2")
        message(STATUS "F16C and AVX2 instruction sets enabled")

        # 添加预处理器定义以便在代码中检测
        add_definitions(-DUSE_F16C=1)
    else()
        message(STATUS "Compiler does not support F16C/AVX2, using software implementation")
        add_definitions(-DUSE_F16C=0)
    endif()
else()
    message(STATUS "Non-x86 architecture, F16C not applicable")
    add_definitions(-DUSE_F16C=0)
endif()

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
include_directories(${TORCH_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)

# 依次加载两个子模块
add_subdirectory(core)
add_subdirectory(py_intf)

#
set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/output)
set_target_properties(kvstar_retrieve PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/lib"
  RUNTIME_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/bin"
  ARCHIVE_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/lib"
)
