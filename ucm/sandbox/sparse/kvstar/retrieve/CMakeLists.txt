cmake_minimum_required(VERSION 3.18)
project(kvstar_retrieve_clib)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|AMD64")
    include(CheckCXXCompilerFlag)

    check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

    if(COMPILER_SUPPORTS_F16C AND COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c -mavx2")
        message(STATUS "F16C and AVX2 instruction sets enabled")

        add_definitions(-DUSE_F16C=1)
    else()
        message(STATUS "Compiler does not support F16C/AVX2, using software implementation")
        add_definitions(-DUSE_F16C=0)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|armv7|armv8")
    message(STATUS "Detected ARM platform - adding aggressive optimization flags")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
          -Ofast
          -DNDEBUG
          -march=native
          -Wall
          -ffast-math
        )
    endif()
else()
    message(STATUS "Non-x86 architecture, F16C not applicable")
    add_definitions(-DUSE_F16C=0)
endif()

FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        10.2.1
)

set(FMT_TEST OFF CACHE BOOL "Disable testing for fmt")
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1
)

set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable testing for spdlog")

set(SPDLOG_USE_EXTERNAL_FMT ON CACHE BOOL "Use external fmt library")
FetchContent_MakeAvailable(spdlog)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
find_package(pybind11 REQUIRED)

set(NUMA_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/numa_install)

FetchContent_Declare(
    numactl
    URL https://github.com/numactl/numactl/releases/download/v2.0.16/numactl-2.0.16.tar.gz
    TLS_VERIFY OFF
)

FetchContent_MakeAvailable(numactl)

if(NOT EXISTS "${NUMA_INSTALL_DIR}/lib/libnuma.so")
    message(STATUS "Configuring numactl...")
    execute_process(
        COMMAND ./configure --prefix=${NUMA_INSTALL_DIR}
        WORKING_DIRECTORY ${numactl_SOURCE_DIR}
        RESULT_VARIABLE numa_configure_result
        OUTPUT_VARIABLE numa_configure_output
        ERROR_VARIABLE numa_configure_error
    )
    if(NOT numa_configure_result EQUAL 0)
        message(FATAL_ERROR "Failed to configure numactl. \n"
                            "Result: ${numa_configure_result}\n"
                            "STDOUT: ${numa_configure_output}\n"
                            "STDERR: ${numa_configure_error}\n")
    endif()

    message(STATUS "Building and installing numactl...")
    execute_process(
        COMMAND make install -j8
        WORKING_DIRECTORY ${numactl_SOURCE_DIR}
        RESULT_VARIABLE numa_install_result
        OUTPUT_VARIABLE numa_install_output
        ERROR_VARIABLE numa_install_error
    )
    if(NOT numa_install_result EQUAL 0)
        message(FATAL_ERROR "Failed to build and install numactl. \n"
                            "Result: ${numa_install_result}\n"
                            "STDOUT: ${numa_install_output}\n"
                            "STDERR: ${numa_install_error}\n")
    endif()
else()
    message(STATUS "Found already built libnuma. Skipping build.")
endif()

add_subdirectory(core)
add_subdirectory(py_intf)

set_target_properties(kvstar_retrieve PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
