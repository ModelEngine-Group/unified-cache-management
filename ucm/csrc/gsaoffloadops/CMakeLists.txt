cmake_minimum_required(VERSION 3.16)
project(gsa_offload_ops)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 从环境变量获取设备类型，默认为cuda
set(RUNTIME_ENVIRONMENT $ENV{RUNTIME_ENVIRONMENT})
if(NOT RUNTIME_ENVIRONMENT)
    set(RUNTIME_ENVIRONMENT "cuda")
endif()

message(STATUS "[INFO] Building gsa_offload_ops for device: ${RUNTIME_ENVIRONMENT}")

# 查找必要的包
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 查找PyTorch路径
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT
)

if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch installation")
endif()

# 查找pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find pybind11")
endif()

find_package(pybind11 REQUIRED)

# 设置基础编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
set(CXX11_ABI "1")

# 根据设备类型配置
set(INCLUDE_DIRS
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${PYTORCH_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(LIBRARY_DIRS
    ${PYTORCH_PATH}/lib
    /usr/local/lib
)

set(LIBRARIES
    torch
    c10
    torch_cpu
    torch_python
    gomp
    pthread
)

# NPU特殊配置
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    message(STATUS "Configuring for NPU/Ascend device")
    
    # 查找torch_npu路径
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))"
        OUTPUT_VARIABLE PYTORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NPU_RESULT
    )
    
    if(NPU_RESULT EQUAL 0)
        message(STATUS "Found torch_npu at: ${PYTORCH_NPU_PATH}")
        list(INSERT INCLUDE_DIRS 0 ${PYTORCH_NPU_PATH}/include)
        list(INSERT LIBRARY_DIRS 0 ${PYTORCH_NPU_PATH}/lib)
        list(INSERT LIBRARIES 0 torch_npu)
        set(CXX11_ABI "0")
    else()
        message(WARNING "torch_npu not found, but RUNTIME_ENVIRONMENT is set to ascend")
    endif()
endif()

# 设置CXX11_ABI宏
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")

# 查找OpenMP
find_package(OpenMP REQUIRED)

# 定义源文件
set(SOURCES
    src/thread_safe_queue.cpp
    src/vec_product.cpp
    src/k_repre.cpp
    src/select_topk_block.cpp
    src/cal_kpre_and_topk.cpp
    src/pybinds.cpp
)

# 创建pybind11模块
pybind11_add_module(gsa_offload_ops ${SOURCES})

# 设置头文件目录
target_include_directories(gsa_offload_ops PRIVATE ${INCLUDE_DIRS})

# 设置库文件目录
target_link_directories(gsa_offload_ops PRIVATE ${LIBRARY_DIRS})

# 链接库
target_link_libraries(gsa_offload_ops PRIVATE ${LIBRARIES})

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(gsa_offload_ops PRIVATE OpenMP::OpenMP_CXX)
endif()

# 确保输出目录存在
set(OUTPUT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/output/lib)
file(MAKE_DIRECTORY ${OUTPUT_LIB_DIR})

# 设置输出路径
set_target_properties(gsa_offload_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR}
)

# 编译后输出信息
add_custom_command(TARGET gsa_offload_ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Built gsa_offload_ops successfully for ${RUNTIME_ENVIRONMENT}"
    COMMAND ${CMAKE_COMMAND} -E echo "CXX11_ABI=${CXX11_ABI}"
    COMMAND ${CMAKE_COMMAND} -E echo "Output location: ${OUTPUT_LIB_DIR}"
)