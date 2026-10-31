message(STATUS "[INFO] Building gsa_prefetch for device: ${RUNTIME_ENVIRONMENT}")

# Enable C language for Huawei Secure C
enable_language(C)

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

# 设置基础编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")
set(CXX11_ABI "1")

# 根据设备类型配置
set(INCLUDE_DIRS
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${PYTORCH_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${UCM_ROOT_DIR}/ucm/store
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
    storeintf
    /usr/local/Ascend/driver/lib64/common/libsecurec.so
)

# 定义源文件
set(SOURCES
    src/pybinds.cpp
    src/kvcache_pre.cpp
    src/kvcache_pre_scatter.cpp
    src/kvcache_task_manger.cpp
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
    else()
        message(WARNING "torch_npu not found, but RUNTIME_ENVIRONMENT is set to ascend")
    endif()
    set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
    add_library(Ascend::ascendcl UNKNOWN IMPORTED)
    set_target_properties(Ascend::ascendcl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
        IMPORTED_LOCATION "${ASCEND_ROOT}/lib64/libascendcl.so"
    )
    add_library(Ascend::opapi UNKNOWN IMPORTED)
    set_target_properties(Ascend::opapi PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
        IMPORTED_LOCATION "${ASCEND_ROOT}/aarch64-linux/lib64/libopapi.so"
    )
    list(INSERT SOURCES 0 ascend/kvcache_trans_ascend.cpp)
    list(INSERT INCLUDE_DIRS 0 ${CMAKE_CURRENT_SOURCE_DIR}/ascend)
    list(INSERT LIBRARIES 0 Ascend::ascendcl)
    list(INSERT LIBRARIES 0 Ascend::opapi)
else()
    message(FATAL_ERROR "RUNTIME_ENVIRONMENT must be 'ascend' (CUDA backend was removed)")
endif()

# 设置CXX11_ABI宏
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")

# 查找OpenMP
find_package(OpenMP REQUIRED)

# 创建pybind11模块
pybind11_add_module(gsa_prefetch ${SOURCES})
get_target_property(SPDLOG_INCLUDE_DIRS spdlog INTERFACE_INCLUDE_DIRECTORIES)
if(SPDLOG_INCLUDE_DIRS)
    target_include_directories(gsa_prefetch BEFORE PRIVATE ${SPDLOG_INCLUDE_DIRS})
endif()

get_target_property(FMT_INCLUDE_DIRS fmt INTERFACE_INCLUDE_DIRECTORIES)
if(FMT_INCLUDE_DIRS)
    target_include_directories(gsa_prefetch BEFORE PRIVATE ${FMT_INCLUDE_DIRS})
endif()
# 设置头文件目录
target_include_directories(gsa_prefetch PRIVATE ${INCLUDE_DIRS})

# 设置库文件目录
target_link_directories(gsa_prefetch PRIVATE ${LIBRARY_DIRS})

# 链接库
target_link_libraries(gsa_prefetch PRIVATE ${LIBRARIES})

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(gsa_prefetch PRIVATE OpenMP::OpenMP_CXX)
endif()

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS gsa_prefetch LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

# 单元/模块/端到端测试套件：仅在开启 BUILD_UNIT_TESTS 时构建，使用 ACL stub + mock store，
# 不依赖 Ascend NPU / torch_npu 环境。
if(BUILD_UNIT_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
