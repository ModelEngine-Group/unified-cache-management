message(STATUS "Building ham_dist (hamming CUDA extension)...")

# ---- CUDA toolchain ----
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "Path to CUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)

# ---- Python ----
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ---- Python ext suffix ----
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
    OUTPUT_VARIABLE PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PY_EXT_SUFFIX='${PY_EXT_SUFFIX}'")

# ---- Locate PyTorch ----
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT
)
if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch installation via Python")
endif()
message(STATUS "Found PyTorch at: ${PYTORCH_PATH}")

# ---- Base compile flags ----
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# ---- ABI (optional) ----
set(CXX11_ABI "1")
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1)))"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_ABI_RESULT
)
if(TORCH_ABI_RESULT EQUAL 0)
    set(CXX11_ABI "${TORCH_ABI}")
endif()
message(STATUS "Using _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}")

# ---- Include dirs / Library dirs ----
set(INCLUDE_DIRS
    ${PYTORCH_PATH}/include
    ${PYTORCH_PATH}/include/torch/csrc/api/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(LIBRARY_DIRS
    ${PYTORCH_PATH}/lib
    /usr/local/lib
)

# ---- Libraries ----
set(LIBRARIES
    torch
    c10
    torch_cpu
    torch_python
    pthread
)

# ---- Build: hamming python module (.so) ----
pybind11_add_module(hamming
    cpy/hamming.cpp
    paged_ham_dist_mla.cu
)

set_target_properties(hamming PROPERTIES
    PREFIX ""
    SUFFIX "${PY_EXT_SUFFIX}"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# ---- Includes / Link dirs / Link libs ----
target_include_directories(hamming PRIVATE ${INCLUDE_DIRS})
target_link_directories(hamming PRIVATE ${LIBRARY_DIRS})
target_link_libraries(hamming PRIVATE ${LIBRARIES} Python::Module)

# ---- Extra compile options (keep your original intent) ----
target_compile_options(hamming PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>

    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
)

target_compile_definitions(hamming PRIVATE TORCH_EXTENSION_NAME=hamming)

# ---- Install ----
file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS hamming LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

message(STATUS "ham_dist target configured: ${INSTALL_REL_PATH}/hamming${PY_EXT_SUFFIX}")
