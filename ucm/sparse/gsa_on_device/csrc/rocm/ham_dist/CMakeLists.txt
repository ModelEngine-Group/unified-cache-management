message(STATUS "Building ham_dist (hamming HIP/ROCm extension)...")

# The project-global release flags strip all symbols (-s), which would remove
# the CPython module init entry point from this extension and make it
# unloadable. Drop -s for this subtree so the .so keeps PyInit_hamming.
string(REPLACE "-s " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "(^| )-s$" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects the
# host GPU(s) via rocm_agent_enumerator and errors if none is found (a no-GPU
# build host must then set the arch explicitly).
enable_language(HIP)
find_package(hip REQUIRED)

set(CUDA_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../cuda/ham_dist)
set(HIP_COMPAT_DIR ${UCM_ROOT_DIR}/ucm/shared/vendor/hip_compat)

# ---- Python ----
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
    OUTPUT_VARIABLE PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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

set(INCLUDE_DIRS
    ${HIP_COMPAT_DIR}
    ${CUDA_SRC_DIR}
    ${PYTORCH_PATH}/include
    ${PYTORCH_PATH}/include/torch/csrc/api/include
)

set(LIBRARY_DIRS
    ${PYTORCH_PATH}/lib
)

set(LIBRARIES
    torch
    c10
    c10_hip
    torch_cpu
    torch_hip
    torch_python
    pthread
    hip::host
)

set_source_files_properties(${CUDA_SRC_DIR}/paged_ham_dist_mla.cu PROPERTIES LANGUAGE HIP)
# NO_EXTRAS disables pybind11's default LTO + strip, which under
# -fvisibility=hidden would drop the CPython module init symbol and make the
# extension unloadable.
pybind11_add_module(hamming NO_EXTRAS
    ${CUDA_SRC_DIR}/cpy/hamming.cpp
    ${CUDA_SRC_DIR}/paged_ham_dist_mla.cu
)

set_target_properties(hamming PROPERTIES
    PREFIX ""
    SUFFIX "${PY_EXT_SUFFIX}"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    HIP_STANDARD 20
)

target_include_directories(hamming PRIVATE ${INCLUDE_DIRS})
target_link_directories(hamming PRIVATE ${LIBRARY_DIRS})
target_link_libraries(hamming PRIVATE ${LIBRARIES} Python::Module)

# torch on ROCm keeps the cuda spelling and selects the HIP backend via these
# macros; the half-operator unmacros are nvcc-only and unneeded under hipcc.
target_compile_definitions(hamming PRIVATE
    TORCH_EXTENSION_NAME=hamming
    __HIP_PLATFORM_AMD__
    USE_ROCM
    _GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}
)

target_compile_options(hamming PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    # The kernel uses INFINITY as a masking sentinel, so do not enable clang
    # -ffast-math (it implies -ffinite-math-only and would discard the inf).
    # torch 2.x headers use C++20 constructs; do not let the project-global
    # -Werror promote warnings from third-party headers to errors here.
    -Wno-error
)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CUDA_SRC_DIR}
)
install(TARGETS hamming LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
