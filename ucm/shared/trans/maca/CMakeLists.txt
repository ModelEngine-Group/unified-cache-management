set(CUDA_ROOT "/opt/maca/tools/cu-bridge" CACHE PATH "Path to WCUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/cucc)
list(APPEND CMAKE_MODULE_PATH "${CUDA_ROOT}/cmake_module/maca")
enable_language(CUDA)
add_library(kernel OBJECT maca_sm_kernel.cu)
target_compile_options(kernel PRIVATE
    -Wall -fPIC
    -std=c++17
)
add_library(trans STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_device.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_buffer.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_stream.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_sm_stream.cc
)

add_library(WCUDA::cudart UNKNOWN IMPORTED)
set_target_properties(WCUDA::cudart PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${CUDA_ROOT}/include"
	IMPORTED_LOCATION "${CUDA_ROOT}/lib/libcuda.so"
)
target_include_directories(WCUDA::cudart INTERFACE
	/opt/maca/include
	/opt/maca/include/mcr
)

target_include_directories(trans PUBLIC ${CUDA_ROOT}/include)
target_link_directories(trans PUBLIC ${CUDA_ROOT}/lib64)
target_link_libraries(trans PUBLIC
    fmt
    WCUDA::cudart
    kernel
)
