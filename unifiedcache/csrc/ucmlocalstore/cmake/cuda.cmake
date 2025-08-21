set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")

add_library(Cuda::cudart UNKNOWN IMPORTED)

set_target_properties(Cuda::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_ROOT}/include"
    IMPORTED_LOCATION "${CUDA_ROOT}/lib64/libcudart.so"
)

add_compile_definitions(CUDA_AVAILABLE=1)