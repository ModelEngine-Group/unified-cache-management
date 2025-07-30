
add_library(Cuda::cudart UNKNOWN IMPORTED)

add_compile_definitions(CUDA_AVAILABLE=1)
