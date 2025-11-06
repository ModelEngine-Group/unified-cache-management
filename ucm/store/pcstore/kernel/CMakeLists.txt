set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)
add_library(pcstoredevice STATIC device.cu stream.cu buffer.cu)
target_link_libraries(pcstoredevice PUBLIC storeinfra nvidia-ml)
target_compile_options(pcstoredevice PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)
