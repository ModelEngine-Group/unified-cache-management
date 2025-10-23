set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)
add_library(storedevice STATIC cuda_device.cu)
target_link_libraries(storedevice PUBLIC storeinfra)
target_compile_options(storedevice PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)
