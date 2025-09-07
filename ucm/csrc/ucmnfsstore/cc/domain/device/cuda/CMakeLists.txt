enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
add_library(device STATIC cuda_device.cu)
target_link_libraries(device PUBLIC infra)
target_compile_options(device PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)
