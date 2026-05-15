set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)
add_library(kernel OBJECT cuda_sm_kernel.cu)
target_compile_options(kernel PRIVATE
    --diag-suppress=128 --diag-suppress=2417 --diag-suppress=2597
    -Wall -fPIC
)
add_library(trans STATIC
    cuda_device.cc
    cuda_buffer.cc
    cuda_stream.cc
    cuda_sm_stream.cc
    gdr/gdr_mr_buffer.cc
    gdr/gdr_config.cc
)
target_include_directories(trans PUBLIC ${CUDA_ROOT}/include)
target_link_directories(trans PUBLIC ${CUDA_ROOT}/lib64)
target_link_libraries(trans PUBLIC
    fmt
    cudart
    kernel
)
if(UCM_ENABLE_GDR_STREAM)
    find_library(IBVERBS_LIB NAMES ibverbs HINTS /usr/lib /usr/lib64 /usr/local/lib)
    if(NOT IBVERBS_LIB)
        message(FATAL_ERROR "libibverbs not found, cannot enable UCM_ENABLE_GDR_STREAM.")
    endif()
    find_package(Threads REQUIRED)
    target_sources(trans PRIVATE
        gdr/gdr_stream.cc
        gdr/gdr_copy.cc
    )
    target_link_libraries(trans PUBLIC
        ${IBVERBS_LIB}
        Threads::Threads
    )
endif()
