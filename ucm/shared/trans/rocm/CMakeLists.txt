# enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects the
# host GPU(s) via rocm_agent_enumerator and errors if none is found (a no-GPU
# build host must then set the arch explicitly). Consumer Radeon users (gfx1100,
# gfx1201, ...) building on a different machine than they deploy to should pass
# -DCMAKE_HIP_ARCHITECTURES=<arch> so the binaries match the target GPU.
enable_language(HIP)
find_package(hip REQUIRED)
message(STATUS "UCM trans: building HIP for arch(s): ${CMAKE_HIP_ARCHITECTURES}")

set(HIP_COMPAT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../vendor/hip_compat)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_sm_kernel.cu
    PROPERTIES LANGUAGE HIP)
add_library(kernel OBJECT ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_sm_kernel.cu)
target_include_directories(kernel PRIVATE ${HIP_COMPAT_DIR} ${CMAKE_CURRENT_LIST_DIR}/../cuda)
if(WIN32)
    target_compile_options(kernel PRIVATE -Wall)
else()
    target_compile_options(kernel PRIVATE -Wall -fPIC)
endif()

add_library(trans STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_device.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_buffer.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_stream.cc
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_sm_stream.cc
)
target_include_directories(trans PUBLIC ${HIP_COMPAT_DIR} ${CMAKE_CURRENT_LIST_DIR}/../cuda)
target_link_libraries(trans PUBLIC
    fmt
    hip::host
    kernel
)
