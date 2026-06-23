# enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects the
# host GPU(s) via rocm_agent_enumerator and errors if none is found (a no-GPU
# build host must then set the arch explicitly).
enable_language(HIP)
find_package(hip REQUIRED)

set(HIP_COMPAT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../shared/vendor/hip_compat)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_device.cu
    PROPERTIES LANGUAGE HIP)
add_library(storedevice STATIC ${CMAKE_CURRENT_LIST_DIR}/../cuda/cuda_device.cu)
target_include_directories(storedevice PUBLIC ${HIP_COMPAT_DIR} ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(storedevice PUBLIC infra_status infra_logger fmt hip::host)
if(WIN32)
    target_compile_options(storedevice PRIVATE -Wall)
else()
    target_compile_options(storedevice PRIVATE -Wall -fPIC)
endif()
