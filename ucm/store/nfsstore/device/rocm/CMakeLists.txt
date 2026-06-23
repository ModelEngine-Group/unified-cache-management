enable_language(HIP)
# enable_language(HIP) auto-detects the host GPU arch (via rocm_agent_enumerator)
# when CMAKE_HIP_ARCHITECTURES is unset; it may list one entry per agent, so
# dedup it. Fall back to gfx90a only when nothing is detected (e.g. a CPU-only
# build host). Pass -DCMAKE_HIP_ARCHITECTURES to override.
if(CMAKE_HIP_ARCHITECTURES)
    list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
else()
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
    message(STATUS "No AMD GPU detected; defaulting CMAKE_HIP_ARCHITECTURES to gfx90a")
endif()
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
