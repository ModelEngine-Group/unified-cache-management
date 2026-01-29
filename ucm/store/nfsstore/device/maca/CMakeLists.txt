set(CUDA_ROOT "/opt/maca/tools/cu-bridge" CACHE PATH "Path to WCUDA root directory")
set(CMAKE_CUDA_COMPILER ${CUDA_ROOT}/bin/cucc)
list(APPEND CMAKE_MODULE_PATH "${CUDA_ROOT}/cmake_module/maca")
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
enable_language(CUDA)

add_library(storedevice STATIC maca_device.cu)

add_library(WCUDA::cudart UNKNOWN IMPORTED)
set_target_properties(WCUDA::cudart PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES "${CUDA_ROOT}/include"
	IMPORTED_LOCATION "${CUDA_ROOT}/lib/libcuda.so"
)
target_include_directories(WCUDA::cudart INTERFACE
	/opt/maca/include
	/opt/maca/include/mcr
)

target_link_libraries(storedevice PUBLIC infra_status infra_logger WCUDA::cudart)
target_compile_options(storedevice PRIVATE -Wall -fPIC -std=c++17)
