cmake_minimum_required(VERSION 3.18)
project(dev-sandbox VERSION 1.0.0 LANGUAGES CXX)

# Runtime backend detection
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(DetectRuntime)
detect_runtime_backend()
if (CUDA_FOUND)
    enable_language(CUDA)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(FLAGS_PUBLIC "-Wall -Werror -fPIC -Wl,-z,relro,-z,now")
set(FLAGS_DEBUG "-O0 -g")
set(FLAGS_RELEASE "-O3 -D_FORTIFY_SOURCE=2")
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if(CMAKE_BUILD_TYPE_LOWER STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_DEBUG}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_PUBLIC} ${FLAGS_RELEASE}")
endif()

add_subdirectory(vendor)
add_subdirectory(module)
