cmake_minimum_required(VERSION 3.14)
project(retrieval_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(retrieval_backend
  retrieval_backend.cpp
)

set(OUTPUT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/output)
set_target_properties(retrieval_backend PROPERTIES
  PREFIX ""            
  SUFFIX ".so"         
  LIBRARY_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/lib"  
  RUNTIME_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/bin"  
  ARCHIVE_OUTPUT_DIRECTORY  "${OUTPUT_ROOT}/lib" 
)

target_compile_options(retrieval_backend PRIVATE -O3 -Wall -fPIC)
target_link_libraries(retrieval_backend PRIVATE Python3::Python)
