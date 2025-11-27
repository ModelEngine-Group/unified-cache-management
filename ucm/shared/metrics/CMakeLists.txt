cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto")

file(GLOB CORE_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/stats/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/*.cc")
add_library(monitor_static STATIC ${CORE_SRCS})
set_property(TARGET monitor_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(monitor_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cc>
    $<INSTALL_INTERFACE:include>)
set_target_properties(monitor_static PROPERTIES OUTPUT_NAME monitor)

file(GLOB BINDINGS_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpy/*.cc")
pybind11_add_module(monitor ${BINDINGS_SRCS})
target_link_libraries(monitor PRIVATE -Wl,--whole-archive monitor_static -Wl,--no-whole-archive)
target_include_directories(monitor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cc)
set_target_properties(monitor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})