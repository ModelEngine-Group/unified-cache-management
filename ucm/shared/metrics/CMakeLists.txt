file(GLOB_RECURSE CORE_SRCS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/stats/*.cc"
     "${CMAKE_CURRENT_SOURCE_DIR}/cc/*.cc")
add_library(monitor_static STATIC ${CORE_SRCS})
set_property(TARGET monitor_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(monitor_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cc>
    $<INSTALL_INTERFACE:include>)
set_target_properties(monitor_static PROPERTIES OUTPUT_NAME monitor)

file(GLOB_RECURSE BINDINGS_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpy/*.cc")
pybind11_add_module(ucmmonitor ${BINDINGS_SRCS})
target_link_libraries(ucmmonitor PRIVATE -Wl,--whole-archive monitor_static -Wl,--no-whole-archive)
target_include_directories(ucmmonitor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cc)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS ucmmonitor LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)