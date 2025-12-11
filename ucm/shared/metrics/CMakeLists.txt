file(GLOB_RECURSE DOMAIN_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/cc/domain/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/cc/domain/stats/*.cc"
)

file(GLOB_RECURSE API_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/cc/api/*.cc"
)

add_library(monitor_static STATIC
  ${DOMAIN_SRCS}
  ${API_SRCS}
)

set_property(TARGET monitor_static PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(monitor_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cc/domain>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cc/api>
  $<INSTALL_INTERFACE:include>
)

set_target_properties(monitor_static PROPERTIES OUTPUT_NAME monitor)

file(GLOB_RECURSE BINDINGS_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpy/*.cc")
pybind11_add_module(ucmmonitor ${BINDINGS_SRCS})
target_link_libraries(ucmmonitor PRIVATE monitor_static)

target_include_directories(ucmmonitor PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cc/api"
)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS ucmmonitor LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)