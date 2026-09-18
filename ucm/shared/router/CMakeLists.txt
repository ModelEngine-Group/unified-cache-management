file(GLOB ROUTER_SOURCES CONFIGURE_DEPENDS *.cc)
add_library(router STATIC ${ROUTER_SOURCES})
set_target_properties(router PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(router PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)