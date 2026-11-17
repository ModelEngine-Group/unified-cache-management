add_library(ucm_common INTERFACE)
target_include_directories(ucm_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ucm_common INTERFACE fmt::fmt)
