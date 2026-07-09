file(GLOB_RECURSE UCM_BUFFER_DELEGATOR_SOURCE_FILES "./cc/*.cc")
add_library(bufferdelegator STATIC ${UCM_BUFFER_DELEGATOR_SOURCE_FILES})
target_include_directories(bufferdelegator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(bufferdelegator PUBLIC infra_status PRIVATE asu_ascend_deps)
