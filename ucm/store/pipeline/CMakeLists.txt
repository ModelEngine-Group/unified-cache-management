file(GLOB_RECURSE UCM_BREAKER_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(breakerstore STATIC ${UCM_BREAKER_STORE_CC_SOURCE_FILES})
set_target_properties(breakerstore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(breakerstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(breakerstore PUBLIC storeintf infra_logger)

file(GLOB_RECURSE UCM_PIPELINE_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmpipelinestore ${UCM_PIPELINE_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmpipelinestore PRIVATE storeintf infra_logger breakerstore)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmpipelinestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
