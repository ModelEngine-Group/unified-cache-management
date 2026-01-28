file(GLOB_RECURSE UCM_PIPELINE_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmpipelinestore ${UCM_PIPELINE_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmpipelinestore PRIVATE storeintf infra_logger)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmpipelinestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
