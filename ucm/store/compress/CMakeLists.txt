file(GLOB_RECURSE UCM_COMPRESSOR_STORE_CC_SOURCE_FILES "./cc/*.cc")
file(GLOB_RECURSE UCM_COMPRESSOR_STORE_CPY_SOURCE_FILES "./cpy/*.cc")

add_library(compressor STATIC ${UCM_COMPRESSOR_STORE_CC_SOURCE_FILES})
target_include_directories(compressor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/compress_lib
)
target_link_libraries(compressor PUBLIC storeintf infra_logger)

file(GLOB_RECURSE UCM_COMPRESSOR_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmcompressor ${UCM_COMPRESSOR_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmcompressor PRIVATE compressor)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmcompressor LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
