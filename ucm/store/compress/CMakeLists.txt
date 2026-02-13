file(GLOB_RECURSE UCM_COMPRESSOR_STORE_CC_SOURCE_FILES "./cc/*.cc")

add_library(compressor SHARED ${UCM_COMPRESSOR_STORE_CC_SOURCE_FILES})
target_include_directories(compressor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/compress_lib
)
target_link_libraries(compressor PUBLIC storeintf infra_logger)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS compressor LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
