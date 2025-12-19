file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(cachestore STATIC ${UCM_CACHE_STORE_CC_SOURCE_FILES})
target_include_directories(cachestore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
)
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger)

file(GLOB_RECURSE UCM_CACHE_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmcachestore ${UCM_CACHE_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmcachestore PRIVATE cachestore)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmcachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
