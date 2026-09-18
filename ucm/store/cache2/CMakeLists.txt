file(GLOB_RECURSE UCM_CACHE2_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(cachestore2 SHARED ${UCM_CACHE2_STORE_CC_SOURCE_FILES})
target_include_directories(cachestore2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(cachestore2 PUBLIC storeintf trans infra_logger metrics)
set_target_properties(cachestore2 PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore2 LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
