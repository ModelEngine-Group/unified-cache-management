file(GLOB_RECURSE UCM_CACHE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(cachestore SHARED ${UCM_CACHE_STORE_CC_SOURCE_FILES})
target_include_directories(cachestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger metrics)
# libmetrics.so installs to ucm/shared/metrics/; cachestore.so to
# ucm/store/cache/ — ../../shared/metrics reaches it.
set_target_properties(cachestore PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
    BUILD_WITH_INSTALL_RPATH TRUE
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
