if(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    set(CACHE_CORE cc)
else()
    set(CACHE_CORE v2)
endif()
file(GLOB_RECURSE UCM_CACHE_STORE_SOURCES "./${CACHE_CORE}/*.cc")
add_library(cachestore SHARED ${UCM_CACHE_STORE_SOURCES})
target_compile_definitions(cachestore PRIVATE
    UCM_RUNTIME_ASCEND_A5=$<STREQUAL:${RUNTIME_ENVIRONMENT},ascend-a5>
)
target_link_libraries(cachestore PUBLIC storeintf trans infra_logger metrics)
set_target_properties(cachestore PROPERTIES
    UCM_CACHE_CORE "${CACHE_CORE}"
    INSTALL_RPATH "$ORIGIN/../../shared/metrics"
)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS cachestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
