add_library(delegator STATIC
    cc/delegator_copy_stream.cc
    cc/delegator_executor.cc
)

target_include_directories(delegator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(delegator
    PUBLIC
        storeintf
        infra_status
        buffer_pool
        trans
    PRIVATE
        infra_logger
)

add_library(delegator_store SHARED
    cc/delegator_store.cc
)
target_link_libraries(delegator_store
    PUBLIC
        storeintf
        delegator
    PRIVATE
        infra_logger
        asustore
)
set_target_properties(delegator_store PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN;$ORIGIN/../asu;$ORIGIN/../../transport/kv/asu"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../asu;$ORIGIN/../../transport/kv/asu"
)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS delegator_store LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
