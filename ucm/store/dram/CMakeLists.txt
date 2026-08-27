if(NOT (RUNTIME_ENVIRONMENT STREQUAL "ascend"
        OR RUNTIME_ENVIRONMENT STREQUAL "simu"))
    return()
endif()

set(DRAM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cc)

set(UCM_DRAM_STORE_CC_SOURCE_FILES
    ${DRAM_SOURCE_DIR}/dram_store.cc
    ${DRAM_SOURCE_DIR}/kv_protocol.cc
    ${DRAM_SOURCE_DIR}/config.cc
    ${DRAM_SOURCE_DIR}/dram_fatal.cc
    ${DRAM_SOURCE_DIR}/transport_executor.cc
    ${DRAM_SOURCE_DIR}/reply_service.cc
    ${DRAM_SOURCE_DIR}/node_actor.cc
    ${DRAM_SOURCE_DIR}/node_scheduler.cc
    ${DRAM_SOURCE_DIR}/task_manager.cc
)
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    list(APPEND UCM_DRAM_STORE_CC_SOURCE_FILES
        ${DRAM_SOURCE_DIR}/transport_manager_backend.cc)
endif()
add_library(dramstore SHARED ${UCM_DRAM_STORE_CC_SOURCE_FILES})

set_target_properties(dramstore PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../transport/p2p"
)
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    target_compile_definitions(dramstore PRIVATE UC_DRAM_ASCEND_BACKEND)
endif()
target_include_directories(dramstore PUBLIC
    ${DRAM_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ucm/transport/p2p/include
    ${CMAKE_SOURCE_DIR}/ucm/shared
)

target_link_libraries(dramstore
    PRIVATE
        trans
    PUBLIC
        storeintf
        infra_logger
        infra_thread
        router
        buffer_pool
)
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    target_link_libraries(dramstore PUBLIC ucm_p2p_transport)
endif()
file(RELATIVE_PATH DRAM_INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS dramstore
    LIBRARY DESTINATION ${DRAM_INSTALL_REL_PATH}
    COMPONENT ucm
)

set(DRAMPOOL_SOURCE_DIR ${DRAM_SOURCE_DIR}/drampool)
set(DRAMPOOL_SOURCE_FILES
    ${DRAMPOOL_SOURCE_DIR}/completion_poller.cc
    ${DRAMPOOL_SOURCE_DIR}/drampool_daemon.cc
    ${DRAMPOOL_SOURCE_DIR}/drampool_launch_config.cc
    ${DRAMPOOL_SOURCE_DIR}/drampool_server.cc
    ${DRAMPOOL_SOURCE_DIR}/drampool_yaml_config.cc
    ${DRAMPOOL_SOURCE_DIR}/health_server.cc
    ${DRAMPOOL_SOURCE_DIR}/metadata.cc
    ${DRAMPOOL_SOURCE_DIR}/task_worker.cc
)
add_library(drampool_core STATIC ${DRAMPOOL_SOURCE_FILES})
target_include_directories(drampool_core
    PUBLIC
        ${DRAMPOOL_SOURCE_DIR}
        ${DRAM_SOURCE_DIR}
)
target_link_libraries(drampool_core
    PUBLIC
        infra_status
        infra_logger
        infra_template
        infra_thread
        buffer_pool
        ucm_p2p_transport
        trans
        dramstore
)
add_executable(drampool ${DRAMPOOL_SOURCE_DIR}/main.cc)
target_link_libraries(drampool PRIVATE drampool_core)
install(TARGETS drampool RUNTIME DESTINATION bin COMPONENT ucm)
