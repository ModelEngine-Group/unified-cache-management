set(UCM_DRAM_STORE_CC_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/dram_store.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/kv_protocol.cc
)
add_library(dramstore SHARED ${UCM_DRAM_STORE_CC_SOURCE_FILES})
target_include_directories(dramstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_include_directories(dramstore PUBLIC
    ${CMAKE_SOURCE_DIR}/ucm/transport/p2p/include/one_sided)
target_link_libraries(dramstore PUBLIC storeintf infra_logger infra_thread buffer_pool)
file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS dramstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

if(BUILD_UCM_DRAMPOOL)
    set(DRAMPOOL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cc/drampool)
    add_library(drampool_core STATIC
        ${DRAMPOOL_SOURCE_DIR}/completion_poller.cc
        ${DRAMPOOL_SOURCE_DIR}/drampool_daemon.cc
        ${DRAMPOOL_SOURCE_DIR}/drampool_launch_config.cc
        ${DRAMPOOL_SOURCE_DIR}/drampool_server.cc
        ${DRAMPOOL_SOURCE_DIR}/drampool_yaml_config.cc
        ${DRAMPOOL_SOURCE_DIR}/metadata.cc
        ${DRAMPOOL_SOURCE_DIR}/task_worker.cc
    )
    target_include_directories(drampool_core
        PUBLIC
            ${DRAMPOOL_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cc
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
endif()
