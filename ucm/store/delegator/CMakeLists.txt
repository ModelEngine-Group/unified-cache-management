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
