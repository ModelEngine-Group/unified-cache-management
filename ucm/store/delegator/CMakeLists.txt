add_library(delegator STATIC
    cc/delegator_buffer_manager.cc
    cc/delegator_copy_stream.cc
)

target_include_directories(delegator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(delegator
    PUBLIC
        infra_status
        buffer_pool
        trans
)
