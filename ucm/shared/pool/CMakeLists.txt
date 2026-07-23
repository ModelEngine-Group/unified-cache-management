add_library(buffer_pool STATIC
    buffer_pool.cc
    variable_buffer_pool.cc
)

target_include_directories(buffer_pool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(buffer_pool
    PUBLIC
        infra_status
        infra_thread
    PRIVATE
        offset_allocator
        trans
)
