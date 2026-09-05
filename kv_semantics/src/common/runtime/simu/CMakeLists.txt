add_library(kv_buffer_runtime STATIC
    simu_buffer.cc
    simu_device.cc
    simu_trans.cc
)
target_link_libraries(kv_buffer_runtime
    PUBLIC
        fmt
)
target_include_directories(kv_buffer_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${UCM_ROOT_DIR}/kv_semantics/include
)
