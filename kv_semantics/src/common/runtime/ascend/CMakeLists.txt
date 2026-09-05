set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Path to Ascend root directory")
add_library(Ascend::ascendcl UNKNOWN IMPORTED)
set_target_properties(Ascend::ascendcl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_ROOT}/include"
    IMPORTED_LOCATION "${ASCEND_ROOT}/lib64/libascendcl.so"
)

add_library(kv_buffer_runtime STATIC
    ascend_buffer.cc
    ascend_device.cc
    ascend_trans.cc
)
if(ASCEND_SUPPORTS_REGISTER_PIN)
    target_compile_definitions(kv_buffer_runtime PRIVATE ASCEND_SUPPORTS_REGISTER_PIN=1)
endif()
target_link_libraries(kv_buffer_runtime
    PUBLIC
        fmt
        Ascend::ascendcl
)
target_include_directories(kv_buffer_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${UCM_ROOT_DIR}/kv_semantics/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../logger
)
target_compile_definitions(kv_buffer_runtime PRIVATE SPDLOG_FMT_EXTERNAL)
