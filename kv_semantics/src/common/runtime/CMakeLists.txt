if(RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_subdirectory(simu)
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    add_subdirectory(ascend)
else()
    message(FATAL_ERROR
        "kv_buffer_runtime requires RUNTIME_ENVIRONMENT=simu or ascend. Current value: ${RUNTIME_ENVIRONMENT}")
endif()

target_include_directories(kv_buffer_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
