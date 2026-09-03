if(NOT (RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "simu"))
    message(FATAL_ERROR
        "BUILD_KV_CLIENT requires RUNTIME_ENVIRONMENT=ascend or simu. Current value: ${RUNTIME_ENVIRONMENT}")
endif()

option(BUILD_KV_CLIENT_PROVIDER_AICPU "Build KV AICPU trans provider" OFF)
option(BUILD_KV_CLIENT_PROVIDER_FAKE "Build KV fake trans provider" ON)
option(BUILD_KV_CLIENT_PROVIDER_AIV "Build KV AIV trans provider (requires external libumc.a)" OFF)

add_subdirectory(src)

if(BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(kv_test)