include(GoogleTest)

file(GLOB KV_TEST_SOURCE_FILES CONFIGURE_DEPENDS
    *.cc
    client/*.cc
    common/*.cc
    metrics/*.cc
    transport/*.cc
)
# The RoCE provider tests pull <infiniband/verbs.h> in through its headers, so
# they can only be built when the provider itself is enabled.
if(NOT BUILD_KV_CLIENT_PROVIDER_ROCE)
    list(FILTER KV_TEST_SOURCE_FILES EXCLUDE REGEX "/roce_(create_connection|memory|one_way_message|shared_state|tcp_channel|trans_provider_framework)")
endif()
add_executable(kv.test ${KV_TEST_SOURCE_FILES})
target_include_directories(kv.test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/ds
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trans
    ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include
)
target_link_libraries(kv.test PRIVATE
    kv_client
    kv_buffer_runtime
    kv_metrics_standalone
    gtest
)
gtest_discover_tests(kv.test)
