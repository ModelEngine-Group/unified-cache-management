include(GoogleTest)

file(GLOB KV_TEST_SOURCE_FILES CONFIGURE_DEPENDS
    *.cc
    client/*.cc
    metrics/*.cc
    transport/*.cc
)
add_executable(kv.test ${KV_TEST_SOURCE_FILES})
target_include_directories(kv.test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/ds
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trans
)
target_link_libraries(kv.test PRIVATE
    kv_client
    kv_buffer_runtime
    kv_metrics_standalone
    gtest
)
gtest_discover_tests(kv.test)
