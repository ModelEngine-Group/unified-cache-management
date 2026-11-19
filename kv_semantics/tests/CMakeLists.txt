include(GoogleTest)

file(GLOB KV_TEST_SOURCE_FILES CONFIGURE_DEPENDS
    *.cc
    client/*.cc
    common/*.cc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include
)
target_link_libraries(kv.test PRIVATE
    kv_client
    kv_buffer_runtime
    kv_metrics_standalone
    gtest
)
add_library(m800_fake_agent SHARED fake_agent/m800_fake_agent.cc)
target_include_directories(m800_fake_agent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include)
add_dependencies(kv.test m800_fake_agent)
target_compile_definitions(kv.test PRIVATE M800_FAKE_AGENT_PATH="$<TARGET_FILE:m800_fake_agent>")
gtest_discover_tests(kv.test)
