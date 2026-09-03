include(GoogleTest)

file(GLOB KV_TEST_SOURCE_FILES CONFIGURE_DEPENDS
    client/*.cpp
    transport/*.cpp
)
add_executable(kv.test ${KV_TEST_SOURCE_FILES})
target_compile_definitions(kv.test PRIVATE ASU_BUILD_TESTS)
target_include_directories(kv.test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trans
    ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_link_libraries(kv.test PRIVATE
    kv_client
    gtest
)
gtest_discover_tests(kv.test)