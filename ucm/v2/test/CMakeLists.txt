file(GLOB_RECURSE UCM_TEST_CC_SOURCES CONFIGURE_DEPENDS "*.cc")
list(FILTER UCM_TEST_CC_SOURCES EXCLUDE REGEX "fixtures/stub_store\\.cc$")
add_executable(ucm_test ${UCM_TEST_CC_SOURCES})
target_link_libraries(ucm_test
    PRIVATE ucm_build_options ucm_common ucm_store ucm_store_cache_core ucm_proxy_core
        GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(ucm_test)

# Test-owned store plugin; assembler tests load it by path, through the
# out-of-tree plugin mechanism of StoreLoader, so it never enters the
# production builtin table.
add_library(ucm_test_stub_store MODULE proxy/fixtures/stub_store.cc)
target_link_libraries(ucm_test_stub_store PRIVATE ucm_build_options ucm_common ucm_store)
target_compile_definitions(ucm_test PRIVATE
    UCM_TEST_STUB_STORE_PATH="$<TARGET_FILE:ucm_test_stub_store>")
