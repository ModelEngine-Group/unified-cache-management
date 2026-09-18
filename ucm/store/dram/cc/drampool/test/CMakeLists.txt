# Unit tests for the DramPool metrics helpers (drampool_metrics.h). The module
# is header-only and has no device dependency, so unlike the drampool suites
# under ucm/store/test/case/dram/drampool these cases need no transport runtime.
add_executable(drampool_metrics.test drampool_metrics_test.cc)
target_link_libraries(drampool_metrics.test PRIVATE
    drampool_core
    gtest_main
    gtest
)
include(GoogleTest)
gtest_discover_tests(drampool_metrics.test)
