file(GLOB KV_COMMON_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(kv_common STATIC ${KV_COMMON_SOURCES})
target_include_directories(kv_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE KV_COMMON_TEST_SOURCE_FILES CONFIGURE_DEPENDS test/*.cpp)
    add_executable(kv_common.test ${KV_COMMON_TEST_SOURCE_FILES})
    target_link_libraries(kv_common.test PRIVATE kv_common gtest_main gtest)
    gtest_discover_tests(kv_common.test)

    add_executable(kv_router_perf perf/router_perf.cpp)
    target_link_libraries(kv_router_perf PRIVATE kv_common)
endif()
