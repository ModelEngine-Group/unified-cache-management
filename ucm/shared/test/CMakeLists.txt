if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES CONFIGURE_DEPENDS "./case/*.cc")
    list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]router_perf\\.cc$")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans
        metrics
        infra_logger
        buffer_pool
        router
        gtest_main
        gtest
    )
    gtest_discover_tests(ucmshared.test)

    add_executable(router_perf case/router/router_perf.cc)
    target_link_libraries(router_perf PRIVATE router)
endif()
