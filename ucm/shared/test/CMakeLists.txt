if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES CONFIGURE_DEPENDS "./case/*.cc")
    list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]router_perf\\.cc$")
    if(NOT TARGET router)
        list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]router[/\\\\].*")
    endif()
    if(NOT TARGET buffer_pool)
        list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]pool[/\\\\].*")
    endif()
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans
        metrics
        infra_logger
        gtest_main
        gtest
    )
    if(TARGET buffer_pool)
        target_link_libraries(ucmshared.test PRIVATE buffer_pool)
    endif()
    if(TARGET router)
        target_link_libraries(ucmshared.test PRIVATE router)
    endif()
    gtest_discover_tests(ucmshared.test)

    if(TARGET router)
        add_executable(router_perf case/router/router_perf.cc)
        target_link_libraries(router_perf PRIVATE router)
    endif()
endif()
