if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES "./case/*.cc")
    if(WIN32)
        list(FILTER UCMSHARED_TEST_SOURCE_FILES EXCLUDE REGEX "thread_pool_test\\.cc$")
    endif()
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans metrics infra_logger
        gtest_main gtest
    )
    gtest_discover_tests(ucmshared.test)
endif()
