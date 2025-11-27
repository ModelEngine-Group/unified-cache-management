if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        trans
        gtest_main gtest mockcpp
    )
    gtest_discover_tests(ucmshared.test)
endif()
