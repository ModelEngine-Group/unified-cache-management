if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT BUILD_UCM_ASU)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]asu[/\\\\].*")
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]delegator[/\\\\].*")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore
        gtest_main gtest gmock
    )
    if(BUILD_UCM_ASU)
        target_link_libraries(ucmstore.test PRIVATE asu_client asu_ascend_deps bufferdelegator)
        target_compile_definitions(ucmstore.test PRIVATE ASU_BUILD_TESTS)
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
