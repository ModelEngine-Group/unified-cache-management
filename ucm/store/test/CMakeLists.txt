if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    # DramPool sources currently depend on shared DramStore types that are not part of this PR.
    list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\].*")
    if(NOT TARGET yuanrongstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*/yuanrong/.*\\.cc$")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore healthbreakerstore
        gtest_main gtest gmock
    )
    if(TARGET yuanrongstore)
        target_compile_definitions(ucmstore.test PRIVATE UCM_YUANRONG_TESTING)
        target_link_libraries(ucmstore.test PRIVATE yuanrongstore)
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
