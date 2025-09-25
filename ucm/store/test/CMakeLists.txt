if(BUILD_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        dramstore nfsstore localstore storeinfra storedevice
        gtest_main gtest mockcpp
    )
    gtest_discover_tests(ucmstore.test)
endif()
