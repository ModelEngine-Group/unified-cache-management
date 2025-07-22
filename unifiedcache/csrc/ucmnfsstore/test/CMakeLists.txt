if(BUILD_TESTS)
    file(GLOB_RECURSE UCMNFSSTORE_TEST_SOURCE_FILES "./*.*")
    add_executable(ucmnfsstore.test ${UCMNFSSTORE_TEST_SOURCE_FILES})
    target_link_libraries(ucmnfsstore.test PRIVATE ucmnfsstore.core gtest gtest_main mockcpp)
    target_include_directories(ucmnfsstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
    gtest_discover_tests(ucmnfsstore.test)
endif(BUILD_TESTS)

