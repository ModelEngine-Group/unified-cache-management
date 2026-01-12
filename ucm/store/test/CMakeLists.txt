if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore
        gtest_main gtest gmock
    )
    # Add RPATH so the test executable can find libinfra_logger.so
    # $ORIGIN/../../shared/infra/logger points to where libinfra_logger.so is built
    target_link_options(ucmstore.test PRIVATE "-Wl,-rpath,\$ORIGIN/../../shared/infra/logger")
    gtest_discover_tests(ucmstore.test)
endif()
