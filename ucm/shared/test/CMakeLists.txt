if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSHARED_TEST_SOURCE_FILES "./case/*.cc")
    add_executable(ucmshared.test ${UCMSHARED_TEST_SOURCE_FILES})
    target_include_directories(ucmshared.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmshared.test PRIVATE
        infra_logger
        trans
        gtest_main gtest
        spdlog
    )
    # Ensure ucmshared.test can find libinfra_logger.so in the build tree
    # Binary:   .../ucm/shared/test/ucmshared.test
    # Library:  .../ucm/shared/infra/logger/libinfra_logger.so
    target_link_options(ucmshared.test PRIVATE "-Wl,-rpath,\$ORIGIN/../infra/logger")
    gtest_discover_tests(ucmshared.test)
endif()
