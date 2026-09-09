if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT TARGET dramstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\].*")
    endif()
    if(NOT TARGET yuanrongstore)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*/yuanrong/.*\\.cc$")
    endif()
    if(NOT TARGET delegator)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]delegator[/\\\\].*")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        set_source_files_properties(case/dram/kv_protocol_test.cc
            PROPERTIES COMPILE_OPTIONS "-Wno-nonnull")
    endif()
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore healthbreakerstore
        gtest_main gtest gmock
    )
    if(TARGET dramstore AND TARGET drampool_core)
        target_link_libraries(ucmstore.test PRIVATE
            dramstore
            drampool_core
        )
    endif()
    if(TARGET yuanrongstore)
        target_compile_definitions(ucmstore.test PRIVATE UCM_YUANRONG_TESTING)
        target_link_libraries(ucmstore.test PRIVATE yuanrongstore)
    endif()
    if(TARGET delegator)
        target_link_libraries(ucmstore.test PRIVATE delegator delegator_store)
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
