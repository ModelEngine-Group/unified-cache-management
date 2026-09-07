if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT BUILD_UCM_ASU)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]asu[/\\\\].*")
    endif()
    if(NOT BUILD_UCM_DELEGATOR)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]delegator[/\\\\].*")
    endif()
    if(RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "simu")
        if(NOT TARGET drampool_core)
            list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\]drampool[/\\\\].*")
        endif()
    else()
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]dram[/\\\\].*")
    endif()
    add_executable(ucmstore.test ${UCMSTORE_TEST_SOURCE_FILES})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        set_source_files_properties(case/dram/kv_protocol_test.cc
            PROPERTIES COMPILE_OPTIONS "-Wno-nonnull")
    endif()
    target_include_directories(ucmstore.test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/case)
    target_link_libraries(ucmstore.test PRIVATE
        posixstore cachestore fakestore pcstore
        gtest_main gtest gmock
    )
    if(RUNTIME_ENVIRONMENT STREQUAL "ascend" OR RUNTIME_ENVIRONMENT STREQUAL "simu")
        target_include_directories(ucmstore.test PRIVATE
            ${CMAKE_SOURCE_DIR}/ucm/store/dram/cc
            ${CMAKE_SOURCE_DIR}/ucm/store/dram/cc/drampool
            ${CMAKE_SOURCE_DIR}/ucm/transport/p2p/include
            ${CMAKE_SOURCE_DIR}/ucm/transport/p2p/include/one_sided
            ${CMAKE_SOURCE_DIR}/ucm/shared)
        target_link_libraries(ucmstore.test PRIVATE
            dramstore
            storeintf
            router)
        if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
            target_sources(ucmstore.test PRIVATE
                ${CMAKE_SOURCE_DIR}/ucm/store/dram/cc/drampool/metadata.cc)
            target_link_libraries(ucmstore.test PRIVATE buffer_pool)
        endif()
    endif()
    if(TARGET drampool_core)
        target_link_libraries(ucmstore.test PRIVATE drampool_core)
    endif()
    if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
        target_compile_definitions(ucmstore.test PRIVATE UCM_TEST_RUNTIME_ASCEND=1)
    endif()
    if(BUILD_UCM_ASU)
        target_link_libraries(ucmstore.test PRIVATE
            kv_client
            buffer_pool
            trans
        )
        target_compile_definitions(ucmstore.test PRIVATE ASU_BUILD_TESTS)
    endif()
    if(BUILD_UCM_DELEGATOR)
        target_link_libraries(ucmstore.test PRIVATE
            delegator
            delegator_store
        )
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
