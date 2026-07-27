if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE UCMSTORE_TEST_SOURCE_FILES "./case/*.cc")
    if(NOT BUILD_UCM_ASU)
        list(FILTER UCMSTORE_TEST_SOURCE_FILES EXCLUDE REGEX ".*[/\\\\]asu[/\\\\].*")
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
            kv_common)
        if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
            target_sources(ucmstore.test PRIVATE
                ${CMAKE_SOURCE_DIR}/ucm/store/dram/cc/drampool/metadata.cc)
            target_link_libraries(ucmstore.test PRIVATE buffer_pool)
        endif()
    endif()
    if(TARGET drampool_core)
        target_link_libraries(ucmstore.test PRIVATE drampool_core)
    endif()
    if(BUILD_UCM_ASU)
        target_link_libraries(ucmstore.test PRIVATE
            asu_client
            delegator
            buffer_pool
            asu_ascend_deps
        )
        target_compile_definitions(ucmstore.test PRIVATE ASU_BUILD_TESTS)
    endif()
    gtest_discover_tests(ucmstore.test)
endif()
