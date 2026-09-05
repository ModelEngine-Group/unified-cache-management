add_executable(kv-test
    src/arg_parser.cc
    src/asu_client_runner.cc
    src/asu_runtime_proxy.cc
    src/bench_runner.cc
    src/buffer_allocator.cc
    src/consistency_checker.cc
    src/hcomm_config_adapter.cc
    src/key_value_generator.cc
    src/kv_test_app.cc
    src/kv_test_config_loader.cc
    src/kv_test_config_helpers.cc
    src/payload_buffer_runtime.cc
    src/kv_test_main.cc
    src/result_writer.cc
)

target_compile_options(kv-test
    PRIVATE
        $<$<CONFIG:Release>:-O3>
)

target_link_options(kv-test
    PRIVATE
        $<$<CONFIG:Release>:-s>
)

target_include_directories(kv-test
    PRIVATE
        include
        ../include
        ../src/trans
        ${UCM_ROOT_DIR}/ucm/shared/infra
)

target_link_libraries(kv-test
    PRIVATE
        ${CMAKE_DL_LIBS}
        trans
        kv_buffer_runtime
)

if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    add_executable(kv-test-config.test
        src/kv_test_config_helpers.cc
        test/kv_test_config_helpers_test.cc
    )
    target_include_directories(kv-test-config.test
        PRIVATE
            include
            ../include
            ../src/trans
    )
    target_link_libraries(kv-test-config.test PRIVATE gtest_main gtest)
    gtest_discover_tests(kv-test-config.test)

    add_executable(kv-test-arg-parser.test
        src/arg_parser.cc
        test/arg_parser_test.cc
    )
    target_include_directories(kv-test-arg-parser.test
        PRIVATE
            include
            ../include
            ../src/trans
    )
    target_link_libraries(kv-test-arg-parser.test PRIVATE gtest_main gtest)
    gtest_discover_tests(kv-test-arg-parser.test)
endif()
