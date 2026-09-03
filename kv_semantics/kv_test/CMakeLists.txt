add_executable(kv-test
    src/arg_parser.cpp
    src/asu_client_runner.cpp
    src/asu_runtime_proxy.cpp
    src/bench_runner.cpp
    src/buffer_allocator.cpp
    src/consistency_checker.cpp
    src/hcomm_config_adapter.cpp
    src/key_value_generator.cpp
    src/kv_test_app.cpp
    src/kv_test_config_loader.cpp
    src/kv_test_config_helpers.cpp
    src/payload_buffer_runtime.cpp
    src/kv_test_main.cpp
    src/result_writer.cpp
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
)

if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    add_executable(kv-test-config.test
        src/kv_test_config_helpers.cpp
        test/kv_test_config_helpers_test.cpp
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
        src/arg_parser.cpp
        test/arg_parser_test.cpp
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
