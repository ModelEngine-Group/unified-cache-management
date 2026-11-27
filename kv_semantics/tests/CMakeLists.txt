include(GoogleTest)

file(GLOB KV_TEST_SOURCE_FILES CONFIGURE_DEPENDS
    *.cc
    client/*.cc
    common/*.cc
    metrics/*.cc
    transport/*.cc
)
add_executable(kv.test ${KV_TEST_SOURCE_FILES})
target_include_directories(kv.test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/ds
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/trans
    ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include
)
target_link_libraries(kv.test PRIVATE
    kv_client
    kv_buffer_runtime
    kv_metrics_standalone
    gtest
)
add_library(m800_fake_agent SHARED fake_agent/m800_fake_agent.cc)
target_include_directories(m800_fake_agent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../agent/include)
add_dependencies(kv.test m800_fake_agent)
target_compile_definitions(kv.test PRIVATE M800_FAKE_AGENT_PATH="$<TARGET_FILE:m800_fake_agent>")

if(BUILD_KV_CLIENT_PROVIDER_ROCE)
    # The tests branch on the build flavour: the stub device layer rejects the
    # verbs backend with UNSUPPORTED, the real one tries to open an adapter.
    target_compile_definitions(kv.test PRIVATE UCM_ASU_ENABLE_ROCE_PROVIDER=1)
endif()

if(BUILD_KV_CLIENT_PROVIDER_HCCP_KV_OVER_ROCE)
    # The provider unit tests dlopen a fake HCCP KV-over-RoCE ABI. The same
    # source builds four variants: full, optional symbols missing, a required
    # symbol missing and Init missing.
    set(KV_FAKE_HCCP_ABI_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/support/fake_hccp_kv_over_roce_abi.cc)
    add_library(hccp_fake_abi_full SHARED ${KV_FAKE_HCCP_ABI_SOURCE})
    add_library(hccp_fake_abi_minimal SHARED ${KV_FAKE_HCCP_ABI_SOURCE})
    add_library(hccp_fake_abi_missing_delete SHARED ${KV_FAKE_HCCP_ABI_SOURCE})
    add_library(hccp_fake_abi_missing_init SHARED ${KV_FAKE_HCCP_ABI_SOURCE})

    foreach(fake_target
        hccp_fake_abi_full
        hccp_fake_abi_minimal
        hccp_fake_abi_missing_delete
        hccp_fake_abi_missing_init
    )
        target_include_directories(${fake_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/trans_provider
        )
    endforeach()

    target_compile_definitions(hccp_fake_abi_minimal PRIVATE HCCP_FAKE_MINIMAL=1)
    target_compile_definitions(hccp_fake_abi_missing_delete
        PRIVATE HCCP_FAKE_MISSING_DELETE=1)
    target_compile_definitions(hccp_fake_abi_missing_init
        PRIVATE HCCP_FAKE_MISSING_INIT=1)

    add_dependencies(kv.test
        hccp_fake_abi_full
        hccp_fake_abi_minimal
        hccp_fake_abi_missing_delete
        hccp_fake_abi_missing_init
    )
    target_compile_definitions(kv.test PRIVATE
        UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER=1
        "UCM_HCCP_FAKE_ABI_FULL_PATH=\"$<TARGET_FILE:hccp_fake_abi_full>\""
        "UCM_HCCP_FAKE_ABI_MINIMAL_PATH=\"$<TARGET_FILE:hccp_fake_abi_minimal>\""
        "UCM_HCCP_FAKE_ABI_MISSING_DELETE_PATH=\"$<TARGET_FILE:hccp_fake_abi_missing_delete>\""
        "UCM_HCCP_FAKE_ABI_MISSING_INIT_PATH=\"$<TARGET_FILE:hccp_fake_abi_missing_init>\""
    )
    target_link_libraries(kv.test PRIVATE ${CMAKE_DL_LIBS})

    # The tests include <acl/acl.h> directly. kv_buffer_runtime already exports
    # the runtime Ascend root, but the provider may have been configured with a
    # different CANN root, so add the provider include directories as well.
    foreach(cann_include_dir
        "${KV_CLIENT_CANN_INCLUDE_DIR}"
        "${KV_CLIENT_CANN_ROOT}/include"
        "${KV_CLIENT_CANN_ROOT}/aarch64-linux/include"
    )
        if(EXISTS "${cann_include_dir}/acl/acl.h")
            target_include_directories(kv.test PRIVATE ${cann_include_dir})
        endif()
    endforeach()
endif()

gtest_discover_tests(kv.test)
