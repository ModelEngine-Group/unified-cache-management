find_path(ASCEND_ACL_INCLUDE_DIR NAMES acl/acl.h PATHS /usr/local/Ascend/ascend-toolkit/latest/include NO_DEFAULT_PATH)
find_library(ASCEND_ACL_LIBRARY NAMES ascendcl PATHS /usr/local/Ascend/ascend-toolkit/latest/lib64 NO_DEFAULT_PATH)
find_library(MOONCAKE_STORE_LIBRARY NAMES mooncake_store PATHS /usr/local/lib NO_DEFAULT_PATH)

if(ASCEND_ACL_INCLUDE_DIR AND ASCEND_ACL_LIBRARY AND MOONCAKE_STORE_LIBRARY)
    file(GLOB_RECURSE SOURCES "./cc/*.cc")
    add_library(mooncakestore SHARED ${SOURCES})

    target_compile_features(mooncakestore PUBLIC cxx_std_20)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(mooncakestore PRIVATE -fcoroutines)
    endif()

    target_compile_definitions(mooncakestore PRIVATE YLT_ENABLE_IBV SPDLOG_FMT_EXTERNAL)

    if(NOT MOONCAKE_STORE_INCLUDE_DIR)
        set(MOONCAKE_STORE_INCLUDE_DIR "/vllm-workspace/Mooncake/mooncake-store/include")
    endif()

    target_include_directories(mooncakestore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc
        ${MOONCAKE_STORE_INCLUDE_DIR}
        ${MOONCAKE_STORE_INCLUDE_DIR}/cachelib_memory_allocator
        ${MOONCAKE_STORE_INCLUDE_DIR}/cachelib_memory_allocator/fake_include
    )
    target_include_directories(mooncakestore SYSTEM PUBLIC
        ${MOONCAKE_STORE_INCLUDE_DIR}/cachelib_memory_allocator/include
        ${ASCEND_ACL_INCLUDE_DIR}
    )

    target_link_libraries(mooncakestore PUBLIC
        storeintf infra_logger fmt trans metrics ${MOONCAKE_STORE_LIBRARY}
        ${ASCEND_ACL_LIBRARY}
    )

    file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    install(TARGETS mooncakestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
else()
    message(STATUS "mooncakestore: Skipping build - required Ascend/Mooncake dependencies not found")
    if(NOT ASCEND_ACL_INCLUDE_DIR)
        message(STATUS "  * Missing: Ascend ACL headers (acl/acl.h)")
    endif()
    if(NOT ASCEND_ACL_LIBRARY)
        message(STATUS "  * Missing: libascendcl.so library")
    endif()
    if(NOT MOONCAKE_STORE_LIBRARY)
        message(STATUS "  * Missing: libmooncake_store.so library")
    endif()
    message(STATUS "  Please ensure Ascend and Mooncake dependencies are installed or build paths are correct")
endif()
