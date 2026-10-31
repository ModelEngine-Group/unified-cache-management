# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
#
# Test build for the gsa_prefetch C++ surface.
#
# Built with KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC so:
#   * ascend/kvcache_trans_ascend.h pulls in kvcache_trans_ascend_stub.h (no
#     Ascend SDK required);
#   * the private members of GSAPrefetchEngineC are exposed as public so the
#     cross-module (MST) and end-to-end (DT) tests can drive internal flows.
#
# All ACL H2D/D2D ops and the UC::StoreV1 (mStoreV1_) surface are stubbed /
# mocked, matching the task requirement.

set(PREFETCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(PREFETCH_SOURCES
    "${PREFETCH_SRC_DIR}/src/kvcache_pre.cpp"
    "${PREFETCH_SRC_DIR}/src/kvcache_pre_scatter.cpp"
    "${PREFETCH_SRC_DIR}/src/kvcache_task_manger.cpp"
    "${PREFETCH_SRC_DIR}/ascend/kvcache_trans_ascend.cpp"
)
# pybinds.cpp is intentionally excluded: PYBIND11_MODULE must live in the
# extension module, not a gtest binary. Its helper functions are covered
# transitively by the Python DT layer that drives the bound methods.

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Locate PyTorch for at::Tensor / torch::Tensor used by TaskManager and the
# engine surface. Reuse the same discovery the main build uses.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)))"
    OUTPUT_VARIABLE PYTORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTORCH_RESULT
)
if(NOT PYTORCH_RESULT EQUAL 0)
    message(FATAL_ERROR "PyTorch not found; required by the prefetch test binary")
endif()

set(UCM_STORE_DIR "${UCM_ROOT_DIR}/ucm/store")
set(UCM_SHARED_DIR "${UCM_ROOT_DIR}/ucm/shared/infra")

# Common compile flags for the test target.
# NOTE: -DDEBUG must NOT be defined here: kvcache_log.h declares an enumerator
# named `DEBUG` inside `enum class LogLevel`, so a global DEBUG macro breaks
# compilation. HBM_ASSERT is a no-op without DEBUG, which is the production
# behaviour we want exercised in tests anyway.
set(TEST_COMPILE_DEFS
    KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC
)

set(TEST_INCLUDE_DIRS
    "${PREFETCH_SRC_DIR}/include"
    "${PREFETCH_SRC_DIR}/ascend"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
    "${UCM_STORE_DIR}"
    "${UCM_STORE_DIR}/detail"
    "${UCM_STORE_DIR}/detail/type"
    "${UCM_SHARED_DIR}"
    "${PYTORCH_PATH}/include"
    "${PYTORCH_PATH}/include/torch/csrc/api/include"
)

# pybind11 (pulled in transitively by kvcache_pre.h) requires Python.h.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_INCLUDE_RESULT
)
if(PYTHON_INCLUDE_RESULT EQUAL 0)
    list(APPEND TEST_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
endif()

# Discover every UT / MST / DT source so new cases are picked up automatically.
file(GLOB_RECURSE PREFETCH_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ut/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mst/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dt/*.cpp"
)

add_executable(gsa_prefetch_tests
    ${PREFETCH_SOURCES}
    ${PREFETCH_TEST_SOURCES}
)

target_include_directories(gsa_prefetch_tests PRIVATE ${TEST_INCLUDE_DIRS})
target_compile_definitions(gsa_prefetch_tests PRIVATE ${TEST_COMPILE_DEFS})
target_compile_options(gsa_prefetch_tests PRIVATE -Wno-unused-parameter -Wno-unused-variable)

# gtest / gmock. Prefer the system package; fall back to FetchContent.
find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    target_link_libraries(gsa_prefetch_tests PRIVATE gtest gtest_main)
else()
    target_link_libraries(gsa_prefetch_tests PRIVATE GTest::GTest GTest::Main)
endif()

# fmt is required by ucm/shared/infra/status/status.h.
get_target_property(FMT_INCLUDE_DIRS fmt INTERFACE_INCLUDE_DIRECTORIES)
if(FMT_INCLUDE_DIRS)
    target_include_directories(gsa_prefetch_tests PRIVATE ${FMT_INCLUDE_DIRS})
endif()

target_link_directories(gsa_prefetch_tests PRIVATE
    "${PYTORCH_PATH}/lib"
    "/usr/local/lib"
    "${UCM_ROOT_DIR}/build/ucm/store"
)

target_link_libraries(gsa_prefetch_tests PRIVATE
    torch
    c10
    torch_cpu
    storeintf
    pthread
    ${CMAKE_DL_LIBS}
    rt
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gsa_prefetch_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

# securec is referenced by the prefetch sources; link it if available.
find_library(SECUREC_LIB securec PATHS "/usr/local/Ascend/driver/lib64/common" "/usr/local/lib")
if(SECUREC_LIB)
    target_link_libraries(gsa_prefetch_tests PRIVATE ${SECUREC_LIB})
else()
    # Provide an empty securec fallback so the build does not fail in CI without
    # the Huawei Secure C library; the prefetch code only uses memset_s/strcpy_s
    # which we substitute with the standard functions.
    target_compile_definitions(gsa_prefetch_tests PRIVATE UCM_PREFETCH_TEST_NO_SECUREC)
endif()

include(GoogleTest)
gtest_discover_tests(gsa_prefetch_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "prefetch;ut;mst;dt"
)

# Coverage target (line + branch coverage >= 90% requirement).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(gsa_prefetch_coverage
            COMMAND ${LCOV_PATH} --directory ${CMAKE_CURRENT_BINARY_DIR} --zerocounters
            COMMAND $<TARGET_FILE:gsa_prefetch_tests>
            COMMAND ${LCOV_PATH} --directory ${CMAKE_CURRENT_BINARY_DIR}
                    --capture --output-file ${CMAKE_CURRENT_BINARY_DIR}/coverage.info
                    --rc branch_coverage=1
            COMMAND ${LCOV_PATH} --remove ${CMAKE_CURRENT_BINARY_DIR}/coverage.info
                    '/usr/*' '${PYTORCH_PATH}/*' '*/gtest/*' '*/test/*'
                    --output-file ${CMAKE_CURRENT_BINARY_DIR}/coverage.filtered.info
                    --rc branch_coverage=1
            COMMAND ${GENHTML_PATH} ${CMAKE_CURRENT_BINARY_DIR}/coverage.filtered.info
                    --branch-coverage --output-directory ${CMAKE_CURRENT_BINARY_DIR}/coverage_html
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating line+branch coverage for gsa_prefetch_tests"
        )
    endif()
endif()
