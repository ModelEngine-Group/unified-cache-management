# Copyright (c) 2026 Huawei Technologies Co., Ltd.

include(GoogleTest)

add_executable(ucm_transport_ffts.test
    ffts_transport_test.cpp
)

target_link_libraries(ucm_transport_ffts.test PRIVATE
    ucm_transport_ffts
    gtest_main
    gtest
)

target_include_directories(ucm_transport_ffts.test SYSTEM PRIVATE
    ${FFTS_ASCEND_INCLUDE_DIR}
)

gtest_discover_tests(ucm_transport_ffts.test)
