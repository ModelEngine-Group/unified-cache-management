# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_executable(ucm_transport_ffts_bench
    ffts_bench_options.cpp
    ffts_bench_runtime.cpp
    ffts_bench_runner.cpp
    ffts_transport_bench.cpp
)

target_link_libraries(ucm_transport_ffts_bench PRIVATE
    ucm_transport_ffts
)

target_include_directories(ucm_transport_ffts_bench SYSTEM PRIVATE
    ${FFTS_ASCEND_INCLUDE_DIR}
)
