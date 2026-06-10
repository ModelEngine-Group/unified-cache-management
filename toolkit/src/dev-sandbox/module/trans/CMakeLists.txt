add_executable(trans trans_main.cc)
target_include_directories(trans PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trans PRIVATE fmt::fmt)
if (CUDA_FOUND)
    target_sources(trans PRIVATE
        cuda/trans_runtime_cuda.cc
        cuda/trans_case_cuda.cc
        cuda/trans_kernel_cuda.cu
    )
    target_link_libraries(trans PRIVATE CUDA::Runtime)
elseif(ASCEND_FOUND)
    target_sources(trans PRIVATE
        ascend/trans_runtime_ascend.cc
        ascend/trans_case_ascend.cc
    )
    target_link_libraries(trans PRIVATE Ascend::Runtime)
else()
    target_sources(trans PRIVATE
        simu/trans_runtime_simu.cc
        simu/trans_case_simu.cc
    )
endif()
