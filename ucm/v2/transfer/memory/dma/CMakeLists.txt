add_library(ucm_transfer_memory_dma OBJECT
    dma_engine.h
    dma_engine_registry.h
)
target_include_directories(ucm_transfer_memory_dma PUBLIC ${UCM_TRANSFER_DIR})
target_link_libraries(ucm_transfer_memory_dma PUBLIC ucm_build_options ucm_common)
if(RUNTIME_ENVIRONMENT STREQUAL "cuda" OR RUNTIME_ENVIRONMENT STREQUAL "simu")
    target_sources(ucm_transfer_memory_dma PRIVATE
        cuda_naive_dma_engine.cc
        cuda_gdr_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    target_sources(ucm_transfer_memory_dma PRIVATE
        ascend_naive_dma_engine.cc
        ascend_pin_bounce_dma_engine.cc
        ascend_ffts_bounce_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend-a3")
    target_sources(ucm_transfer_memory_dma PRIVATE
        ascend_naive_dma_engine.cc
        ascend_pin_bounce_dma_engine.cc
        ascend_ffts_bounce_dma_engine.cc
        ascend_ffts_dma_engine.cc
    )
elseif(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    target_sources(ucm_transfer_memory_dma PRIVATE
        ascend_naive_dma_engine.cc
        ascend_pin_bounce_dma_engine.cc
    )
else()
    message(FATAL_ERROR "RUNTIME_ENVIRONMENT must be one of cuda, ascend, ascend-a3, ascend-a5, got: '${RUNTIME_ENVIRONMENT}'")
endif()
