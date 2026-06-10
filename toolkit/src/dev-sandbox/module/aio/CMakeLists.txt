add_executable(aio aio_main.cc)
target_sources(aio PRIVATE aio_impl.cc)
target_link_libraries(aio PRIVATE fmt::fmt)

# Select implementation based on runtime backend
if(RUNTIME_BACKEND STREQUAL "CUDA")
    target_sources(aio PRIVATE host_buffer_cuda.cc)
    target_link_libraries(aio PRIVATE CUDA::Runtime)
elseif(RUNTIME_BACKEND STREQUAL "Ascend")
    target_sources(aio PRIVATE host_buffer_ascend.cc)
    target_link_libraries(aio PRIVATE Ascend::Runtime)
else()
    target_sources(aio PRIVATE host_buffer.cc)
endif()
