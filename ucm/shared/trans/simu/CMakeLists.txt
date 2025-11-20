add_library(trans STATIC
    simu_device.cc
    simu_buffer.cc
    simu_stream.cc
)
target_link_libraries(trans PUBLIC
    fmt
)
