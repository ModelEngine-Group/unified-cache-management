add_library(offset_allocator STATIC
    offset_allocator.cc
)

target_include_directories(offset_allocator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
