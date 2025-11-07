# 添加编译目标
pybind11_add_module(hash_retrieval_backend cpy/hash_retrieval_backend.cpp)

# 设置输出库的目录
set_target_properties(hash_retrieval_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# 设置头文件目录，以确保 numaf.h 能找到
target_include_directories(hash_retrieval_backend PUBLIC
    ${NUMA_INSTALL_DIR}/include
    ${Torch_INCLUDE_DIRS}
)

# 链接所需的库
target_link_libraries(hash_retrieval_backend PUBLIC
    ${NUMA_INSTALL_DIR}/lib/libnuma.so
    ${Torch_LIBRARIES}
)