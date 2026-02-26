# 添加编译目标
pybind11_add_module(hash_retrieval_backend cpy/hash_retrieval_backend.cpp)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS hash_retrieval_backend LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

# 设置头文件目录，以确保 numaf.h 能找到
target_include_directories(hash_retrieval_backend PUBLIC
    ${NUMA_INSTALL_DIR}/include
    ${Torch_INCLUDE_DIRS}
)

# 链接所需的库
target_link_libraries(hash_retrieval_backend PUBLIC
    $<$<BOOL:${BUILD_NUMA}>:${NUMA_INSTALL_DIR}/lib/libnuma.so>
    ${Torch_LIBRARIES}
)