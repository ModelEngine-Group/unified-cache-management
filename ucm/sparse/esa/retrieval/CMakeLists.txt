# 添加编译目标
pybind11_add_module(retrieval_backend cpy/retrieval_backend.cpp)

# 设置输出库的目录
file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS retrieval_backend LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

# 设置头文件目录，以确保 numaf.h 能找到
target_include_directories(retrieval_backend PUBLIC
    ${NUMA_INSTALL_DIR}/include
    ${Torch_INCLUDE_DIRS}
)

# 链接所需的库
target_link_libraries(retrieval_backend PUBLIC
    $<$<BOOL:${BUILD_NUMA}>:${NUMA_INSTALL_DIR}/lib/libnuma.so>
    ${Torch_LIBRARIES}
)
