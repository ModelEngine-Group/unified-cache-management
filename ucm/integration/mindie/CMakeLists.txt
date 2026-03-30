pybind11_add_module(
    uc_hash_ext
    hash_mindie/uc_hash_ext.cpp
)
target_compile_options(uc_hash_ext PRIVATE -O3 -march=native)
target_link_libraries(uc_hash_ext PRIVATE pybind11::module)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS uc_hash_ext LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
