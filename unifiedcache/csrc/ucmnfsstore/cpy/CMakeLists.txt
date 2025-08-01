file(GLOB_RECURSE UCMNFSSTORE_CPY_SOURCE_FILES "*.cc")
pybind11_add_module(ucmnfsstore ${UCMNFSSTORE_CPY_SOURCE_FILES})
target_link_libraries(ucmnfsstore PRIVATE ucmnfsstore.core)
