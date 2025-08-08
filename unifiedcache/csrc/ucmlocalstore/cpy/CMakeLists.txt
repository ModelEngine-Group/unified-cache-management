file(GLOB_RECURSE UCMLOCALSTORE_CPY_SOURCE_FILES "*.cc")
pybind11_add_module(ucmlocalstore ${UCMLOCALSTORE_CPY_SOURCE_FILES})
target_link_libraries(ucmlocalstore PRIVATE ucmlocalstore.core)
