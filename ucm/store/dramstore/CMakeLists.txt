file(GLOB_RECURSE UCMSTORE_DRAM_CC_SOURCE_FILES "./cc/*.cc")
add_library(dramstore STATIC ${UCMSTORE_DRAM_CC_SOURCE_FILES})
target_include_directories(dramstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(dramstore PUBLIC storeinfra)

file(GLOB_RECURSE UCMSTORE_DRAM_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmdramstore ${UCMSTORE_DRAM_CPY_SOURCE_FILES})
target_link_libraries(ucmdramstore PRIVATE dramstore)
