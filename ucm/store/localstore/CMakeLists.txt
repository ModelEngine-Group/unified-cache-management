file(GLOB_RECURSE UCMSTORE_LOCAL_CC_SOURCE_FILES "./cc/*.cc")
add_library(localstore STATIC ${UCMSTORE_LOCAL_CC_SOURCE_FILES})
target_include_directories(localstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)
target_link_libraries(localstore PUBLIC storeinfra storetask)

file(GLOB_RECURSE UCMSTORE_LOCAL_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmlocalstore ${UCMSTORE_LOCAL_CPY_SOURCE_FILES})
target_link_libraries(ucmlocalstore PRIVATE localstore)
set_target_properties(ucmlocalstore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
