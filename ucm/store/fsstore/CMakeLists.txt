file(GLOB_RECURSE UCMSTORE_FS_CC_SOURCE_FILES "./cc/*.cc")
add_library(fsstore STATIC ${UCMSTORE_FS_CC_SOURCE_FILES})
target_include_directories(fsstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)
target_link_libraries(fsstore PUBLIC storeinfra storedevice)

file(GLOB_RECURSE UCMSTORE_FS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmfsstore ${UCMSTORE_FS_CPY_SOURCE_FILES})
target_link_libraries(ucmfsstore PRIVATE fsstore)
set_target_properties(ucmfsstore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
