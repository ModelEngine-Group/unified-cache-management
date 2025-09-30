file(GLOB_RECURSE UCMSTORE_NFS_CC_SOURCE_FILES "./cc/*.cc")
add_library(nfsstore STATIC ${UCMSTORE_NFS_CC_SOURCE_FILES})
target_include_directories(nfsstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)
target_link_libraries(nfsstore PUBLIC storeinfra storedevice)

file(GLOB_RECURSE UCMSTORE_NFS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmnfsstore ${UCMSTORE_NFS_CPY_SOURCE_FILES})
target_link_libraries(ucmnfsstore PRIVATE nfsstore)
set_target_properties(ucmnfsstore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
