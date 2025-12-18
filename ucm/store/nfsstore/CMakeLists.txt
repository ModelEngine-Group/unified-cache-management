add_subdirectory(device)
file(GLOB_RECURSE UCMSTORE_NFS_CC_SOURCE_FILES "./cc/*.cc")
add_library(nfsstore STATIC ${UCMSTORE_NFS_CC_SOURCE_FILES})
target_include_directories(nfsstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)
target_link_libraries(nfsstore PUBLIC storeintf storedevice infra_logger metrics)

file(GLOB_RECURSE UCMSTORE_NFS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmnfsstore ${UCMSTORE_NFS_CPY_SOURCE_FILES})
target_link_libraries(ucmnfsstore PRIVATE nfsstore)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS ucmnfsstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
