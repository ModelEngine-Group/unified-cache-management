file(GLOB_RECURSE UCM_POSIX_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(posixstore STATIC ${UCM_POSIX_STORE_CC_SOURCE_FILES})
target_include_directories(posixstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
)
target_link_libraries(posixstore PUBLIC storeintf infra_logger)

file(GLOB_RECURSE UCM_POSIX_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmposixstore ${UCM_POSIX_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmposixstore PRIVATE posixstore)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmposixstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
