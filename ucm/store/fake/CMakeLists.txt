file(GLOB_RECURSE UCM_FAKE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(fakestore STATIC ${UCM_FAKE_STORE_CC_SOURCE_FILES})
target_include_directories(fakestore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
)
target_link_libraries(fakestore PUBLIC storeintf infra_logger)

file(GLOB_RECURSE UCM_FAKE_STORE_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmfakestore ${UCM_FAKE_STORE_CPY_SOURCE_FILES})
target_link_libraries(ucmfakestore PRIVATE fakestore)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucmfakestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
