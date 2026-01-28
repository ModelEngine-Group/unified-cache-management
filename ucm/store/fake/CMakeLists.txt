file(GLOB_RECURSE UCM_FAKE_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(fakestore SHARED ${UCM_FAKE_STORE_CC_SOURCE_FILES})
target_include_directories(fakestore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
target_link_libraries(fakestore PUBLIC storeintf infra_logger)

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS fakestore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
