file(GLOB_RECURSE UCM_POSIX_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(posixstore SHARED ${UCM_POSIX_STORE_CC_SOURCE_FILES})
target_include_directories(posixstore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cc)
find_library(URING_LIB uring)
target_link_libraries(posixstore PUBLIC storeintf infra_logger ${URING_LIB})

file(RELATIVE_PATH INSTALL_REL_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS posixstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
