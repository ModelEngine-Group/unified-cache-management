file(GLOB_RECURSE UCM_ASU_STORE_CC_SOURCE_FILES "./cc/*.cc")
add_library(asustore SHARED ${UCM_ASU_STORE_CC_SOURCE_FILES})
target_include_directories(asustore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc
)
target_link_libraries(asustore PUBLIC storeintf asu_client infra_logger)

file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS asustore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
