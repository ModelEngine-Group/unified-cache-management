file(GLOB_RECURSE UCM_TRANSFER_FILE_CC_SOURCES CONFIGURE_DEPENDS "*.h" "*.cc")
add_library(ucm_transfer_file OBJECT ${UCM_TRANSFER_FILE_CC_SOURCES})
target_include_directories(ucm_transfer_file PUBLIC ${UCM_TRANSFER_DIR})
target_link_libraries(ucm_transfer_file PUBLIC ucm_build_options ucm_common)
