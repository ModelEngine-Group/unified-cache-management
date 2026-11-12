file(GLOB_RECURSE UCM_TRANSFER_IPC_CC_SOURCES CONFIGURE_DEPENDS "*.h" "*.cc")
add_library(ucm_transfer_ipc OBJECT ${UCM_TRANSFER_IPC_CC_SOURCES})
target_link_libraries(ucm_transfer_ipc PUBLIC ucm_build_options ucm_include ucm_common)
