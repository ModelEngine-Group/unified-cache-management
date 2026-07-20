if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    add_subdirectory(p2p)
endif()
add_subdirectory(kv)
