if(RUNTIME_ENVIRONMENT STREQUAL "ascend"
   OR RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_subdirectory(p2p)
endif()
