add_subdirectory(esa)
add_subdirectory(gsa)
add_subdirectory(gsa_on_device)

# kvstar is not supported on Ascend NPU yet
# Skip kvstar compilation on ascend platform to avoid installation errors
if(NOT RUNTIME_ENVIRONMENT STREQUAL "ascend")
    add_subdirectory(kvstar)
else()
    message(STATUS "Skipping kvstar on Ascend NPU platform (not yet supported)")
endif()
