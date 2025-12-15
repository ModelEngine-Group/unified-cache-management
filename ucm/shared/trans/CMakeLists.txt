if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    add_subdirectory(ascend)
endif()
if(RUNTIME_ENVIRONMENT STREQUAL "maca")
    add_subdirectory(maca)
endif()
if(RUNTIME_ENVIRONMENT STREQUAL "cuda")
    add_subdirectory(cuda)
endif()
if(RUNTIME_ENVIRONMENT STREQUAL "simu")
    add_subdirectory(simu)
endif()
target_include_directories(trans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(trans PUBLIC infra_status)

file(GLOB_RECURSE UCMTRANS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmtrans ${UCMTRANS_CPY_SOURCE_FILES})
target_link_libraries(ucmtrans PRIVATE trans)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS ucmtrans LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)