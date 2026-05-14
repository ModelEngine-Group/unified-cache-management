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
if(NOT RUNTIME_ENVIRONMENT STREQUAL "cuda")
    target_sources(trans PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gdr/gdr_mr_buffer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gdr/gdr_config.cc
    )
endif()
if(RUNTIME_ENVIRONMENT STREQUAL "cuda" AND UCM_ENABLE_GDR_STREAM)
    target_compile_definitions(trans PUBLIC UCM_ENABLE_GDR_STREAM=1)
else()
    target_compile_definitions(trans PUBLIC UCM_ENABLE_GDR_STREAM=0)
endif()
target_include_directories(trans PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(trans PUBLIC infra_status infra_logger)

file(GLOB_RECURSE UCMTRANS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmtrans ${UCMTRANS_CPY_SOURCE_FILES})
target_link_libraries(ucmtrans PRIVATE trans)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS ucmtrans LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
