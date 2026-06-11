file(GLOB_RECURSE UCMMETRICS_CC_SOURCE_FILES "./cc/*.cc")
add_library(metrics SHARED ${UCMMETRICS_CC_SOURCE_FILES})
if(WIN32)
    set_target_properties(metrics PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_include_directories(metrics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)

file(GLOB_RECURSE UCMMETRICS_CPY_SOURCE_FILES CONFIGURE_DEPENDS "./cpy/*.cc")
pybind11_add_module(ucmmetrics ${UCMMETRICS_CPY_SOURCE_FILES})
target_link_libraries(ucmmetrics PRIVATE metrics)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS metrics LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
install(TARGETS ucmmetrics LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
