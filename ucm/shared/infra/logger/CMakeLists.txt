file(GLOB UCMINFRA_LOGGER_CC_FILES 
    "./cc/*.cc"
    "./*.cc"
)
add_library(infra_logger STATIC ${UCMINFRA_LOGGER_CC_FILES})
target_include_directories(infra_logger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(infra_logger PUBLIC fmt spdlog zlibstatic)

file(GLOB UCMINFRA_LOGGER_CPY_FILES CONFIGURE_DEPENDS "./cpy/*.cc")
pybind11_add_module(ucmlogger ${UCMINFRA_LOGGER_CPY_FILES})
target_link_libraries(ucmlogger PRIVATE infra_logger pybind11::module)

file(RELATIVE_PATH UCMINFRA_INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/..
)
install(TARGETS ucmlogger
        LIBRARY DESTINATION ${UCMINFRA_INSTALL_REL_PATH}
        COMPONENT ucm)