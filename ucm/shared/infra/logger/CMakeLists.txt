file(GLOB UCMINFRA_LOGGER_SOURCE_FILES "*.h")
file(GLOB UCMINFRA_LOGGER_CC_FILES "${LOGGER_BACKEND}/cc/*.cc")

add_library(infra_logger SHARED ${UCMINFRA_LOGGER_SOURCE_FILES} ${UCMINFRA_LOGGER_CC_FILES})
target_include_directories(infra_logger PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${LOGGER_BACKEND}/cc
)
target_link_libraries(infra_logger PUBLIC fmt spdlog zlib)

file(GLOB UCMINFRA_LOGGER_CPY_FILES CONFIGURE_DEPENDS "${LOGGER_BACKEND}/cpy/*.cc")
pybind11_add_module(ucmlogger ${UCMINFRA_LOGGER_CPY_FILES})
target_include_directories(ucmlogger PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${LOGGER_BACKEND}/cc
)
target_link_libraries(ucmlogger PRIVATE infra_logger pybind11::module)

# Ensure ucmlogger can find libinfra_logger.so in the same directory at runtime
target_link_options(ucmlogger PRIVATE "-Wl,-rpath,\$ORIGIN")
target_link_options(infra_logger PRIVATE "-Wl,-rpath,\$ORIGIN")

file(RELATIVE_PATH UCMINFRA_INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/..
)
install(TARGETS ucmlogger infra_logger
        LIBRARY DESTINATION ${UCMINFRA_INSTALL_REL_PATH}
        COMPONENT ucm)

