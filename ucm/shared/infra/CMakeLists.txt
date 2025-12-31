file(GLOB_RECURSE UCMINFRA_STATUS_SOURCE_FILES "status/*.*")
add_library(infra_status OBJECT ${UCMINFRA_STATUS_SOURCE_FILES})
target_include_directories(infra_status PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infra_status PUBLIC fmt)

file(GLOB UCMINFRA_LOGGER_SOURCE_FILES "logger/*.*")
file(GLOB UCMINFRA_LOGGER_DETAIL_SOURCE_FILES "logger/${LOGGER_BACKEND}/*.cc")
file(GLOB BINDINGS_SRCS CONFIGURE_DEPENDS "logger/${LOGGER_BACKEND}/cpy/spdlog_logger.py.cc")
file(GLOB SOURCE_LOCATION_SRCS CONFIGURE_DEPENDS "logger/${LOGGER_BACKEND}/cpy/source_location.py.cc")

add_library(infra_logger OBJECT ${UCMINFRA_LOGGER_SOURCE_FILES} ${UCMINFRA_LOGGER_DETAIL_SOURCE_FILES})
target_include_directories(infra_logger PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infra_logger PUBLIC fmt spdlog zlib)

pybind11_add_module(spdlog_logger ${BINDINGS_SRCS})
target_include_directories(spdlog_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spdlog_logger PRIVATE infra_logger pybind11::module zlib)

pybind11_add_module(source_location ${SOURCE_LOCATION_SRCS})
target_include_directories(source_location PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(source_location PRIVATE infra_logger pybind11::module zlib)

file(RELATIVE_PATH UCMINFRA_INSTALL_REL_PATH
     ${CMAKE_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS spdlog_logger source_location LIBRARY DESTINATION ${UCMINFRA_INSTALL_REL_PATH} COMPONENT ucm)

file(GLOB_RECURSE UCMINFRA_TEMPLATE_SOURCE_FILES "template/*.*")
add_library(infra_template OBJECT ${UCMINFRA_TEMPLATE_SOURCE_FILES})
target_include_directories(infra_template PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UCMINFRA_THREAD_SOURCE_FILES "thread/*.*")
add_library(infra_thread OBJECT ${UCMINFRA_THREAD_SOURCE_FILES})
target_include_directories(infra_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UCMINFRA_TIME_SOURCE_FILES "time/*.*")
add_library(infra_time OBJECT ${UCMINFRA_TIME_SOURCE_FILES})
target_include_directories(infra_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
