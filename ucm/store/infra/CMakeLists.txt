add_library(storeinfra STATIC)
target_include_directories(storeinfra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE UCMSTORE_COMMON_FILE_SOURCE_FILES "file/*.cc")
target_sources(storeinfra PRIVATE ${UCMSTORE_COMMON_FILE_SOURCE_FILES})
target_link_libraries(storeinfra PUBLIC
    infra_status
    infra_logger
    infra_template
    infra_thread
    infra_time
)
