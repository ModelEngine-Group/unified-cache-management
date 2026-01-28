file(GLOB_RECURSE UCMINFRA_STATUS_SOURCE_FILES "status/*.*")
add_library(infra_status OBJECT ${UCMINFRA_STATUS_SOURCE_FILES})
target_include_directories(infra_status PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infra_status PUBLIC fmt)

add_subdirectory(logger)

file(GLOB_RECURSE UCMINFRA_TEMPLATE_SOURCE_FILES "template/*.*")
add_library(infra_template OBJECT ${UCMINFRA_TEMPLATE_SOURCE_FILES})
target_include_directories(infra_template PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UCMINFRA_THREAD_SOURCE_FILES "thread/*.*")
add_library(infra_thread OBJECT ${UCMINFRA_THREAD_SOURCE_FILES})
target_include_directories(infra_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE UCMINFRA_TIME_SOURCE_FILES "time/*.*")
add_library(infra_time OBJECT ${UCMINFRA_TIME_SOURCE_FILES})
target_include_directories(infra_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
