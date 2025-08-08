file(GLOB_RECURSE UCM_LOCAL_STORE_SOURCE_FILES "*.cc")
add_library(ucmlocalstore SHARED ${UCM_LOCAL_STORE_SOURCE_FILES})
target_link_libraries(ucmlocalstore PUBLIC spdlog)
target_include_directories(
    ucmlocalstore
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/api/
    ${CMAKE_CURRENT_SOURCE_DIR}/domain/
    ${CMAKE_CURRENT_SOURCE_DIR}/infra/
)
