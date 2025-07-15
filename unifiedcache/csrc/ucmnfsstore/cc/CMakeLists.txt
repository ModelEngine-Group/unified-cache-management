file(GLOB_RECURSE UCMNFSSTORE_CC_SOURCE_FILES "*.cc")
add_library(ucmnfsstore.core STATIC ${UCMNFSSTORE_CC_SOURCE_FILES})
target_link_libraries(ucmnfsstore.core PUBLIC spdlog)
target_include_directories(
    ucmnfsstore.core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/domain
    ${CMAKE_CURRENT_SOURCE_DIR}/infra
)
