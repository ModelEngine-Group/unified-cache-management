file(GLOB_RECURSE UCMNFSSTORE_CC_INFRA_SOURCE_FILES "*.cc")
add_library(infra STATIC ${UCMNFSSTORE_CC_INFRA_SOURCE_FILES})
target_link_libraries(infra PUBLIC spdlog)
target_include_directories(infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
