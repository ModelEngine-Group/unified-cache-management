file(GLOB_RECURSE UCMNFSSTORE_CC_API_SOURCE_FILES "*.cc")
add_library(api STATIC ${UCMNFSSTORE_CC_API_SOURCE_FILES})
target_link_libraries(api PUBLIC infra domain)
target_include_directories(api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
