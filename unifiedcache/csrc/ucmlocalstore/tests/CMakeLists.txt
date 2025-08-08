file(GLOB_RECURSE UCM_LOCAL_STORE_TEST_SOURCE_FILES "*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/../cc/*.cc")
add_executable(ucmlocalstore.test ${UCM_LOCAL_STORE_TEST_SOURCE_FILES})
target_link_libraries(ucmlocalstore.test PRIVATE gtest_main gtest mockcpp atomic)
target_include_directories(
    ucmlocalstore.test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/api/
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/domain/
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/infra/
)
gtest_discover_tests(ucmlocalstore.test)