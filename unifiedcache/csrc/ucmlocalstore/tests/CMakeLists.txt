file(GLOB_RECURSE UCM_LOCAL_STORE_TEST_SOURCE_FILES "*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/../cc/*.cc")
if(${RUNTIME_ENVIRONMENT} STREQUAL "ascend")
    include(${CMAKE_SOURCE_DIR}/cmake/ascend.cmake)
    list(REMOVE_ITEM UCM_LOCAL_STORE_TEST_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../cc/domain/device/cuda_device.cc")
else()
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
    list(REMOVE_ITEM UCM_LOCAL_STORE_TEST_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../cc/domain/device/aclrt_device.cc")
endif()
add_executable(ucmlocalstore.test ${UCM_LOCAL_STORE_TEST_SOURCE_FILES})
target_link_libraries(ucmlocalstore.test PRIVATE gtest_main gtest mockcpp atomic)
if(${RUNTIME_ENVIRONMENT} STREQUAL "ascend")
    target_link_libraries(ucmlocalstore.test PUBLIC Ascend::ascendcl)
else()
    target_link_libraries(ucmlocalstore.test PUBLIC Cuda::cudart)
endif()
target_include_directories(
    ucmlocalstore.test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/api/
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/domain/
    ${CMAKE_CURRENT_SOURCE_DIR}/../cc/infra/
)
gtest_discover_tests(ucmlocalstore.test)