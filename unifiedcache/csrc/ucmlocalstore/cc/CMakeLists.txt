file(GLOB_RECURSE UCM_LOCAL_STORE_SOURCE_FILES "*.cc")
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    include(${CMAKE_SOURCE_DIR}/cmake/ascend.cmake)
    list(REMOVE_ITEM UCM_LOCAL_STORE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/domain/device/cuda_device.cc")
else()
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
    list(REMOVE_ITEM UCM_LOCAL_STORE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/domain/device/aclrt_device.cc")
endif()
add_library(ucmlocalstore.core STATIC ${UCM_LOCAL_STORE_SOURCE_FILES})
target_link_libraries(ucmlocalstore.core PUBLIC spdlog)
if(${RUNTIME_ENVIRONMENT} STREQUAL "ascend")
    target_link_libraries(ucmlocalstore.core PUBLIC Ascend::ascendcl)
else()
    target_link_libraries(ucmlocalstore.core PUBLIC Cuda::cudart)
endif()
target_include_directories(
    ucmlocalstore.core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/api/
    ${CMAKE_CURRENT_SOURCE_DIR}/domain/
    ${CMAKE_CURRENT_SOURCE_DIR}/infra/
)