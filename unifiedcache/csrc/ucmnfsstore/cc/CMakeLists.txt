file(GLOB_RECURSE UCMNFSSTORE_CC_SOURCE_FILES "*.cc")
if(RUNTIME_ENVIRONMENT STREQUAL "ascend")
    include(${CMAKE_SOURCE_DIR}/cmake/ascend.cmake)
    list(REMOVE_ITEM UCMNFSSTORE_CC_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/domain/device/cuda_device.cc")
else()
    list(REMOVE_ITEM UCMNFSSTORE_CC_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/domain/device/caclrt_device.cc")
endif()
add_library(ucmnfsstore.core STATIC ${UCMNFSSTORE_CC_SOURCE_FILES})
target_link_libraries(ucmnfsstore.core PUBLIC spdlog)
if(${RUNTIME_ENVIRONMENT} STREQUAL "ascend")
    target_link_libraries(ucmnfsstore.core PUBLIC Ascend::ascendcl)
else()
    target_link_libraries(ucmnfsstore.core PUBLIC Cuda::cudart)
endif()
target_include_directories(
    ucmnfsstore.core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/api
    ${CMAKE_CURRENT_SOURCE_DIR}/domain
    ${CMAKE_CURRENT_SOURCE_DIR}/infra
)
