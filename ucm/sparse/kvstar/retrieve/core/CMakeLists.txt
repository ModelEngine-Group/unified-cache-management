file(GLOB_RECURSE CORE_SRC_FILES "*.cpp" "api/*.cpp" "api/**/*.cpp" "domain/*.cpp" "domain/**/*.cpp" "infra/*.cpp" "infra/**/*.cpp")

add_library(kvstar_retrieve.core STATIC ${CORE_SRC_FILES})

get_target_property(SPDLOG_INCLUDE_DIRS spdlog INTERFACE_INCLUDE_DIRECTORIES)
if(SPDLOG_INCLUDE_DIRS)
        target_include_directories(kvstar_retrieve.core BEFORE PUBLIC ${SPDLOG_INCLUDE_DIRS})
endif()

get_target_property(FMT_INCLUDE_DIRS fmt INTERFACE_INCLUDE_DIRECTORIES)
if(FMT_INCLUDE_DIRS)
        target_include_directories(kvstar_retrieve.core BEFORE PUBLIC ${FMT_INCLUDE_DIRS})
endif()

target_include_directories(kvstar_retrieve.core PUBLIC
        "." "api" "domain" "infra"
        ${NUMA_INSTALL_DIR}/include
        ${CMAKE_SOURCE_DIR}/ucm/shared/infra
)

target_link_libraries(kvstar_retrieve.core PUBLIC
        $<$<BOOL:${BUILD_NUMA}>:${NUMA_INSTALL_DIR}/lib/libnuma.so>
        ${Torch_LIBRARIES}
        infra_logger
)
