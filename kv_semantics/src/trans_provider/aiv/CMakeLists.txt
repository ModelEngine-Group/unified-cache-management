find_package(Threads REQUIRED)

set(AIV_PROVIDER_FOUNDATION_SOURCES
    src/runtime/acl_rt_loader.cc
    src/runtime/ascendcl_loader.cc
    src/runtime/hccp_v2_loader.cc
    src/ub/status.cc)

add_library(aiv_provider_foundation OBJECT ${AIV_PROVIDER_FOUNDATION_SOURCES})
target_include_directories(aiv_provider_foundation
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${UCM_ROOT_DIR}/ucm/shared/infra/logger"
        "${UCM_ROOT_DIR}/ucm/shared/infra/logger/cc")
target_compile_features(aiv_provider_foundation PRIVATE cxx_std_17)
target_compile_definitions(aiv_provider_foundation PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(aiv_provider_foundation
    PRIVATE Threads::Threads ${CMAKE_DL_LIBS} fmt spdlog)
set_target_properties(aiv_provider_foundation PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
