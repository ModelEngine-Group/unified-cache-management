find_package(Threads REQUIRED)

set(AIV_PROVIDER_FOUNDATION_SOURCES
    src/runtime/acl_rt_loader.cpp
    src/runtime/ascendcl_loader.cpp
    src/runtime/hccp_v2_loader.cpp
    src/ub/status.cpp)

add_library(aiv_provider_foundation OBJECT ${AIV_PROVIDER_FOUNDATION_SOURCES})
target_include_directories(aiv_provider_foundation
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(aiv_provider_foundation PRIVATE cxx_std_17)
target_link_libraries(aiv_provider_foundation
    PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(aiv_provider_foundation PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
