file(GLOB_RECURSE CORE_SRC_FILES "*.cpp" "api/*.cpp" "api/**/*.cpp" "domain/*.cpp" "domain/**/*.cpp" "infra/*.cpp" "infra/**/*.cpp")

add_library(kvstar_retrieve.core STATIC ${CORE_SRC_FILES})

target_include_directories(kvstar_retrieve.core PUBLIC
        "." "api" "domain" "infra"
        ${EXTERNAL_INCLUDE_DIRS}
        ${NUMA_INSTALL_DIR}/include
)

target_link_libraries(kvstar_retrieve.core PUBLIC
        spdlog::spdlog
        fmt::fmt
        ${NUMA_INSTALL_DIR}/lib/libnuma.so
        ${Torch_LIBRARIES}
)
