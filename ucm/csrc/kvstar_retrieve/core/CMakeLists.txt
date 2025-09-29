file(GLOB_RECURSE CORE_SRC_FILES "*.cpp" "api/*.cpp" "api/**/*.cpp" "domain/*.cpp" "domain/**/*.cpp" "infra/*.cpp" "infra/**/*.cpp")

add_library(kvstar_retrieve.core STATIC ${CORE_SRC_FILES})

# --- 头文件路径 ---
# 不再使用${Torch_INCLUDE_DIRS}
# 直接使用从 setup.py 注入的、包含了所有路径的 EXTERNAL_INCLUDE_DIRS 变量。
# 这保证了编译器在编译时能看到正确的 -I/path/to/include 参数。
target_include_directories(kvstar_retrieve.core PUBLIC
        "." "api" "domain" "infra"
        ${EXTERNAL_INCLUDE_DIRS}
)

# --- 链接库 ---
target_link_libraries(kvstar_retrieve.core PUBLIC
        spdlog::spdlog
        fmt::fmt
        numa
        ${Torch_LIBRARIES}
)