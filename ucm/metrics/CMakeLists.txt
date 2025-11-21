function(generate_auto_register target_name stats_dir)
    # 1. 扫描 stats_dir 下所有含 REGISTER_STATS 宏的头文件
    file(GLOB_RECURSE STATS_HEADERS "${stats_dir}/*.h")
    set(REGISTER_CODE "")  # 存储生成的注册代码

    foreach(header ${STATS_HEADERS})
        # 读取文件内容，检查是否包含 REGISTER_STATS(ClassName)
        file(STRINGS "${header}" REGISTER_LINE REGEX "REGISTER_STATS\\([A-Za-z0-9_]+\\)")
        if(REGISTER_LINE)
            # 提取类名（如从 REGISTER_STATS(DramStats) 中提取 DramStats）
            string(REGEX MATCH "REGISTER_STATS\\(([A-Za-z0-9_]+)\\)" _ ${REGISTER_LINE})
            set(class_name ${CMAKE_MATCH_1})

            # 生成注册代码：包含头文件 + 实例化注册器
            # 注意：路径用双引号包裹，避免空格问题
            set(REGISTER_CODE "${REGISTER_CODE} \n\
                #include \"${header}\" \n\
                template class StatsRegistrar<${class_name}>; \n\
                static StatsRegistrar<${class_name}> ${class_name}_registrar;")
        endif()
    endforeach()

    # 2. 写入自动生成的源文件
    set(AUTO_REGISTER_FILE "${CMAKE_BINARY_DIR}/auto_stats_registrar.cpp")
    file(WRITE "${AUTO_REGISTER_FILE}" "${REGISTER_CODE}")

    # 3. 添加到目标编译
    target_sources(${target_name} PRIVATE "${AUTO_REGISTER_FILE}")
endfunction()

# --------------------------
# 主项目配置
# --------------------------
# 1. 编译 metrics 静态库（cc目录下的代码）
file(GLOB_RECURSE UCM_METRICS_CC_SOURCE_FILES "./cc/*.cc")  # 修正变量名（原UCMSTORE_NFS_CC_SOURCE_FILES）
add_library(metrics STATIC ${UCM_METRICS_CC_SOURCE_FILES})
# 修正头文件路径：对应你的目录结构（cc/stats等）
target_include_directories(metrics PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cc"  # 包含cc下的所有头文件（如cc/stats）
)
# 调用自动注册函数：扫描 cc/stats 目录（对应你的目录结构）
generate_auto_register(metrics "${CMAKE_CURRENT_SOURCE_DIR}/cc/stats")

# 2. 编译 pybind11 模块（cpy目录下的代码）
file(GLOB_RECURSE UCM_METRICS_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmmetrics ${UCM_METRICS_CPY_SOURCE_FILES})
# 链接 metrics 静态库
target_link_libraries(ucmmetrics PRIVATE metrics)
# 修正输出目录（可选，确保生成的.so在当前目录）
set_target_properties(ucmmetrics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)