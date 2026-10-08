find_package(Python3 QUIET COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    add_custom_target(kv_metrics_codegen
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kv_metrics.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating standalone default metric descriptors"
        VERBATIM
    )
    add_custom_target(kv_metrics_config_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kv_metrics.py --check
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking standalone default metric descriptors"
        VERBATIM
    )
else()
    add_custom_target(kv_metrics_codegen
        COMMAND ${CMAKE_COMMAND} -E echo
                "Building standalone metrics requires a Python3 interpreter to generate default descriptors."
        COMMAND ${CMAKE_COMMAND} -E false
        VERBATIM
    )
    add_custom_target(kv_metrics_config_check DEPENDS kv_metrics_codegen)
endif()

add_library(kv_metrics SHARED src/metrics.cc)

target_include_directories(kv_metrics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kv_metrics PRIVATE pthread)
set_target_properties(kv_metrics PROPERTIES
    OUTPUT_NAME kv_metrics
    INSTALL_RPATH "$ORIGIN"
)

add_library(kv_metrics_standalone STATIC src/standalone_metrics_backend.cc)
add_dependencies(kv_metrics_standalone kv_metrics_codegen)
target_include_directories(kv_metrics_standalone
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kv_metrics_standalone PUBLIC kv_metrics pthread)

install(TARGETS kv_metrics
        LIBRARY DESTINATION kv_semantics/src
        COMPONENT ucm)
install(TARGETS kv_metrics
        LIBRARY DESTINATION ucm/store/asu
        COMPONENT ucm)
install(FILES config/kv_metrics.yaml
        DESTINATION kv_semantics/metrics
        COMPONENT ucm)
