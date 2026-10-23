add_library(kv_metrics_ucm_adapter SHARED src/ucm_metrics_adapter.cc)
target_include_directories(kv_metrics_ucm_adapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kv_metrics_ucm_adapter PUBLIC kv_metrics PRIVATE metrics infra_logger)

# CachedMetric owns bindings whose destructors live in this DSO. Keep it and its
# dependencies resident for the worker lifetime, even if a PipelineStore dlcloses
# libasustore.so. This also preserves adapter identity across store recreation.
target_link_options(kv_metrics_ucm_adapter PRIVATE "LINKER:-z,nodelete")
set_target_properties(kv_metrics_ucm_adapter PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../../shared/metrics"
)
install(TARGETS kv_metrics_ucm_adapter LIBRARY DESTINATION ucm/store/asu COMPONENT ucm)
