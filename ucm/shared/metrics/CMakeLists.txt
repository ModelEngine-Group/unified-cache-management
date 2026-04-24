file(GLOB_RECURSE UCMMETRICS_CC_SOURCE_FILES "./cc/*.cc")
# metrics must be SHARED: store backends (cachestore.so, posixstore.so) and
# the ucmmetrics pybind module all need to share one Metrics singleton
# instance so C++ UpdateStats() calls are visible to Python's
# get_all_stats_and_clear(). A STATIC library would be duplicated into each
# consumer .so and each copy would own a distinct singleton, dropping all
# store-side metrics.
add_library(metrics SHARED ${UCMMETRICS_CC_SOURCE_FILES})
target_include_directories(metrics PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)

file(GLOB_RECURSE UCMMETRICS_CPY_SOURCE_FILES CONFIGURE_DEPENDS "./cpy/*.cc")
pybind11_add_module(ucmmetrics ${UCMMETRICS_CPY_SOURCE_FILES})
target_link_libraries(ucmmetrics PRIVATE metrics)
# ucmmetrics.so and libmetrics.so live in the same directory.
set_target_properties(ucmmetrics PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS metrics LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
install(TARGETS ucmmetrics LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
