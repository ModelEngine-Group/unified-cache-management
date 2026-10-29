# GSA 2.0 build slot.
# List C++ translation units in GSA2_SOURCES when the algorithm is ready.
# An empty list leaves the sandbox switch wired and compiles nothing.

set(GSA2_SOURCES
)

if(NOT GSA2_SOURCES)
    message(STATUS "UCM sandbox gsa2: no C++ sources, skip pybind module")
    return()
endif()

pybind11_add_module(gsa2 ${GSA2_SOURCES})
target_include_directories(gsa2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${UCM_ROOT_DIR}/ucm/store
)
target_link_libraries(gsa2 PRIVATE storeintf)

file(RELATIVE_PATH INSTALL_REL_PATH
     ${UCM_ROOT_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}
)
install(TARGETS gsa2 LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)
