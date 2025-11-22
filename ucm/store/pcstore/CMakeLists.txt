file(GLOB_RECURSE UCMSTORE_PC_CC_SOURCE_FILES "./cc/*.cc")
add_library(pcstore STATIC ${UCMSTORE_PC_CC_SOURCE_FILES})
target_include_directories(pcstore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
    ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
)
target_link_libraries(pcstore PUBLIC trans storeinfra)

file(GLOB_RECURSE UCMSTORE_PC_CPY_SOURCE_FILES "./cpy/*.cc")
pybind11_add_module(ucmpcstore ${UCMSTORE_PC_CPY_SOURCE_FILES})
target_link_libraries(ucmpcstore PRIVATE pcstore)
set_target_properties(ucmpcstore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
