set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(ucm_build_options INTERFACE)
target_compile_options(ucm_build_options INTERFACE
    -Wall -Werror
    $<$<CONFIG:Debug>:-O0;-g>
    $<$<CONFIG:Release>:-O2;-fstack-protector-strong;-D_FORTIFY_SOURCE=2>)
target_link_options(ucm_build_options INTERFACE
    -Wl,-z,relro,-z,now
    $<$<CONFIG:Release>:-s>)

add_library(ucm_include INTERFACE)
target_include_directories(ucm_include INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/transfer
    ${CMAKE_CURRENT_SOURCE_DIR}/store
    ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

add_subdirectory(vendor)
add_subdirectory(common)
add_subdirectory(transfer)
add_subdirectory(store)
add_subdirectory(proxy)
if(BUILD_UNIT_TESTS)
    add_subdirectory(bench)
    add_subdirectory(test)
endif()
