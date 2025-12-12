find_path(HF3FS_USRBIO_INCLUDE_DIR NAMES hf3fs_usrbio.h PATHS /usr/local/3fs/src/lib/api NO_DEFAULT_PATH)
find_library(HF3FS_USRBIO_LIBRARY NAMES hf3fs_api_shared PATHS /usr/local/3fs/src/lib/rs/hf3fs-usrbio-sys/lib NO_DEFAULT_PATH)

if(HF3FS_USRBIO_INCLUDE_DIR AND HF3FS_USRBIO_LIBRARY)
    file(GLOB_RECURSE UCMSTORE_DS3FS_CC_SOURCE_FILES "./cc/*.cc")
    add_library(ds3fsstore STATIC ${UCMSTORE_DS3FS_CC_SOURCE_FILES})
    target_include_directories(ds3fsstore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc/api
        ${CMAKE_CURRENT_SOURCE_DIR}/cc/domain
    )

    target_include_directories(ds3fsstore PRIVATE ${HF3FS_USRBIO_INCLUDE_DIR})
    target_link_libraries(ds3fsstore PRIVATE ${HF3FS_USRBIO_LIBRARY})
    target_link_libraries(ds3fsstore PUBLIC trans storeinfra)

    file(GLOB_RECURSE UCMSTORE_DS3FS_CPY_SOURCE_FILES "./cpy/*.cc")
    pybind11_add_module(ucmds3fsstore ${UCMSTORE_DS3FS_CPY_SOURCE_FILES})
    target_link_libraries(ucmds3fsstore PRIVATE ds3fsstore)
    set_target_properties(ucmds3fsstore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(STATUS "ds3fsstore: Skipping build - required HF3FS dependencies not found")
    if(NOT HF3FS_USRBIO_INCLUDE_DIR)
        message(STATUS "  - Missing: hf3fs_usrbio.h ")
    endif()
    if(NOT HF3FS_USRBIO_LIBRARY)
        message(STATUS "  - Missing: hf3fs_api_shared library ")
    endif()
    message(STATUS "  Please ensure HF3FS dependencies are installed or build paths are correct")
endif()