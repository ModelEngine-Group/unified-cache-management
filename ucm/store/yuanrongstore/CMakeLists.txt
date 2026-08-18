set(YUANRONG_SDK_ROOT "" CACHE PATH "Path to the YuanRong DataSystem SDK")
set(YUANRONG_INCLUDE_DIR "" CACHE PATH "Path to YuanRong DataSystem headers")
set(YUANRONG_LIBRARY "" CACHE FILEPATH "Path to libdatasystem.so")
set(YUANRONG_ASCEND_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Path to Ascend toolkit")
file(GLOB YUANRONG_ASCEND_CANN_ROOTS LIST_DIRECTORIES true "/usr/local/Ascend/cann-*")

find_path(YUANRONG_ASCEND_ACL_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS
        "${YUANRONG_ASCEND_ROOT}"
        ${YUANRONG_ASCEND_CANN_ROOTS}
    PATHS
        /usr/local/Ascend/ascend-toolkit/latest
        /usr/local/Ascend/latest
    PATH_SUFFIXES include
)
find_library(YUANRONG_ASCEND_ACL_LIBRARY
    NAMES ascendcl
    HINTS
        "${YUANRONG_ASCEND_ROOT}"
        ${YUANRONG_ASCEND_CANN_ROOTS}
    PATHS
        /usr/local/Ascend/ascend-toolkit/latest
        /usr/local/Ascend/latest
    PATH_SUFFIXES lib64 lib
)

if(NOT YUANRONG_SDK_ROOT)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_EXECUTABLE)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c
                    "import pathlib, yr.datasystem as ds; print(pathlib.Path(ds.__file__).resolve().parent)"
            OUTPUT_VARIABLE YUANRONG_SDK_ROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
endif()

if(NOT YUANRONG_INCLUDE_DIR AND YUANRONG_SDK_ROOT)
    set(YUANRONG_INCLUDE_DIR "${YUANRONG_SDK_ROOT}/include")
endif()

if(NOT YUANRONG_LIBRARY AND YUANRONG_SDK_ROOT)
    find_library(YUANRONG_LIBRARY
        NAMES datasystem
        PATHS
            "${YUANRONG_SDK_ROOT}/lib"
            "${YUANRONG_SDK_ROOT}/lib64"
            "${YUANRONG_SDK_ROOT}/output/lib"
        NO_DEFAULT_PATH
    )
endif()
if(NOT YUANRONG_LIBRARY AND YUANRONG_SDK_ROOT)
    # find_library may miss the library when paths are not in cmake's default
    # search list (e.g. Python site-packages). Fall back to a direct existence check.
    foreach(_libdir lib lib64 output/lib)
        if(EXISTS "${YUANRONG_SDK_ROOT}/${_libdir}/libdatasystem.so")
            set(YUANRONG_LIBRARY "${YUANRONG_SDK_ROOT}/${_libdir}/libdatasystem.so")
            break()
        endif()
    endforeach()
endif()

if(EXISTS "${YUANRONG_INCLUDE_DIR}/datasystem/hetero_client.h"
   AND YUANRONG_LIBRARY
   AND YUANRONG_ASCEND_ACL_INCLUDE_DIR
   AND YUANRONG_ASCEND_ACL_LIBRARY)
    file(GLOB_RECURSE YUANRONG_STORE_SOURCES "./cc/*.cc")
    add_library(yuanrongstore SHARED ${YUANRONG_STORE_SOURCES})
    target_include_directories(yuanrongstore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cc
        ${YUANRONG_INCLUDE_DIR}
    )
    target_include_directories(yuanrongstore SYSTEM PRIVATE
        ${YUANRONG_ASCEND_ACL_INCLUDE_DIR}
    )
    target_link_libraries(yuanrongstore PUBLIC
        storeintf
        infra_logger
        fmt
        trans
        ${YUANRONG_LIBRARY}
        ${YUANRONG_ASCEND_ACL_LIBRARY}
    )
    get_filename_component(YUANRONG_LIBRARY_DIR "${YUANRONG_LIBRARY}" DIRECTORY)
    set_target_properties(yuanrongstore PROPERTIES
        BUILD_RPATH "${YUANRONG_LIBRARY_DIR}"
        INSTALL_RPATH "${YUANRONG_LIBRARY_DIR}"
    )
    file(RELATIVE_PATH INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    install(TARGETS yuanrongstore LIBRARY DESTINATION ${INSTALL_REL_PATH} COMPONENT ucm)

else()
    message(STATUS "yuanrongstore: skipping build - YuanRong SDK not found")
    message(STATUS "  YUANRONG_INCLUDE_DIR=${YUANRONG_INCLUDE_DIR}")
    message(STATUS "  YUANRONG_LIBRARY=${YUANRONG_LIBRARY}")
    message(STATUS "  YUANRONG_ASCEND_ACL_INCLUDE_DIR=${YUANRONG_ASCEND_ACL_INCLUDE_DIR}")
    message(STATUS "  YUANRONG_ASCEND_ACL_LIBRARY=${YUANRONG_ASCEND_ACL_LIBRARY}")
endif()
