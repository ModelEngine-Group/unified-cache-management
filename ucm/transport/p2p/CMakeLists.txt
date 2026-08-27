if(NOT (RUNTIME_ENVIRONMENT STREQUAL "ascend"
        OR RUNTIME_ENVIRONMENT STREQUAL "ascend-a3"
        OR RUNTIME_ENVIRONMENT STREQUAL "simu"))
    return()
endif()

if(NOT DEFINED ASCEND_ROOT)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Path to Ascend root directory")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_ROOT "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "Path to Ascend root directory")
    else()
        set(ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH
            "Path to Ascend root directory")
    endif()
endif()

if(NOT DEFINED HIXL_ROOT AND DEFINED ENV{HIXL_ROOT})
    set(HIXL_ROOT "$ENV{HIXL_ROOT}" CACHE PATH "HIXL installation prefix")
endif()
if(NOT DEFINED HIXL_INCLUDE_DIR AND DEFINED ENV{HIXL_INCLUDE_DIR})
    set(HIXL_INCLUDE_DIR "$ENV{HIXL_INCLUDE_DIR}" CACHE PATH "Directory containing hixl headers")
endif()
if(NOT DEFINED HIXL_LIB_DIR AND DEFINED ENV{HIXL_LIB_DIR})
    set(HIXL_LIB_DIR "$ENV{HIXL_LIB_DIR}" CACHE PATH "Directory containing the HIXL library")
endif()

set(P2P_ASCEND_ROOT_HINTS ${ASCEND_ROOT})
foreach(P2P_ASCEND_PARENT_HINT
        "${ASCEND_ROOT}" "${ASCEND_ROOT}/.." "${ASCEND_ROOT}/../..")
    file(GLOB P2P_CANN_ROOT_HINTS LIST_DIRECTORIES true
        "${P2P_ASCEND_PARENT_HINT}/cann-*"
    )
    list(APPEND P2P_ASCEND_ROOT_HINTS ${P2P_CANN_ROOT_HINTS})
endforeach()
list(REMOVE_DUPLICATES P2P_ASCEND_ROOT_HINTS)

set(P2P_HIXL_INCLUDE_HINTS
    ${HIXL_INCLUDE_DIR}
    ${HIXL_ROOT}
    ${HIXL_ROOT}/include
    ${HIXL_ROOT}/aarch64-linux/include
    ${HIXL_ROOT}/arm64-linux/include
)
foreach(P2P_ASCEND_ROOT_HINT IN LISTS P2P_ASCEND_ROOT_HINTS)
    list(APPEND P2P_HIXL_INCLUDE_HINTS
        ${P2P_ASCEND_ROOT_HINT}
        ${P2P_ASCEND_ROOT_HINT}/include
        ${P2P_ASCEND_ROOT_HINT}/aarch64-linux/include
        ${P2P_ASCEND_ROOT_HINT}/arm64-linux/include
    )
endforeach()

find_path(P2P_HIXL_INCLUDE_DIR
    NAMES hixl/hixl.h
    HINTS ${P2P_HIXL_INCLUDE_HINTS}
)

set(P2P_HIXL_LIBRARY_HINTS
    ${HIXL_LIB_DIR}
    ${HIXL_ROOT}
    ${HIXL_ROOT}/lib
    ${HIXL_ROOT}/lib64
    ${HIXL_ROOT}/aarch64-linux/lib64
    ${HIXL_ROOT}/arm64-linux/lib64
)
foreach(P2P_ASCEND_ROOT_HINT IN LISTS P2P_ASCEND_ROOT_HINTS)
    list(APPEND P2P_HIXL_LIBRARY_HINTS
        ${P2P_ASCEND_ROOT_HINT}
        ${P2P_ASCEND_ROOT_HINT}/lib
        ${P2P_ASCEND_ROOT_HINT}/lib64
        ${P2P_ASCEND_ROOT_HINT}/aarch64-linux/lib64
        ${P2P_ASCEND_ROOT_HINT}/arm64-linux/lib64
    )
endforeach()

find_library(P2P_HIXL_LIBRARY
    NAMES cann_hixl
    HINTS ${P2P_HIXL_LIBRARY_HINTS}
)

find_library(P2P_METADEF_LIBRARY
    NAMES metadef
    HINTS ${P2P_HIXL_LIBRARY_HINTS}
)

set(P2P_TRANSPORT_SOURCES
    src/common/binary_codec.cpp
    src/control/control_channel.cpp
    src/control/control_protocol.cpp
    src/core/transport_manager.cpp
    src/channels/tcp/tcp_message_channel.cpp
)

set(P2P_HIXL_AVAILABLE OFF)
if(UCM_RUNTIME_ASCEND_FAMILY
   AND P2P_HIXL_INCLUDE_DIR
   AND P2P_HIXL_LIBRARY
   AND P2P_METADEF_LIBRARY)
    set(P2P_HIXL_AVAILABLE ON)
    list(APPEND P2P_TRANSPORT_SOURCES
        src/common/acl_runtime_context.cpp
        src/protocols/hixl/hixl_instance.cpp
        src/protocols/hixl/hixl_transport.cpp
    )
    message(STATUS "P2P HIXL transport backend is enabled")
elseif(UCM_RUNTIME_ASCEND_FAMILY)
    message(STATUS
        "P2P HIXL transport backend is disabled: "
        "hixl/hixl.h, libcann_hixl, or libmetadef was not found"
    )
else()
    message(STATUS
        "P2P HIXL transport backend is disabled for "
        "RUNTIME_ENVIRONMENT=${RUNTIME_ENVIRONMENT}"
    )
endif()

add_library(ucm_p2p_transport SHARED ${P2P_TRANSPORT_SOURCES})
target_include_directories(ucm_p2p_transport
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:ucm/transport/p2p/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_compile_definitions(ucm_p2p_transport PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(ucm_p2p_transport
    PUBLIC
        infra_status
    PRIVATE
        infra_logger
        pthread
        rt
)
if(P2P_HIXL_AVAILABLE)
    target_compile_definitions(ucm_p2p_transport PRIVATE UCM_P2P_HAS_HIXL)
    target_include_directories(ucm_p2p_transport PRIVATE
        ${ASCEND_ROOT}/include
        ${ASCEND_ROOT}/aarch64-linux/include
        ${ASCEND_ROOT}/arm64-linux/include
        ${P2P_HIXL_INCLUDE_DIR}
    )
    target_link_directories(ucm_p2p_transport PRIVATE
        ${ASCEND_ROOT}/lib64
        ${ASCEND_ROOT}/aarch64-linux/lib64
        ${ASCEND_ROOT}/aarch64-linux/devlib
        ${ASCEND_ROOT}/arm64-linux/lib64
        ${ASCEND_ROOT}/arm64-linux/devlib
    )
    target_link_libraries(ucm_p2p_transport PRIVATE
        ascendcl
        ${P2P_HIXL_LIBRARY}
        ${P2P_METADEF_LIBRARY}
    )
endif()
set_target_properties(ucm_p2p_transport PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

file(RELATIVE_PATH P2P_INSTALL_REL_PATH ${UCM_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS ucm_p2p_transport
    LIBRARY DESTINATION ${P2P_INSTALL_REL_PATH}
    COMPONENT ucm
)
install(DIRECTORY include/
    DESTINATION ${P2P_INSTALL_REL_PATH}/include
    COMPONENT ucm
    FILES_MATCHING PATTERN "*.h"
)
