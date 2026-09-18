# cmake/DetectRuntime.cmake
# Runtime backend detection module

function(detect_runtime_backend)
    set(RUNTIME_BACKEND "Simulation" PARENT_SCOPE)

    # ============================================
    # CUDA Detection
    # ============================================
    set(_CUDA_ROOT "")

    # Priority: CACHE variable > Environment variable > Default path
    if(DEFINED CACHE{CUDA_ROOT} AND NOT "$CACHE{CUDA_ROOT}" STREQUAL "")
        set(_CUDA_ROOT "$CACHE{CUDA_ROOT}")
    elseif(DEFINED ENV{CUDA_HOME})
        set(_CUDA_ROOT "$ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH})
        set(_CUDA_ROOT "$ENV{CUDA_PATH}")
    elseif(EXISTS "/usr/local/cuda")
        set(_CUDA_ROOT "/usr/local/cuda")
    endif()

    if(_CUDA_ROOT)
        find_path(_CUDA_INCLUDE_DIR cuda_runtime.h
            PATHS "${_CUDA_ROOT}/include"
            NO_DEFAULT_PATH
        )
        find_library(_CUDA_LIBRARY cudart
            PATHS "${_CUDA_ROOT}/lib64" "${_CUDA_ROOT}/lib"
            NO_DEFAULT_PATH
        )

        if(_CUDA_INCLUDE_DIR AND _CUDA_LIBRARY)
            message(STATUS "Found CUDA runtime: ${_CUDA_ROOT}")
            message(STATUS "  Include: ${_CUDA_INCLUDE_DIR}")
            message(STATUS "  Library: ${_CUDA_LIBRARY}")

            # Create imported target
            add_library(CUDA::Runtime INTERFACE IMPORTED GLOBAL)
            target_include_directories(CUDA::Runtime INTERFACE "${_CUDA_INCLUDE_DIR}")
            target_link_libraries(CUDA::Runtime INTERFACE "${_CUDA_LIBRARY}")

            # Set global variables
            set(RUNTIME_BACKEND "CUDA" PARENT_SCOPE)
            set(CUDA_FOUND TRUE PARENT_SCOPE)
            set(CUDA_ROOT "${_CUDA_ROOT}" PARENT_SCOPE)

            # Enable CUDA language
            set(CMAKE_CUDA_COMPILER "${_CUDA_ROOT}/bin/nvcc" PARENT_SCOPE)
            set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 PARENT_SCOPE)

            return()
        endif()
    endif()

    # ============================================
    # Ascend Detection
    # ============================================
    set(_ASCEND_ROOT "")

    if(DEFINED CACHE{ASCEND_ROOT} AND NOT "$CACHE{ASCEND_ROOT}" STREQUAL "")
        set(_ASCEND_ROOT "$CACHE{ASCEND_ROOT}")
    elseif(DEFINED ENV{ASCEND_HOME})
        set(_ASCEND_ROOT "$ENV{ASCEND_HOME}")
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(_ASCEND_ROOT "$ENV{ASCEND_TOOLKIT_HOME}")
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(_ASCEND_ROOT "$ENV{ASCEND_HOME_PATH}")
    elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
        set(_ASCEND_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
    endif()

    if(_ASCEND_ROOT)
        find_path(_ASCEND_INCLUDE_DIR acl/acl.h
            PATHS "${_ASCEND_ROOT}/include"
            NO_DEFAULT_PATH
        )
        find_library(_ASCEND_LIBRARY ascendcl
            PATHS "${_ASCEND_ROOT}/lib64" "${_ASCEND_ROOT}/lib"
            NO_DEFAULT_PATH
        )

        if(_ASCEND_INCLUDE_DIR AND _ASCEND_LIBRARY)
            find_library(_ASCEND_RUNTIME_LIBRARY runtime
                PATHS
                    "${_ASCEND_ROOT}/lib64"
                    "${_ASCEND_ROOT}/lib"
                    "${_ASCEND_ROOT}/runtime/lib64"
                    "${_ASCEND_ROOT}/runtime/lib"
                    "${_ASCEND_ROOT}/aarch64-linux/lib64"
                    "${_ASCEND_ROOT}/aarch64-linux/lib"
                    "/usr/local/Ascend/cann/aarch64-linux/lib64"
                    "/usr/local/Ascend/cann/aarch64-linux/lib"
                NO_DEFAULT_PATH
            )
            find_path(_ASCEND_FFTS_INCLUDE_DIR
                NAMES runtime/rt_ffts_plus.h rt_external_ffts.h
                PATHS
                    "${_ASCEND_ROOT}/include"
                    "${_ASCEND_ROOT}/pkg_inc"
                    "${_ASCEND_ROOT}/pkg_inc/runtime"
                    "${_ASCEND_ROOT}/aarch64-linux/pkg_inc"
                    "${_ASCEND_ROOT}/aarch64-linux/pkg_inc/runtime"
                    "/usr/local/Ascend/cann/aarch64-linux/pkg_inc"
                    "/usr/local/Ascend/cann/aarch64-linux/pkg_inc/runtime"
                NO_DEFAULT_PATH
            )
            if(_ASCEND_FFTS_INCLUDE_DIR)
                set(_ASCEND_FFTS_INCLUDE_DIRS "${_ASCEND_FFTS_INCLUDE_DIR}")
                get_filename_component(_ASCEND_FFTS_INCLUDE_PARENT
                    "${_ASCEND_FFTS_INCLUDE_DIR}" DIRECTORY)
                if(_ASCEND_FFTS_INCLUDE_DIR MATCHES "/runtime$"
                    AND EXISTS "${_ASCEND_FFTS_INCLUDE_PARENT}")
                    list(APPEND _ASCEND_FFTS_INCLUDE_DIRS "${_ASCEND_FFTS_INCLUDE_PARENT}")
                endif()

                set(_ASCEND_FFTS_EXTRA_INCLUDE_CANDIDATES
                    "${_ASCEND_ROOT}/pkg_inc"
                    "${_ASCEND_ROOT}/pkg_inc/toolchain"
                    "${_ASCEND_ROOT}/pkg_inc/profiling"
                    "${_ASCEND_ROOT}/aarch64-linux/pkg_inc"
                    "${_ASCEND_ROOT}/aarch64-linux/pkg_inc/toolchain"
                    "${_ASCEND_ROOT}/aarch64-linux/pkg_inc/profiling"
                    "/usr/local/Ascend/cann/aarch64-linux/pkg_inc"
                    "/usr/local/Ascend/cann/aarch64-linux/pkg_inc/toolchain"
                    "/usr/local/Ascend/cann/aarch64-linux/pkg_inc/profiling")
                foreach(_ASCEND_EXTRA_INCLUDE_DIR
                    ${_ASCEND_FFTS_EXTRA_INCLUDE_CANDIDATES})
                    if(EXISTS "${_ASCEND_EXTRA_INCLUDE_DIR}")
                        list(APPEND _ASCEND_FFTS_INCLUDE_DIRS "${_ASCEND_EXTRA_INCLUDE_DIR}")
                    endif()
                endforeach()
                find_path(_ASCEND_PROF_COMMON_INCLUDE_DIR
                    NAMES prof_common.h
                    PATHS ${_ASCEND_FFTS_EXTRA_INCLUDE_CANDIDATES}
                    NO_DEFAULT_PATH
                )
                if(_ASCEND_PROF_COMMON_INCLUDE_DIR)
                    list(APPEND _ASCEND_FFTS_INCLUDE_DIRS
                        "${_ASCEND_PROF_COMMON_INCLUDE_DIR}")
                endif()
                find_path(_ASCEND_PROF_API_INCLUDE_DIR
                    NAMES prof_api.h toolchain/prof_api.h profiling/prof_api.h
                    PATHS ${_ASCEND_FFTS_EXTRA_INCLUDE_CANDIDATES}
                    NO_DEFAULT_PATH
                )
                if(_ASCEND_PROF_API_INCLUDE_DIR)
                    list(APPEND _ASCEND_FFTS_INCLUDE_DIRS
                        "${_ASCEND_PROF_API_INCLUDE_DIR}")
                endif()
                list(REMOVE_DUPLICATES _ASCEND_FFTS_INCLUDE_DIRS)
            endif()

            message(STATUS "Found Ascend runtime: ${_ASCEND_ROOT}")
            message(STATUS "  Include: ${_ASCEND_INCLUDE_DIR}")
            message(STATUS "  Library: ${_ASCEND_LIBRARY}")
            if(_ASCEND_RUNTIME_LIBRARY AND _ASCEND_FFTS_INCLUDE_DIR)
                message(STATUS "  FFTS Includes: ${_ASCEND_FFTS_INCLUDE_DIRS}")
                message(STATUS "  Runtime Library: ${_ASCEND_RUNTIME_LIBRARY}")
            else()
                message(STATUS "  FFTS support: disabled (missing FFTS header or libruntime)")
            endif()

            # Create imported target
            add_library(Ascend::Runtime INTERFACE IMPORTED GLOBAL)
            target_include_directories(Ascend::Runtime INTERFACE "${_ASCEND_INCLUDE_DIR}")
            target_link_libraries(Ascend::Runtime INTERFACE "${_ASCEND_LIBRARY}")

            # Set global variables
            set(RUNTIME_BACKEND "Ascend" PARENT_SCOPE)
            set(ASCEND_FOUND TRUE PARENT_SCOPE)
            set(ASCEND_ROOT "${_ASCEND_ROOT}" PARENT_SCOPE)
            if(_ASCEND_RUNTIME_LIBRARY AND _ASCEND_FFTS_INCLUDE_DIR)
                set(HAVE_ASCEND_FFTS_RUNTIME TRUE PARENT_SCOPE)
                set(ASCEND_RUNTIME_LIBRARY "${_ASCEND_RUNTIME_LIBRARY}" PARENT_SCOPE)
                set(ASCEND_FFTS_INCLUDE_DIRS "${_ASCEND_FFTS_INCLUDE_DIRS}" PARENT_SCOPE)
            endif()

            return()
        endif()
    endif()

    # ============================================
    # Fallback to simulation mode
    # ============================================
    message(STATUS "No GPU runtime found, using CPU simulation mode")
endfunction()
