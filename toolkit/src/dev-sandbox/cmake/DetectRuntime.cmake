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
            message(STATUS "Found Ascend runtime: ${_ASCEND_ROOT}")
            message(STATUS "  Include: ${_ASCEND_INCLUDE_DIR}")
            message(STATUS "  Library: ${_ASCEND_LIBRARY}")

            # Create imported target
            add_library(Ascend::Runtime INTERFACE IMPORTED GLOBAL)
            target_include_directories(Ascend::Runtime INTERFACE "${_ASCEND_INCLUDE_DIR}")
            target_link_libraries(Ascend::Runtime INTERFACE "${_ASCEND_LIBRARY}")

            # Set global variables
            set(RUNTIME_BACKEND "Ascend" PARENT_SCOPE)
            set(ASCEND_FOUND TRUE PARENT_SCOPE)
            set(ASCEND_ROOT "${_ASCEND_ROOT}" PARENT_SCOPE)

            return()
        endif()
    endif()

    # ============================================
    # Fallback to simulation mode
    # ============================================
    message(STATUS "No GPU runtime found, using CPU simulation mode")
endfunction()
