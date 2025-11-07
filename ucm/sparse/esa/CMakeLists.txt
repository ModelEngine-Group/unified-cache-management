set(NUMA_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/numa_install)
FetchContent_Declare(
    numactl
    URL https://github.com/numactl/numactl/releases/download/v2.0.16/numactl-2.0.16.tar.gz
    TLS_VERIFY OFF
)
FetchContent_MakeAvailable(numactl)
if(NOT EXISTS "${NUMA_INSTALL_DIR}/lib/libnuma.so")
    message(STATUS "Configuring numactl...")
    execute_process(
        COMMAND ./configure --prefix=${NUMA_INSTALL_DIR}
        WORKING_DIRECTORY ${numactl_SOURCE_DIR}
        RESULT_VARIABLE numa_configure_result
        OUTPUT_VARIABLE numa_configure_output
        ERROR_VARIABLE numa_configure_error
    )
    if(NOT numa_configure_result EQUAL 0)
        message(FATAL_ERROR "Failed to configure numactl. \n"
                            "Result: ${numa_configure_result}\n"
                            "STDOUT: ${numa_configure_output}\n"
                            "STDERR: ${numa_configure_error}\n")
    endif()

    message(STATUS "Building and installing numactl...")
    execute_process(
        COMMAND make install -j8
        WORKING_DIRECTORY ${numactl_SOURCE_DIR}
        RESULT_VARIABLE numa_install_result
        OUTPUT_VARIABLE numa_install_output
        ERROR_VARIABLE numa_install_error
    )
    if(NOT numa_install_result EQUAL 0)
        message(FATAL_ERROR "Failed to build and install numactl. \n"
                            "Result: ${numa_install_result}\n"
                            "STDOUT: ${numa_install_output}\n"
                            "STDERR: ${numa_install_error}\n")
    endif()
else()
    message(STATUS "Found already built libnuma. Skipping build.")
endif()

add_subdirectory(retrieval)
