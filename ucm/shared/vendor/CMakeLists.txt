function(EnableDept)
    cmake_parse_arguments(DEPT "" "NAME;TAG" "GIT_URLS" ${ARGN})
    find_program(GIT_EXECUTABLE git)
    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "git not found!")
    endif()
    foreach(GIT_URL IN LISTS DEPT_GIT_URLS)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} ls-remote --heads "${GIT_URL}"
            RESULT_VARIABLE GIT_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
            TIMEOUT 15
        )
        if(GIT_RESULT EQUAL 0)
            set(VALID_GIT_URL ${GIT_URL})
            break()
        endif()
    endforeach()
    if(NOT VALID_GIT_URL)
        message(FATAL_ERROR "all urls for ${DEPT_NAME} are not reachable!")
    endif()
    message(STATUS "Fetching ${DEPT_NAME}(${DEPT_TAG}) from ${VALID_GIT_URL}")
    FetchContent_Declare(${DEPT_NAME} GIT_REPOSITORY ${VALID_GIT_URL} GIT_TAG ${DEPT_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEPT_NAME})
endfunction()

if(DOWNLOAD_DEPENDENCE)
    include(FetchContent)
    EnableDept(
        NAME fmt
        TAG 11.2.0
        GIT_URLS
            https://github.com/fmtlib/fmt.git
            https://gitcode.com/GitHub_Trending/fm/fmt.git
    )
    EnableDept(
        NAME spdlog
        TAG v1.15.3
        GIT_URLS
            https://github.com/gabime/spdlog.git
            https://gitcode.com/GitHub_Trending/sp/spdlog.git
    )
    EnableDept(
        NAME pybind11
        TAG v3.0.1
        GIT_URLS
            https://github.com/pybind/pybind11.git
            https://gitcode.com/GitHub_Trending/py/pybind11.git
    )
else()
    add_subdirectory(fmt)
    add_subdirectory(spdlog)
    add_subdirectory(pybind11)
endif()
