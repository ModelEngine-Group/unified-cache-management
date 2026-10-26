set(FMT_INSTALL OFF CACHE INTERNAL "" FORCE)
set(FMT_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(FMT_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_FMT_NAME fmt)
    set(DEP_FMT_TAG 11.2.0)
    if(NOT UCM_DEP_FMT_URL)
        set(DEP_FMT_GIT_URLS
            https://github.com/fmtlib/fmt.git
            https://gitcode.com/GitHub_Trending/fm/fmt.git
        )
        include(helper.cmake)
        find_reachable_git_url(REACHABLE_URL DEP_FMT_GIT_URLS)
        set(UCM_DEP_FMT_URL ${REACHABLE_URL} CACHE INTERNAL "Resolved git URL for fmt")
    endif()
    include(FetchContent)
    message(STATUS "Fetching ${DEP_FMT_NAME}(${DEP_FMT_TAG}) from ${UCM_DEP_FMT_URL}")
    FetchContent_Declare(${DEP_FMT_NAME} GIT_REPOSITORY ${UCM_DEP_FMT_URL} GIT_TAG ${DEP_FMT_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEP_FMT_NAME})
else()
    add_subdirectory(fmt)
endif()
