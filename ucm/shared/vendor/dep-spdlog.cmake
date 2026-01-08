set(SPDLOG_INSTALL OFF CACHE INTERNAL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(SPDLOG_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_SPDLOG_NAME spdlog)
    set(DEP_SPDLOG_TAG v1.15.3)
    set(DEP_SPDLOG_GIT_URLS
        https://github.com/gabime/spdlog.git
        https://gitcode.com/GitHub_Trending/sp/spdlog.git
    )
    include(helper.cmake)
    find_reachable_git_url(REACHABLE_URL DEP_SPDLOG_GIT_URLS)
    include(FetchContent)
    message(STATUS "Fetching ${DEP_SPDLOG_NAME}(${DEP_SPDLOG_TAG}) from ${REACHABLE_URL}")
    FetchContent_Declare(${DEP_SPDLOG_NAME} GIT_REPOSITORY ${REACHABLE_URL} GIT_TAG ${DEP_SPDLOG_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEP_SPDLOG_NAME})
else()
    add_subdirectory(spdlog)
endif()
