set(INSTALL_GTEST OFF CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_GOOGLETEST_NAME googletest)
    set(DEP_GOOGLETEST_TAG v1.15.2)
    set(DEP_GOOGLETEST_GIT_URLS
        https://github.com/google/googletest.git
        https://gitcode.com/GitHub_Trending/go/googletest.git
    )
    include(helper.cmake)
    find_reachable_git_url(REACHABLE_URL DEP_GOOGLETEST_GIT_URLS)
    include(FetchContent)
    message(STATUS "Fetching ${DEP_GOOGLETEST_NAME}(${DEP_GOOGLETEST_TAG}) from ${REACHABLE_URL}")
    FetchContent_Declare(${DEP_GOOGLETEST_NAME}
        GIT_REPOSITORY ${REACHABLE_URL}
        GIT_TAG ${DEP_GOOGLETEST_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(${DEP_GOOGLETEST_NAME})
else()
    add_subdirectory(googletest)
endif()
