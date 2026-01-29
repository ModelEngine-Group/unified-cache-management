set(SKIP_INSTALL_ALL ON CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)

if(DOWNLOAD_DEPENDENCE)
    set(DEP_ZLIB_NAME zlib)
    set(DEP_ZLIB_TAG v1.3.1)
    set(DEP_ZLIB_GIT_URLS
        https://github.com/madler/zlib.git
        https://gitcode.com/GitHub_Trending/sp/ZLIB.git
    )
    include(helper.cmake)
    find_reachable_git_url(REACHABLE_URL DEP_ZLIB_GIT_URLS)
    include(FetchContent)
    message(STATUS "Fetching ${DEP_ZLIB_NAME}(${DEP_ZLIB_TAG}) from ${REACHABLE_URL}")
    FetchContent_Declare(${DEP_ZLIB_NAME} GIT_REPOSITORY ${REACHABLE_URL} GIT_TAG ${DEP_ZLIB_TAG} GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(${DEP_ZLIB_NAME})

    target_compile_options(zlibstatic PRIVATE -fPIC)
else()
    add_subdirectory(zlib)
    target_compile_options(zlibstatic PRIVATE -fPIC)
endif()