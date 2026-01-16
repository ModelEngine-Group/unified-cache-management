set(ZLIB_INSTALL OFF CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)

# Save current BUILD_SHARED_LIBS value
if(DEFINED BUILD_SHARED_LIBS)
    set(_ZLIB_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()

# Force zlib to build as static library (not shared .so) so it gets embedded
# This prevents creating a separate libz.so file - zlib will be statically linked
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "" FORCE)

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
else()
    add_subdirectory(ZLIB)
endif()

# Restore BUILD_SHARED_LIBS if it was set (to avoid affecting other dependencies)
if(DEFINED _ZLIB_SAVED_BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${_ZLIB_SAVED_BUILD_SHARED_LIBS} CACHE INTERNAL "" FORCE)
    unset(_ZLIB_SAVED_BUILD_SHARED_LIBS)
endif()
