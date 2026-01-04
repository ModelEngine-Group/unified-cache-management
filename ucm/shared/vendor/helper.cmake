function(find_reachable_git_url OUT_REACHABLE_URL IN_URL_LIST)
    find_program(GIT_EXECUTABLE git)
    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "git not found!")
    endif()

    if(DEFINED ${IN_URL_LIST})
        set(URL_LIST ${${IN_URL_LIST}})
    else()
        set(URL_LIST ${ARGN})
    endif()

    foreach(GIT_URL IN LISTS URL_LIST)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} ls-remote --heads "${GIT_URL}"
            RESULT_VARIABLE GIT_RESULT
            OUTPUT_QUIET ERROR_QUIET
            TIMEOUT 5
        )
        if(GIT_RESULT EQUAL 0)
            set(${OUT_REACHABLE_URL} ${GIT_URL} PARENT_SCOPE)
            return()
        endif()
    endforeach()

    message(FATAL_ERROR "All git URLs are not reachable!")
endfunction()
