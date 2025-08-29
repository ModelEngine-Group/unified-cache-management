add_compile_definitions(UC_VAR_PROJECT_NAME="${PROJECT_NAME}")
add_compile_definitions(UC_VAR_PROJECT_VERSION="${PROJECT_VERSION}")
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE UC_VAR_GIT_COMMIT_ID
                OUTPUT_STRIP_TRAILING_WHITESPACE)
add_compile_definitions(UC_VAR_GIT_COMMIT_ID="${UC_VAR_GIT_COMMIT_ID}")
add_compile_definitions(UC_VAR_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
