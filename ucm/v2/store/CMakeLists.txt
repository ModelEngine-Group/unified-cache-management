add_library(ucm_store INTERFACE)
target_include_directories(ucm_store INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set(UCM_STORE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(ucm_add_store NAME)
    file(GLOB_RECURSE UCM_STORE_CC_SOURCES CONFIGURE_DEPENDS "*.h" "*.cc")
    add_library(ucm_store_${NAME}_core OBJECT ${UCM_STORE_CC_SOURCES})
    target_include_directories(ucm_store_${NAME}_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(ucm_store_${NAME}_core PUBLIC ucm_build_options ucm_common ucm_store)

    set(UCM_STORE_FACTORY_SOURCES "")
    list(LENGTH ARGN UCM_STORE_ARGN_COUNT)
    if(UCM_STORE_ARGN_COUNT EQUAL 2)
        list(GET ARGN 0 UCM_STORE_HEADER)
        list(GET ARGN 1 UCM_STORE_CLASS)
        configure_file(${UCM_STORE_DIR}/factory.cc.in ${NAME}_factory.cc @ONLY)
        set(UCM_STORE_FACTORY_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_factory.cc)
    endif()
    add_library(ucm_store_${NAME} MODULE $<TARGET_OBJECTS:ucm_store_${NAME}_core> ${UCM_STORE_FACTORY_SOURCES})
    target_link_libraries(ucm_store_${NAME} PRIVATE ucm_store_${NAME}_core)

    set_property(GLOBAL APPEND PROPERTY UCM_BUILTIN_STORES "${NAME}=$<TARGET_FILE:ucm_store_${NAME}>")
endfunction()

add_subdirectory(cache)
add_subdirectory(posix)
