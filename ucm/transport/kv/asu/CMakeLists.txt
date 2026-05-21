file(GLOB ASU_TRANSPORT_SOURCES CONFIGURE_DEPENDS trans/src/*.cpp)
add_library(asu_transport SHARED ${ASU_TRANSPORT_SOURCES})
target_include_directories(asu_transport
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_link_libraries(asu_transport PUBLIC pthread)

file(GLOB ASU_CLIENT_SOURCES CONFIGURE_DEPENDS client/src/*.cpp)
add_library(asu_client SHARED ${ASU_CLIENT_SOURCES})
target_include_directories(asu_client
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/client/include
)
target_include_directories(asu_client
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/client/src
        ${UCM_ROOT_DIR}/ucm/shared/infra
)
target_link_libraries(asu_client PUBLIC asu_transport pthread)

if(BUILD_UNIT_TESTS)
    include(GoogleTest)
    file(GLOB_RECURSE ASU_TEST_SOURCE_FILES CONFIGURE_DEPENDS test/case/*.cc)
    add_executable(asu.test ${ASU_TEST_SOURCE_FILES})
    target_include_directories(asu.test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/client/src
        ${CMAKE_CURRENT_SOURCE_DIR}/trans/src
    )
    target_link_libraries(asu.test PRIVATE
        asu_client
        gtest_main gtest
    )
    gtest_discover_tests(asu.test)
endif()
