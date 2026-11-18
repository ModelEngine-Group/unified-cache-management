file(GLOB_RECURSE UCM_STORE_DETAIL_SOURCE "*.*")
add_library(storedetail OBJECT ${UCM_STORE_DETAIL_SOURCE})
target_include_directories(storedetail PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Header-only OBJECT library: pin the language so CMake can generate the target.
set_target_properties(storedetail PROPERTIES LINKER_LANGUAGE CXX)
