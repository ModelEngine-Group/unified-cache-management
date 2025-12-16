file(GLOB_RECURSE UCM_STORE_DETAIL_SOURCE "*.*")
add_library(storedetail OBJECT ${UCM_STORE_DETAIL_SOURCE})
target_include_directories(storedetail PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
