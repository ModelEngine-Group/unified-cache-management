add_library(storetask STATIC task_manager.cc)
target_include_directories(storetask PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(storetask PUBLIC storeinfra)
