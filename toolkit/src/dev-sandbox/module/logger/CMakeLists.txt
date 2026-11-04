add_library(logger STATIC logger.h logger.cc)
target_link_libraries(logger PUBLIC fmt::fmt)

add_executable(logger_example logger_example.cc)
target_link_libraries(logger_example PRIVATE logger)
