// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __LOG_H__
#define __LOG_H__
#include <stdio.h>

// 1. 定义日志级别常量
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_NONE  // 关闭所有日志
} LogLevel;

extern LogLevel g_current_log_level;

#define LOG_PRINT(level, level_str, color, format, ...)                                     \
    do {                                                                                    \
        if (level >= g_current_log_level) {                                                 \
            printf(color "[%s] [%s:%d] " format "\033[0m\n", level_str, __FILE__, __LINE__, \
                   ##__VA_ARGS__);                                                          \
        }                                                                                   \
    } while (0)

// 4. 用户调用的 API
#define LOG_D(fmt, ...) LOG_PRINT(LOG_LEVEL_DEBUG, "DEBUG", "\033[1;30m", fmt, ##__VA_ARGS__)
#define LOG_I(fmt, ...) LOG_PRINT(LOG_LEVEL_INFO, "INFO ", "\033[0;32m", fmt, ##__VA_ARGS__)
#define LOG_W(fmt, ...) LOG_PRINT(LOG_LEVEL_WARN, "WARN ", "\033[0;33m", fmt, ##__VA_ARGS__)
#define LOG_E(fmt, ...) LOG_PRINT(LOG_LEVEL_ERROR, "ERROR", "\033[0;31m", fmt, ##__VA_ARGS__)

// 5. 动态设置级别的函数声明
void set_log_level(LogLevel level);

#endif