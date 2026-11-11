// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/log.h"

// 初始化默认级别为 INFO
LogLevel g_current_log_level = LOG_LEVEL_INFO;

// 提供一个函数来动态修改
void set_log_level(LogLevel level)
{
    g_current_log_level = level;
    printf(">> Log level changed to %d\n", level);
}