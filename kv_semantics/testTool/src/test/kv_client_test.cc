/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "test/kv_client_test.h"

#include <string>

#include "log/logger_manager.h"
#include "worker/scheduler.h"

bool RunKvClientDemo()
{
    std::string err;
    Scheduler scheduler;
    if (!scheduler.Init("conf/kv_client.conf", err)) {
        LOG_ERROR("kv client demo 初始化失败: {}", err);
        return false;
    }
    LOG_INFO("kv client demo：初始化、建链与内存注册成功");

    const bool passed = scheduler.Run(err);
    if (!passed) {
        LOG_ERROR("kv client demo 功能验证失败: {}", err);
    }

    scheduler.Shutdown();
    return passed;
}
