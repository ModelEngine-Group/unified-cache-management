/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "test/kv_client_test.h"

#include <string>

#include "log/logger_manager.h"
#include "worker/client.h"
#include "worker/executor.h"

bool RunKvClientDemo()
{
    std::string err;
    Client client;
    if (!client.Init("conf/kv_client.conf", err)) {
        LOG_ERROR("kv client demo 初始化失败: {}", err);
        return false;
    }
    LOG_INFO("kv client demo：初始化与建链成功");

    Executor executor;
    executor.SetClient(&client);
    const bool passed = executor.Run(err);
    if (!passed) {
        LOG_ERROR("kv client demo 功能验证失败: {}", err);
    }

    client.Shutdown();
    return passed;
}
