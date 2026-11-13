/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include <iostream>
#include <string>

#include "log/logger_manager.h"
#include "test/kv_client_test.h"
#include "test/log_test.h"
#include "utils/config_util.h"

/** 程序入口：加载配置、初始化日志，正式功能（request / worker 模块）后续在此扩展 */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <配置文件路径>" << std::endl;
        return 1;
    }
    std::string configPath = argv[1];

    std::string err;
    if (!ConfigUtil::Load(configPath, err)) {
        std::cerr << "[失败] 配置加载失败: " << err << std::endl;
        return 1;
    }

    if (!LoggerManager::Init(err)) {
        std::cerr << "[失败] 日志初始化失败: " << err << std::endl;
        return 1;
    }
    LOG_INFO("testTool v{} 启动，配置文件: {}", "0.1.0", configPath);
    ConfigUtil::Print();

    // 临时：kv client 建链/注册/读写 demo（FAKE provider），正式功能由 worker 模块接管后移除本段
    // 顺序要求：必须放在 RunLogTest 之前——其结尾会调用 LoggerManager::Shutdown，放在其后的日志会被静默丢弃
    if (!RunKvClientDemo()) {
        LoggerManager::Shutdown();
        return 1;
    }

    // 临时：日志模块功能自测，正式功能开发后移除本段
    if (!RunLogTest()) {
        LoggerManager::Shutdown();
        return 1;
    }

    // TODO: 正式功能（request / worker 模块）启动流程在此扩展

    LoggerManager::Shutdown();
    return 0;
}

/**
    * 测试的时候：1、 提供kv client so
    *  2. 头文件
    * 3. 配置文件
    * 4. 
 **/ 
