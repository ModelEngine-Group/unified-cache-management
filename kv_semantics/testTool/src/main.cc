/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include <pthread.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "log/logger_manager.h"
#include "request/request_manager.h"
#include "scheduler.h"
#include "test/log_test.h"
#include "utils/config_util.h"

/**
 * 统一收尾：按初始化逆序回收各组件，main 的全部退出路径（正常与各异常分支）统一经此
 * 出口，不逐点散落 Shutdown 调用；未初始化的组件在其 Shutdown 内安全跳过，可重复调用
 * @param scheduler 调度器（生命周期由 main 栈保证）
 */
static void CloseAll(Scheduler &scheduler)
{
    scheduler.Shutdown();
    LoggerManager::Shutdown();
}

/**
 * 程序入口：加载配置、初始化日志（全量日志 + 非性能模式的 get 读回日志），先完成调度器
 * 初始化（kv client 建链 + 值缓冲池一次性分配注册，与 input 无关，参考 kv_test：region
 * 注册仅此一次），之后按 onlyconnect 分流——on 建链验证完成后即退出（不解析 input）；
 * off 为执行模式，按顺序平铺：解析 input 流量文件驱动测试、输出指标；全部路径经
 * CloseAll 统一收尾
 * @param argv[1] 测试配置文件路径（test_config.conf）
 * @param argv[2] kv client 配置文件路径（kv_client.conf）
 * @param argv[3] 流量模型输入文件路径（input.txt；仅执行模式必传，建链模式可省略）
 */
int main(int argc, char *argv[])
{
    // 主线程命名（调试可见：gdb info threads、top -H、/proc/<pid>/task/<tid>/comm；上限 15 字符）
    pthread_setname_np(pthread_self(), "main");

    // ---- 参数解析 ----
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <test config path> <kv client config path> [input file path]" << std::endl;
        return 1;
    }
    const std::string configPath = argv[1];
    const std::string kvClientConfPath = argv[2];
    const std::string inputPath = argc >= 4 ? argv[3] : "";

    // ---- 配置与日志初始化（此时无组件资源；LoggerManager::Init 失败时内部已自清理）----
    std::string err;
    ConfigUtil &config = ConfigUtil::GetInstance();
    if (!config.Load(configPath, err)) {
        std::cerr << "[FAIL] config load failed: " << err << std::endl;
        return 1;
    }
    if (!LoggerManager::Init(config, err)) {
        std::cerr << "[FAIL] logger init failed: " << err << std::endl;
        return 1;
    }
    LOG_INFO("testTool v{} starting, test config: {}, kv client config: {}, input: {}", "0.1.0",
             configPath, kvClientConfPath, inputPath.empty() ? std::string("-") : inputPath);
    config.Print();

    // ---- 调度器初始化：建链 + 值缓冲池分配注册（两模式共用，region 注册仅此一次；
    //      内存类型按 buffertype：host 主机内存 / device 设备内存）----
    Scheduler scheduler;
    if (!scheduler.initializeAndRegister(kvClientConfPath, config, err)) {
        LOG_ERROR("scheduler init failed: {}", err);
        CloseAll(scheduler);
        return 1;
    }

    // ---- 只建链模式：建链与 region 注册验证完成后即退出，不解析 input ----
    if (config.GetOnlyConnect()) {
        LOG_INFO("onlyconnect=on: connection established, exit in connect-only mode, IO test skipped");
        CloseAll(scheduler);
        return 0;
    }

    // ---- 执行模式（onlyconnect=off）：input 为唯一驱动源，缺失即退出 ----
    if (inputPath.empty()) {
        std::cerr << "[FAIL] input file path required when onlyconnect=off" << std::endl;
        CloseAll(scheduler);
        return 1;
    }

    // testmode 校验：值生成规则只定义了 1（系统生成）与 2（行尾 value），其他值直接报错可感知
    const int testmode = config.GetTestMode();
    if (testmode != 1 && testmode != 2) {
        std::cerr << "[FAIL] invalid testmode " << testmode << ", expect 1 or 2" << std::endl;
        CloseAll(scheduler);
        return 1;
    }

    // 解析 input 流量文件为任务序列
    RequestManager requestManager;
    if (!requestManager.Load(inputPath, err)) {
        LOG_ERROR("input load failed: {}", err);
        CloseAll(scheduler);
        return 1;
    }

    // 按任务流执行（持续执行语义，false 仅当工具链失败）
    const bool runOk = scheduler.Run(requestManager, testmode, err);
    if (!runOk) {
        LOG_ERROR("test flow failed: {}", err);
    }

    // 临时：日志模块功能自测，全部正式功能开发完成后随 src/test/ 目录整体移除本段（仅正式流程成功时运行）
    const bool logTestOk = runOk ? RunLogTest(config) : true;

    // ---- 统一收尾：回收调度器（注销 region、关闭 kv client）与全部日志资源 ----
    CloseAll(scheduler);
    return (runOk && logTestOk) ? 0 : 1;
}
