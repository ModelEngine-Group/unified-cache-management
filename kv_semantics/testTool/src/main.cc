/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <pthread.h>
#include <string>
#include <thread>
#include "log/logger_manager.h"
#include "request/request_manager.h"
#include "scheduler.h"
// #include "test/log_test.h"  // 日志模块自测暂时停用，恢复时随下方调用段一并取消注释
#include "utils/config_util.h"

/** SIGINT 到达标志（信号处理器与主线程共享；lock-free 原子 store 为 async-signal-safe） */
static std::atomic<bool> gInterrupted{false};

/** SIGINT 处理器：仅置标志位，不做任何非 async-signal-safe 操作 */
static void OnSigint(int) { gInterrupted.store(true); }

/**
 * 统一收尾：按初始化逆序回收各组件，main 的全部退出路径（正常、各失败分支与异常兜底）
 * 统一经此出口，不逐点散落 Shutdown 调用；未初始化的组件在其 Shutdown 内安全跳过，可重复调用
 * @param scheduler 调度器（生命周期由 main 栈保证）
 */
static void CloseAll(Scheduler &scheduler)
{
    scheduler.Shutdown();
    LoggerManager::Shutdown();
}

/**
 * 程序入口：加载配置、初始化日志（全量日志 + 请求明细日志），先完成调度器
 * 初始化（kv client 建链 + 值缓冲池一次性分配注册，与 input 无关，参考 kv_test：region
 * 注册仅此一次），之后按 onlyconnect 分流——on 建链验证完成后挂起等待 Ctrl+C 再退出
 * （不解析 input）；off 为执行模式，按顺序平铺：解析 input 流量文件驱动测试（圈数/时长由
 * runtime/loop 控制）、输出指标与完成标记；配置 getter 抛出的异常（缺 key/坏值）统一
 * catch 报错退出；全部路径经 CloseAll 统一收尾
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

    // 调度器生命周期归 main 栈；异常兜底分支同样要能对它统一收尾，声明在 try 之外
    Scheduler scheduler;
    std::string err;
    try {
        // ---- 配置与日志初始化（此时无组件资源；LoggerManager::Init 失败时内部已自清理）----
        ConfigUtil &config = ConfigUtil::GetInstance();
        if (!config.Load(configPath, err)) {
            std::cerr << "[FAIL] config load failed: " << err << std::endl;
            return 1;
        }
        if (!LoggerManager::Init(config, err)) {
            std::cerr << "[FAIL] logger init failed: " << err << std::endl;
            return 1;
        }
        // 启动标记行（与完成标记对称）：用 warn 级别保证 perf=on（主日志级别强制 warn、
        // warn 及 error 均输出）时同样可见
        LOG_WARN("======Test Starting=======");
        LOG_INFO("testTool v{} starting, test config: {}, kv client config: {}, input: {}", "0.1.0",
                 configPath, kvClientConfPath, inputPath.empty() ? std::string("-") : inputPath);
        config.Print();

        // ---- 调度器初始化：建链 + 值缓冲池分配注册（两模式共用，region 注册仅此一次；
        //      内存类型按 buffertype：host 主机内存 / device 设备内存）----
        if (!scheduler.initializeAndRegister(kvClientConfPath, config, err)) {
            LOG_ERROR("scheduler init failed: {}", err);
            CloseAll(scheduler);
            return 1;
        }

        // ---- 只建链模式：建链与 region 注册验证完成后挂起等待 Ctrl+C（不解析
        // input、不直接退出）----
        if (config.GetOnlyConnect()) {
            LOG_INFO("onlyconnect=on: connection established, waiting for Ctrl+C to exit...");
            std::signal(SIGINT, OnSigint);
            // 100ms 轮询等待 SIGINT（信号处理器仅置原子标志，无锁无系统调用，async-signal-safe）
            while (!gInterrupted.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            LOG_INFO("onlyconnect=on: Ctrl+C received, exiting");
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

        // 按任务流执行（重复执行语义，false 仅当工具链失败）
        const bool runOk =
            scheduler.Run(requestManager, testmode, config.GetRuntime(), config.GetLoop(), err);
        if (!runOk) { LOG_ERROR("test flow failed: {}", err); }

        // 完成标记行（仅成功时输出；与启动标记对称）：warn 级别保证 perf=on 时同样可见
        if (runOk) { LOG_WARN("======Test Completed Successfully======="); }

        // 临时：日志模块功能自测，全部正式功能开发完成后随 src/test/
        // 目录整体移除本段（仅正式流程成功时运行） ——暂时停用（2026-11-17 指令：暂时注释 log
        // 模块的验证），恢复时取消下行注释并删除占位行 const bool logTestOk = runOk ?
        // RunLogTest(config) : true;
        const bool logTestOk = true;  // 日志自测停用期间的占位恒真值

        // ---- 统一收尾：回收调度器（注销 region、关闭 kv client）与全部日志资源 ----
        CloseAll(scheduler);
        return (runOk && logTestOk) ? 0 : 1;
    } catch (const std::exception &e) {
        // 配置 getter 安全访问抛出的缺 key / 坏值等异常统一在此兜底：报错退出前对已初始化组件收尾
        std::cerr << "[FAIL] " << e.what() << std::endl;
        CloseAll(scheduler);
        return 1;
    }
}
