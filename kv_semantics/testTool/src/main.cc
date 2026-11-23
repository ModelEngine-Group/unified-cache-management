/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

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
 * 注册仅此一次），之后按 onlyconnect 分流——on 建链验证完成后挂起（autodisconnect>0
 * 时计满 n 秒自动断链但不退出，继续挂起等 Ctrl+C 供观察断链后状态；均不解析 input）；off
 * 为执行模式， 按顺序平铺：解析 input 流量文件驱动测试（圈数/时长由 runtime/loop
 * 控制）、输出指标与完成标记；配置 getter 抛出的异常（缺 key/坏值）统一 catch 报错退出；全部路径经
 * CloseAll 统一收尾
 * @param argv[1] 测试配置文件路径（test_config.conf）
 * @param argv[2] kv client 配置文件路径（kv_client.conf）
 * @param argv[3] 流量模型输入文件路径（可缺省，缺省取启动目录下 input.txt——相对
 * 当前工作目录查找，与 conf 路径同款语义；建链模式不使用）
 */
int main(int argc, char *argv[])
{
    // 主线程命名（调试可见：gdb info threads、top -H、/proc/<pid>/task/<tid>/comm；上限 15 字符）
    pthread_setname_np(pthread_self(), "test_main");

    // ---- 参数解析 ----
    if (argc < 3) {
        std::cerr
            << "usage: " << argv[0]
            << " <test config path> <kv client config path> [input file path, default input.txt]"
            << std::endl;
        return 1;
    }
    const std::string configPath = argv[1];
    const std::string kvClientConfPath = argv[2];
    // input 可缺省：缺省取启动目录下 input.txt（找不到文件时由 Load 报错可感知）
    const std::string inputPath = argc >= 4 ? argv[3] : "input.txt";

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
        // 启动标记行（与完成标记对称）：warn 级别仅作醒目标记（perf 不再强制主日志
        // 级别，可见性随 log.level 配置：log.level=err 时此行不输出，属配置预期）
        LOG_WARN("======Test Starting=======");
        LOG_INFO("testTool v{} starting, test config: {}, kv client config: {}, input: {}", "0.1.0",
                 configPath, kvClientConfPath, inputPath);
        config.Print();

        // ---- 调度器初始化：建链 + 值缓冲池分配注册（两模式共用，region 注册仅此一次；
        //      内存类型按 buffertype：cpu 主机内存 / gpu CUDA 显存 / npu ACL NPU 内存）----
        if (!scheduler.initializeAndRegister(kvClientConfPath, config, err)) {
            LOG_ERROR("scheduler init failed: {}", err);
            CloseAll(scheduler);
            return 1;
        }

        // ---- 只建链模式：建链与 region 注册验证完成后挂起；autodisconnect>0 时计时
        // n 秒自动断链，断链后不退出——继续挂起等 Ctrl+C（供观察断链后两侧状态）----
        if (config.GetOnlyConnect()) {
            const int autoDisconnectS = config.GetAutoDisconnect();
            if (autoDisconnectS > 0) {
                LOG_INFO(
                    "onlyconnect=on: connection established, waiting for Ctrl+C or auto "
                    "disconnect after {}s...",
                    autoDisconnectS);
            } else {
                LOG_INFO("onlyconnect=on: connection established, waiting for Ctrl+C to exit...");
            }
            std::signal(SIGINT, OnSigint);
            // 100ms 轮询等待 SIGINT / 计满 n 秒（信号处理器仅置原子标志，无锁无系统调用，
            // async-signal-safe；断开时延精度 0.1s，测试工具场景足够）
            int waitedMs = 0;
            while (!gInterrupted.load()) {
                if (autoDisconnectS > 0 && waitedMs >= autoDisconnectS * 1000) { break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                waitedMs += 100;
            }
            bool disconnected = false;
            if (!gInterrupted.load() && autoDisconnectS > 0) {
                // 计时到：仅断链（kv client Shutdown：发送断链协议帧、销毁 QP），日志保持
                // 存活；断链后不退出，继续挂起等 Ctrl+C——供观察断链后两侧状态
                // （对端 store 的链路感知、本进程残留、日志尾部等）
                LOG_INFO(
                    "onlyconnect=on: auto disconnect after {}s, disconnected (waiting for "
                    "Ctrl+C to observe post-disconnect state...)",
                    autoDisconnectS);
                scheduler.Shutdown();
                disconnected = true;
            } else {
                LOG_INFO("onlyconnect=on: Ctrl+C received, exiting");
            }
            // 断链后观察期等待 SIGINT 再退出；提前 Ctrl+C 时标志已置位直接通过
            // （Scheduler::Shutdown 幂等，CloseAll 内重复调用安全）
            while (!gInterrupted.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (disconnected) { LOG_INFO("onlyconnect=on: Ctrl+C received, exiting"); }
            CloseAll(scheduler);
            return 0;
        }

        // ---- 执行模式（onlyconnect=off）：input 为唯一驱动源（缺失时已在参数解析
        // 阶段补默认值 input.txt，此处不再校验空路径；文件不存在由 Load 报错退出）----

        // autodisconnect 仅在建链挂起（onlyconnect=on）场景生效，执行模式配了打告警防静默误配
        if (config.GetAutoDisconnect() > 0) {
            LOG_WARN(
                "autodisconnect={} configured but onlyconnect=off, ignored "
                "(auto disconnect only works with onlyconnect=on)",
                config.GetAutoDisconnect());
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

        // 完成标记行（仅成功时输出；与启动标记对称）：warn 级别仅作醒目标记，可见性随 log.level
        if (runOk) { LOG_WARN("======Test Completed Successfully======="); }

        // ---- 统一收尾：回收调度器（注销 region、关闭 kv client）与全部日志资源 ----
        CloseAll(scheduler);
        return runOk ? 0 : 1;
    } catch (const std::exception &e) {
        // 配置 getter 安全访问抛出的缺 key / 坏值等异常统一在此兜底：报错退出前对已初始化组件收尾
        std::cerr << "[FAIL] " << e.what() << std::endl;
        CloseAll(scheduler);
        return 1;
    }
}
