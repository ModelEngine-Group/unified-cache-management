/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "worker/client.h"
#include "worker/executor.h"
#include "worker/request.h"

/**
 * 调度器：统一管理 kv client 与全部执行器的生命周期——
 * 初始化（kv client 建链 + 请求内存注册）、驱动执行器执行增删查验证、
 * 秒级周期汇总各执行器埋点指标并输出、停止时注销请求内存并关闭 client。
 * 当前为单执行器同步执行形态，多执行器并发（线程数取自配置）为后续扩展。
 */
class Scheduler {
public:
    Scheduler();
    /** 兜底析构：未显式 Shutdown 时停止指标线程并释放全部资源 */
    ~Scheduler();

    /** 初始化：kv client 建链、组装请求数据并完成内存注册（当前为内置占位请求，requestManager 落地后由 input.txt 提供） */
    bool Init(const std::string &confPath, std::string &err);

    /** 启动指标汇总线程并驱动全部执行器执行一轮验证；失败时 err 说明首次失败步骤 */
    bool Run(std::string &err);

    /** 停止指标线程、注销请求内存、关闭 kv client 并释放执行器；可重复调用 */
    void Shutdown();

private:
    /** 指标汇总线程主循环：每秒采样全部执行器累计指标，输出窗口速率 */
    void MetricsLoop();

    /** 组装占位请求数据（全新 key + 固定 pattern 值；requestManager 落地后由其取代） */
    void BuildPlaceholderRequests();

    /** 采集全部执行器当前指标快照 */
    std::vector<ExecutorMetrics> SnapshotExecutors() const;

    /** 输出一个汇总窗口的指标（窗口增量速率：IOPS / 流量 / TTFT / 错误率） */
    void OutputMetricsWindow(const std::vector<ExecutorMetrics> &last,
                             const std::vector<ExecutorMetrics> &current) const;

    /** 输出运行结束时的指标总计（全部执行器累计值汇总） */
    void OutputMetricsTotal() const;

    /** kv client 封装（生命周期归本类） */
    std::unique_ptr<Client> client_;
    /** 全部执行器（当前单个，多线程扩展时按配置创建多个） */
    std::vector<std::unique_ptr<Executor>> executors_;
    /** 请求数据包（含已注册的值 buffer） */
    Request request_;
    /** 指标汇总线程 */
    std::thread metricsThread_;
    /** 指标线程运行标志（false 时线程退出） */
    std::atomic<bool> metricsRunning_{false};
};
