/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kv_client.h"

class Client;
struct Request;

/** 执行器埋点指标（自创建起的累计值，单调递增；Scheduler 周期采样求窗口速率） */
struct ExecutorMetrics {
    /** key 级操作总数（查询/写入/读取/删除的每个 entry 计 1） */
    std::uint64_t opCount = 0;
    /** 失败的操作数（任务级失败按其全部 entry 计） */
    std::uint64_t errorCount = 0;
    /** 写入 + 读取的 value 流量字节数（查询/删除无 value 载荷，不计） */
    std::uint64_t ioBytes = 0;
    /** 完成等待的任务数（TTFT 样本数，每任务一个样本） */
    std::uint64_t taskCount = 0;
    /** 任务提交→完成时延累计（纳秒）；TTFT 口径暂定为任务级时延，requestManager 落地后按请求序列细化 */
    std::uint64_t latencyNsTotal = 0;
};

/**
 * 客户端测试执行器：只负责对给定请求数据执行增删查与结果验证，
 * 并记录 IOPS / 流量 / TTFT / 错误率埋点指标；
 * 内存注册与注销由 Scheduler 负责，请求数据由调用方装配。
 */
class Executor {
public:
    /** 绑定已初始化的 kv client（不拥有，生命周期由调用方保证） */
    void SetClient(Client *client);

    /** 对给定请求数据执行一轮验证（写前查→写入→写后查→回读比对→删除→删后查）；失败时 err 说明首次失败的步骤 */
    bool Run(Request &request, std::string &err);

    /** 获取当前指标快照（线程安全，供 Scheduler 周期采样） */
    ExecutorMetrics GetMetrics() const;

private:
    /** 提交查询并等待，校验所有 key 的存在性与 expect 一致；stage 用于日志区分验证步骤 */
    bool CheckExists(const std::vector<kv::CacheKey> &keys, bool expect, const char *stage);

    /** 提交写入任务并等待完成，校验任务级与 entry 级状态 */
    bool StoreAndWait(const std::vector<kv::KVBuffer> &entries);

    /** 提交读取任务并等待完成，校验任务级与 entry 级状态 */
    bool LoadAndWait(const std::vector<kv::KVBuffer> &entries);

    /** 提交删除任务并等待完成，校验任务级与 entry 级状态 */
    bool DeleteAndWait(const std::vector<kv::CacheKey> &keys);

    /** 埋点一次 IO 任务：entryCount 为 key 级操作数，failedCount 为失败数，bytes 为流量字节数；taskCompleted 为 true 时 latencyNs 计入 TTFT 样本（提交/等待失败的任务不计样本） */
    void RecordTask(std::size_t entryCount, std::size_t failedCount, std::uint64_t bytes,
                    bool taskCompleted, std::int64_t latencyNs);

    /** 统计任务结果中的失败 entry 数：任务级失败按全部 entry 计，否则按逐 entry 状态统计 */
    static std::size_t CountFailed(const kv::TaskResult &result, std::size_t entryCount);

    /** 汇总请求项的 value 流量字节数（各 entry 的 region 大小求和） */
    static std::uint64_t SumBytes(const std::vector<kv::KVBuffer> &entries);

    /** 绑定的 kv client（不拥有） */
    Client *client_{nullptr};

    // ---- 埋点计数器（原子变量：执行与采样线程并发安全） ----
    std::atomic<std::uint64_t> opCount_{0};
    std::atomic<std::uint64_t> errorCount_{0};
    std::atomic<std::uint64_t> ioBytes_{0};
    std::atomic<std::uint64_t> taskCount_{0};
    std::atomic<std::uint64_t> latencyNsTotal_{0};
};
