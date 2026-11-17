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

class KvClientUtil;
class DeviceAllocator;
struct Request;

/** 执行操作类型（执行侧词汇；input 侧 OrcOp 到此的映射由 Scheduler 完成） */
enum class OpType {
    QUERY,  // 存在性查询（只查记录，不断言）
    STORE,  // 异步写入
    LOAD,   // 异步读取（有基准时回读比对）
    DELETE  // 异步删除
};

/** 执行器埋点指标（自创建起的累计值，单调递增；Scheduler 周期采样求窗口速率） */
struct ExecutorMetrics {
    /** key 级操作总数（查询/写入/读取/删除的每个 entry 计 1） */
    std::uint64_t opCount = 0;
    /** 失败的操作数（提交/等待失败与 entry 级非 NOT_FOUND 错误） */
    std::uint64_t errorCount = 0;
    /** NOT_FOUND 观察计数（get/delete 打不存在的 key；流量注入的正常观察结果，非工具失败） */
    std::uint64_t notFoundCount = 0;
    /** 回读比对不一致计数（get 读回数据与基准不符） */
    std::uint64_t mismatchCount = 0;
    /** 写入 + 读取的 value 流量字节数（查询/删除无 value 载荷，不计） */
    std::uint64_t ioBytes = 0;
    /** 完成等待的任务数（TTFT 样本数，每任务一个样本） */
    std::uint64_t taskCount = 0;
    /** 任务提交→完成时延累计（纳秒） */
    std::uint64_t latencyNsTotal = 0;
};

/**
 * 客户端测试执行器：按单个测试任务执行增删查读并记录结果埋点；
 * 持续执行语义：NOT_FOUND、回读比对不一致等观察结果只计数不构成失败，
 * 返回 false 仅表示工具链失败（提交/等待调用失败），由调用方决定是否继续。
 */
class Executor {
public:
    /** 绑定已初始化的 kv client 工具封装（不拥有，生命周期由调用方保证） */
    void SetClient(KvClientUtil *client);

    /** 绑定值缓冲内存分配器（不拥有；load 回读比对经其 D2H 读回，生命周期由调用方保证） */
    void SetAllocator(DeviceAllocator *allocator);

    /**
     * 执行单个测试任务（op 决定操作，request 为任务数据包）
     * @param op 执行操作类型
     * @param request 任务数据包（GET 时 expected 提供逐项比对基准，空 vector 不比对）
     * @param err 工具链失败时的错误描述
     * @return false 仅当提交/等待失败（工具链失败）；观察结果只进埋点
     */
    bool Execute(OpType op, const Request &request, std::string &err);

    /** 获取当前指标快照（线程安全，供 Scheduler 周期采样） */
    ExecutorMetrics GetMetrics() const;

private:
    /** 提交查询并等待，found/not found 逐项记录进埋点；返回 false 仅当提交/等待失败 */
    bool QueryAndRecord(const std::vector<kv::CacheKey> &keys, std::string &err);

    /** 提交写入并等待，entry 失败计 errorCount；返回 false 仅当提交/等待失败 */
    bool StoreAndRecord(const std::vector<kv::KVBuffer> &entries, std::string &err);

    /** 提交读取并等待：NOT_FOUND 计 notFoundCount、有基准逐项 memcmp 不一致计 mismatchCount、其他失败计 errorCount */
    bool LoadAndRecord(const Request &request, std::string &err);

    /** 非性能模式把 get 任务响应明细写入 get 读回日志（每任务一行）；读回失败视为工具链失败 */
    bool WriteGetLog(const Request &request, const kv::TaskResult &result, std::string &err);

    /** 提交删除并等待：NOT_FOUND 计 notFoundCount、其他失败计 errorCount */
    bool DeleteAndRecord(const std::vector<kv::CacheKey> &keys, std::string &err);

    /** 埋点一次任务的操作数与流量；taskCompleted 为 true 时 latencyNs 计入 TTFT 样本 */
    void RecordTask(std::size_t entryCount, std::uint64_t bytes, bool taskCompleted,
                    std::int64_t latencyNs);

    /** 判定状态是否为“目标 key 无数据”观察（get/delete 不存在 key 的正常观察结果，非工具失败） */
    static bool IsNotFound(const kv::Status &status);

    /** 统计任务结果中的失败 entry 数：任务级失败按全部 entry 计，entryStatus 缺失的差额也按失败计 */
    static std::size_t CountFailed(const kv::TaskResult &result, std::size_t entryCount);

    /** 汇总请求项的 value 流量字节数（各 entry 的 region 大小求和） */
    static std::uint64_t SumBytes(const std::vector<kv::KVBuffer> &entries);

    /** 绑定的 kv client 工具封装（不拥有） */
    KvClientUtil *client_{nullptr};

    /** 绑定的值缓冲内存分配器（不拥有；回读比对用） */
    DeviceAllocator *allocator_{nullptr};

    // ---- 埋点计数器（原子变量：执行与采样线程并发安全） ----
    std::atomic<std::uint64_t> opCount_{0};
    std::atomic<std::uint64_t> errorCount_{0};
    std::atomic<std::uint64_t> notFoundCount_{0};
    std::atomic<std::uint64_t> mismatchCount_{0};
    std::atomic<std::uint64_t> ioBytes_{0};
    std::atomic<std::uint64_t> taskCount_{0};
    std::atomic<std::uint64_t> latencyNsTotal_{0};
};
