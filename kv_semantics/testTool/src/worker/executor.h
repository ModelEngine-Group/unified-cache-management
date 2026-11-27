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

/** 执行操作类型（执行侧词汇；input 侧 OrcOp 到此的映射由 Scheduler 完成）；枚举值即指标分桶下标 */
enum class OpType {
    QUERY,  // 存在性查询（只查记录，不断言）
    STORE,  // 异步写入
    LOAD,   // 异步读取（有基准时回读比对）
    DELETE  // 异步删除
};

/** 分桶的操作类型总数（OpType 枚举值范围） */
constexpr std::size_t kOpTypeCount = 4;

/** 执行器埋点指标（自创建起的累计值，单调递增；Scheduler 周期采样求窗口速率） */
struct ExecutorMetrics {
    /** key 级操作总数（查询/写入/读取/删除的每个 entry 计 1；全量口径，含未命中与失败） */
    std::uint64_t opCount = 0;
    /** 失败的操作数（提交/等待失败与 entry 级非 NOT_FOUND 错误） */
    std::uint64_t errorCount = 0;
    /** NOT_FOUND 观察计数（get/delete 打不存在的 key；流量注入的正常观察结果，非工具失败） */
    std::uint64_t notFoundCount = 0;
    /** 回读比对不一致计数（get 读回数据与基准不符） */
    std::uint64_t mismatchCount = 0;
    /** 写入 + 读取的 value 流量字节数（成功 entry 口径；查询/删除无载荷不计） */
    std::uint64_t ioBytes = 0;
    /** 完成等待的任务数（TTFT 样本数，每任务一个样本） */
    std::uint64_t taskCount = 0;
    /** 任务提交→完成时延累计（纳秒） */
    std::uint64_t latencyNsTotal = 0;
    /** 分接口操作数（下标为 OpType 枚举值；全量口径） */
    std::uint64_t opCountByOp[kOpTypeCount] = {0};
    /** 分接口流量字节数（下标为 OpType 枚举值；成功 entry 口径） */
    std::uint64_t ioBytesByOp[kOpTypeCount] = {0};
};

/**
 * 客户端测试执行器：按单个测试任务执行增删查读并记录结果埋点；
 * 持续执行语义：NOT_FOUND、回读比对不一致等观察结果只计数不构成失败，
 * 返回 false 仅表示工具链失败（提交/等待调用失败），由调用方决定是否继续。
 */
class Executor {
public:
    /** 构造：分接口原子计数桶显式清零（atomic 数组不能就地默认初始化） */
    Executor();

    /** 绑定已初始化的 kv client 工具封装（不拥有，生命周期由调用方保证） */
    void SetClient(KvClientUtil *client);

    /** 绑定值缓冲内存分配器（不拥有；load 回读比对经其 D2H 读回，生命周期由调用方保证） */
    void SetAllocator(DeviceAllocator *allocator);

    /** 设置 perf 压测模式（on 时全部操作仅记失败请求明细，成功不记） */
    void SetPerfMode(bool perf);

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
    bool QueryAndRecord(const Request &request, std::string &err);

    /** 提交写入并等待，entry 失败计 errorCount；返回 false 仅当提交/等待失败 */
    bool StoreAndRecord(const Request &request, std::string &err);

    /** 提交读取并等待：NOT_FOUND 计 notFoundCount、有基准逐项 memcmp 不一致计
     * mismatchCount、其他失败计 errorCount */
    bool LoadAndRecord(const Request &request, std::string &err);

    /**
     * 把 put / get 任务响应明细写入请求明细日志（每 entry 一行，字段名(值) 形式：
     * (opcode), ns, return batch code(整批全部 entry 状态码逗号拼接，逐行重复),
     * key code(本 entry 返回码), inputkey, key+offset(低64位 高16位 offset), offset,
     * value(读回缓冲实际内容), len(低64位位数+offset位数), tail(尾 8 字节)；
     * get 无论 entry 状态总是读回缓冲（预填哨兵串即“没读到数据”标记），put 仅成功
     * entry 读回、失败以 "-" 占位；读回失败视为工具链失败）
     */
    bool WritePutGetLog(const Request &request, const kv::TaskResult &result, const char *opcode,
                        std::string &err);

    /** 把 exist 任务响应明细写入其余操作明细日志（每任务一行聚合格式：(exist)
     *  all_block_ids[input 原文 key 逗号拼接] key_low_high[逐 key“低64位 高16位”拼接]
     *  prefix_hit_len(前缀命中 key 数)）；调用侧已保证 queryResult 有值且尺寸匹配 */
    void WriteExistLog(const Request &request, const kv::TaskResult &result);

    /** 把 delete 任务响应明细写入其余操作明细日志（每任务一行聚合格式，字段同 exist；
     *  delete 无前缀命中信息，prefix_hit_len 以 "-" 占位；delete 无 value 载荷；
     *  notFound/failed 为调用侧算好的未命中与失败计数，供 perf 模式失败判定） */
    void WriteDeleteLog(const Request &request, std::uint64_t notFound, std::uint64_t failed);

    /** 提交删除并等待：NOT_FOUND 计 notFoundCount、其他失败计 errorCount */
    bool DeleteAndRecord(const Request &request, std::string &err);

    /** 埋点一次任务的操作数与流量（op 用于分接口桶；bytes 为成功 entry 口径，失败/未命中计
     * 0）；taskCompleted 为 true 时 latencyNs 计入 TTFT 样本 */
    void RecordTask(OpType op, std::size_t entryCount, std::uint64_t bytes, bool taskCompleted,
                    std::int64_t latencyNs);

    /** 判定状态是否为“目标 key 无数据”观察（get/delete 不存在 key 的正常观察结果，非工具失败） */
    static bool IsNotFound(const kv::Status &status);

    /** 统计任务结果中的失败 entry 数：任务级失败按全部 entry 计，entryStatus 缺失的差额也按失败计
     */
    static std::size_t CountFailed(const kv::TaskResult &result, std::size_t entryCount);

    /** 汇总请求项的 value 流量字节数（各 entry 的 region 大小求和） */
    static std::uint64_t SumBytes(const std::vector<kv::KVBuffer> &entries);

    /** 成功 entry 的流量字节和；entryStatus 缺失时按任务级状态归类全部 */
    static std::uint64_t SumOkBytes(const kv::TaskResult &result,
                                    const std::vector<kv::KVBuffer> &entries);

    /** 绑定的 kv client 工具封装（不拥有） */
    KvClientUtil *client_{nullptr};

    /** 绑定的值缓冲内存分配器（不拥有；回读比对用） */
    DeviceAllocator *allocator_{nullptr};

    /** perf 压测模式（on 时仅记失败请求明细） */
    bool perf_{false};

    // ---- 埋点计数器（原子变量：执行与采样线程并发安全） ----
    std::atomic<std::uint64_t> opCount_{0};
    std::atomic<std::uint64_t> errorCount_{0};
    std::atomic<std::uint64_t> notFoundCount_{0};
    std::atomic<std::uint64_t> mismatchCount_{0};
    std::atomic<std::uint64_t> ioBytes_{0};
    std::atomic<std::uint64_t> taskCount_{0};
    std::atomic<std::uint64_t> latencyNsTotal_{0};
    // 分接口桶（下标为 OpType 枚举值；数组 of atomic 不可拷贝，用逐成员数组）
    std::atomic<std::uint64_t> opCountByOp_[kOpTypeCount];
    std::atomic<std::uint64_t> ioBytesByOp_[kOpTypeCount];
};
