/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/executor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "log/logger_manager.h"
#include "worker/client.h"
#include "worker/request.h"

namespace {

/** 读取前回读 buffer 的填充值：读回后若仍是该值说明数据未真正回填 */
constexpr std::uint8_t kLoadFillByte = 0xA5;

/** 当前时刻的纳秒数（steady_clock 单调时钟，埋点时延用） */
std::int64_t NowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/** 打印任务结果的异常明细：任务级与逐 entry 非 OK 项逐条告警 */
void LogTaskResult(const kv::TaskResult &result, const char *opName)
{
    if (!result.status.ok()) {
        LOG_WARN("Executor：{} 任务级状态异常: code={} message={}", opName,
                 static_cast<int>(result.status.code), result.status.message);
    }
    for (std::size_t i = 0; i < result.entryStatus.size(); ++i) {
        if (!result.entryStatus[i].ok()) {
            LOG_WARN("Executor：{} entry {} 状态异常: code={} message={}", opName, i,
                     static_cast<int>(result.entryStatus[i].code), result.entryStatus[i].message);
        }
    }
}

} // namespace

void Executor::SetClient(Client *client)
{
    client_ = client;
}

bool Executor::Run(Request &request, std::string &err)
{
    if (client_ == nullptr) {
        err = "Executor 未绑定 kv client";
        return false;
    }

    // 期望副本须在读写前留存（回读比对基准）
    const std::vector<std::vector<std::uint8_t>> expected = request.valueBuffers;

    bool allPassed = true;

    // 1. 写入前查询：新 key 应全部不存在
    if (!CheckExists(request.keys, false, "写入前")) {
        err = "写入前查询验证失败";
        allPassed = false;
    }

    // 2. 异步写入并等待完成
    if (allPassed) {
        if (StoreAndWait(request.entries)) {
            LOG_INFO("Executor：写入 {} 条数据完成", request.entries.size());
        } else {
            err = "写入验证失败";
            allPassed = false;
        }
    }

    // 3. 写入后查询：应全部存在
    if (allPassed) {
        if (!CheckExists(request.keys, true, "写入后")) {
            err = "写入后查询验证失败";
            allPassed = false;
        }
    }

    // 4. 回读比对：先填 0xA5 再读回，逐字节与写入 pattern 比对
    if (allPassed) {
        for (auto &buffer : request.valueBuffers) {
            std::fill(buffer.begin(), buffer.end(), kLoadFillByte);
        }
        bool loadPassed = LoadAndWait(request.entries);
        for (std::size_t i = 0; i < request.valueBuffers.size(); ++i) {
            if (std::memcmp(request.valueBuffers[i].data(), expected[i].data(),
                            request.valueBuffers[i].size()) != 0) {
                LOG_ERROR("Executor：entry {} 回读数据与写入不一致", i);
                loadPassed = false;
            }
        }
        if (loadPassed) {
            LOG_INFO("Executor：回读比对通过，{} 条数据逐字节一致", request.valueBuffers.size());
        } else {
            err = "回读比对验证失败";
        }
        allPassed = loadPassed && allPassed;
    }

    // 5. 异步删除并等待完成
    if (allPassed) {
        if (DeleteAndWait(request.keys)) {
            LOG_INFO("Executor：删除 {} 条数据完成", request.keys.size());
        } else {
            err = "删除验证失败";
            allPassed = false;
        }
    }

    // 6. 删除后查询：应全部不存在
    if (allPassed) {
        if (!CheckExists(request.keys, false, "删除后")) {
            err = "删除后查询验证失败";
            allPassed = false;
        }
    }

    LOG_INFO("Executor：查询/写入/回读比对/删除全链路{}", allPassed ? "验证通过" : "验证失败");
    return allPassed;
}

ExecutorMetrics Executor::GetMetrics() const
{
    ExecutorMetrics metrics;
    metrics.opCount = opCount_.load(std::memory_order_relaxed);
    metrics.errorCount = errorCount_.load(std::memory_order_relaxed);
    metrics.ioBytes = ioBytes_.load(std::memory_order_relaxed);
    metrics.taskCount = taskCount_.load(std::memory_order_relaxed);
    metrics.latencyNsTotal = latencyNsTotal_.load(std::memory_order_relaxed);
    return metrics;
}

bool Executor::CheckExists(const std::vector<kv::CacheKey> &keys, bool expect, const char *stage)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    const std::int64_t beginNs = NowNs();
    if (!client_->QueryAsync(keys, taskId, err)) {
        LOG_ERROR("Executor：{}查询提交失败: {}", stage, err);
        RecordTask(keys.size(), keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：{}查询等待失败: {}", stage, err);
        RecordTask(keys.size(), keys.size(), 0, false, 0);
        return false;
    }
    const std::int64_t latencyNs = NowNs() - beginNs;

    // entryStatus 中的 KEY_NOT_EXIST/KEY_EXIST 是存在性回答的编码（不算异常），只校验任务级状态
    std::size_t mismatch = 0;
    if (!result.status.ok()) {
        LOG_ERROR("Executor：{}查询任务级状态异常: code={} message={}", stage,
                  static_cast<int>(result.status.code), result.status.message);
        mismatch = keys.size();
    } else if (!result.queryResult.has_value() ||
               result.queryResult->exists.size() != keys.size()) {
        LOG_ERROR("Executor：{}查询结果数量异常", stage);
        mismatch = keys.size();
    } else {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (result.queryResult->exists[i] != static_cast<std::uint8_t>(expect ? 1 : 0)) {
                LOG_ERROR("Executor：{}查询 entry {} 存在性异常: exists={} 期望={}", stage, i,
                          static_cast<int>(result.queryResult->exists[i]),
                          static_cast<int>(expect ? 1 : 0));
                ++mismatch;
            }
        }
    }
    RecordTask(keys.size(), mismatch, 0, true, latencyNs);

    if (mismatch != 0) {
        return false;
    }
    LOG_INFO("Executor：{}查询通过，{} 个 key 存在性均为 {}", stage, keys.size(),
             expect ? "存在" : "不存在");
    return true;
}

bool Executor::StoreAndWait(const std::vector<kv::KVBuffer> &entries)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    const std::int64_t beginNs = NowNs();
    if (!client_->StoreAsync(entries, taskId, err)) {
        LOG_ERROR("Executor：写入任务提交失败: {}", err);
        RecordTask(entries.size(), entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：写入任务等待失败: {}", err);
        RecordTask(entries.size(), entries.size(), 0, false, 0);
        return false;
    }
    const std::int64_t latencyNs = NowNs() - beginNs;

    const std::size_t failed = CountFailed(result, entries.size());
    RecordTask(entries.size(), failed, SumBytes(entries), true, latencyNs);
    if (failed != 0) {
        LogTaskResult(result, "写入");
    }
    return failed == 0;
}

bool Executor::LoadAndWait(const std::vector<kv::KVBuffer> &entries)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    const std::int64_t beginNs = NowNs();
    if (!client_->LoadAsync(entries, taskId, err)) {
        LOG_ERROR("Executor：读取任务提交失败: {}", err);
        RecordTask(entries.size(), entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：读取任务等待失败: {}", err);
        RecordTask(entries.size(), entries.size(), 0, false, 0);
        return false;
    }
    const std::int64_t latencyNs = NowNs() - beginNs;

    const std::size_t failed = CountFailed(result, entries.size());
    RecordTask(entries.size(), failed, SumBytes(entries), true, latencyNs);
    if (failed != 0) {
        LogTaskResult(result, "读取");
    }
    return failed == 0;
}

bool Executor::DeleteAndWait(const std::vector<kv::CacheKey> &keys)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    std::string err;
    const std::int64_t beginNs = NowNs();
    if (!client_->DeleteAsync(keys, taskId, err)) {
        LOG_ERROR("Executor：删除任务提交失败: {}", err);
        RecordTask(keys.size(), keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor：删除任务等待失败: {}", err);
        RecordTask(keys.size(), keys.size(), 0, false, 0);
        return false;
    }
    const std::int64_t latencyNs = NowNs() - beginNs;

    const std::size_t failed = CountFailed(result, keys.size());
    RecordTask(keys.size(), failed, 0, true, latencyNs);
    if (failed != 0) {
        LogTaskResult(result, "删除");
    }
    return failed == 0;
}

void Executor::RecordTask(std::size_t entryCount, std::size_t failedCount, std::uint64_t bytes,
                          bool taskCompleted, std::int64_t latencyNs)
{
    opCount_.fetch_add(entryCount, std::memory_order_relaxed);
    errorCount_.fetch_add(failedCount, std::memory_order_relaxed);
    ioBytes_.fetch_add(bytes, std::memory_order_relaxed);
    if (taskCompleted) {
        taskCount_.fetch_add(1, std::memory_order_relaxed);
        latencyNsTotal_.fetch_add(static_cast<std::uint64_t>(latencyNs),
                                  std::memory_order_relaxed);
    }
}

std::size_t Executor::CountFailed(const kv::TaskResult &result, std::size_t entryCount)
{
    if (!result.status.ok()) {
        return entryCount;
    }
    std::size_t failed = 0;
    for (const auto &status : result.entryStatus) {
        if (!status.ok()) {
            ++failed;
        }
    }
    // 防御 entryStatus 与 entries 数量不一致的异常场景，失败数不超过 entry 总数
    return std::min(failed, entryCount);
}

std::uint64_t Executor::SumBytes(const std::vector<kv::KVBuffer> &entries)
{
    std::uint64_t bytes = 0;
    for (const auto &entry : entries) {
        bytes += entry.buffer.region.size;
    }
    return bytes;
}
