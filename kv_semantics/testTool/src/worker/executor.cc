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
#include "request/request.h"
#include "worker/device_allocator.h"
#include "worker/kv_client_util.h"

namespace {

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
        LOG_WARN("Executor: {} task status error: code={} message={}", opName,
                 static_cast<int>(result.status.code), result.status.message);
    }
    for (std::size_t i = 0; i < result.entryStatus.size(); ++i) {
        if (!result.entryStatus[i].ok()) {
            LOG_WARN("Executor: {} entry {} status error: code={} message={}", opName, i,
                     static_cast<int>(result.entryStatus[i].code), result.entryStatus[i].message);
        }
    }
}

} // namespace

void Executor::SetClient(KvClientUtil *client)
{
    client_ = client;
}

void Executor::SetAllocator(DeviceAllocator *allocator)
{
    allocator_ = allocator;
}

bool Executor::Execute(OpType op, const Request &request, std::string &err)
{
    if (client_ == nullptr) {
        err = "executor has no kv client bound";
        return false;
    }
    switch (op) {
        case OpType::QUERY:
            return QueryAndRecord(request.keys, err);
        case OpType::STORE:
            return StoreAndRecord(request.entries, err);
        case OpType::LOAD:
            return LoadAndRecord(request, err);
        case OpType::DELETE:
            return DeleteAndRecord(request.keys, err);
    }
    err = "unknown op type";
    return false;
}

ExecutorMetrics Executor::GetMetrics() const
{
    ExecutorMetrics metrics;
    metrics.opCount = opCount_.load(std::memory_order_relaxed);
    metrics.errorCount = errorCount_.load(std::memory_order_relaxed);
    metrics.notFoundCount = notFoundCount_.load(std::memory_order_relaxed);
    metrics.mismatchCount = mismatchCount_.load(std::memory_order_relaxed);
    metrics.ioBytes = ioBytes_.load(std::memory_order_relaxed);
    metrics.taskCount = taskCount_.load(std::memory_order_relaxed);
    metrics.latencyNsTotal = latencyNsTotal_.load(std::memory_order_relaxed);
    return metrics;
}

bool Executor::QueryAndRecord(const std::vector<kv::CacheKey> &keys, std::string &err)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t beginNs = NowNs();
    if (!client_->QueryAsync(keys, taskId, err)) {
        LOG_ERROR("Executor: query submit failed: {}", err);
        RecordTask(keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: query wait failed: {}", err);
        RecordTask(keys.size(), 0, false, 0);
        return false;
    }
    RecordTask(keys.size(), 0, true, NowNs() - beginNs);

    // 任务级失败或结果缺失：观察记录（errorCount），不算工具链失败
    if (!result.status.ok() || !result.queryResult.has_value() ||
        result.queryResult->exists.size() != keys.size()) {
        LOG_ERROR("Executor: query result invalid, task status code={}",
                  static_cast<int>(result.status.code));
        errorCount_.fetch_add(keys.size(), std::memory_order_relaxed);
        return true;
    }

    // 逐 entry 记录 found / not found 分布（观察结果，不断言）
    std::uint64_t found = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (result.queryResult->exists[i] != 0) {
            ++found;
        }
    }
    notFoundCount_.fetch_add(keys.size() - found, std::memory_order_relaxed);
    LOG_INFO("Executor: query {} keys, {} found, {} not found", keys.size(), found,
             keys.size() - found);
    return true;
}

bool Executor::StoreAndRecord(const std::vector<kv::KVBuffer> &entries, std::string &err)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t beginNs = NowNs();
    if (!client_->StoreAsync(entries, taskId, err)) {
        LOG_ERROR("Executor: store submit failed: {}", err);
        RecordTask(entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: store wait failed: {}", err);
        RecordTask(entries.size(), 0, false, 0);
        return false;
    }
    RecordTask(entries.size(), SumBytes(entries), true, NowNs() - beginNs);

    // entry 级失败计数（观察）
    const std::size_t failed = CountFailed(result, entries.size());
    if (failed != 0) {
        LogTaskResult(result, "store");
    }
    errorCount_.fetch_add(failed, std::memory_order_relaxed);
    LOG_INFO("Executor: stored {} entries, {} failed", entries.size(), failed);
    return true;
}

bool Executor::LoadAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::KVBuffer> &entries = request.entries;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t beginNs = NowNs();
    if (!client_->LoadAsync(entries, taskId, err)) {
        LOG_ERROR("Executor: load submit failed: {}", err);
        RecordTask(entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: load wait failed: {}", err);
        RecordTask(entries.size(), 0, false, 0);
        return false;
    }
    RecordTask(entries.size(), SumBytes(entries), true, NowNs() - beginNs);

    // entry 级分类优先（任务级 PARTIAL_FAILED 时 entryStatus 仍带逐项详情）：
    // 未命中系计 notFound、有基准逐字节比对不一致计 mismatch、其他计 error
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t failed = 0;
    if (!result.entryStatus.empty()) {
        const std::size_t count = std::min(result.entryStatus.size(), entries.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (result.entryStatus[i].ok()) {
                // 读回数据与基准逐字节比对（基准尺寸即块尺寸；有基准才比）；
                // device 语义值缓冲不能直接解引用，经分配器 D2H 读回 host 再比对
                if (i < request.expected.size() && !request.expected[i].empty()) {
                    if (allocator_ == nullptr) {
                        err = "load readback requires a device allocator bound";
                        return false;
                    }
                    std::vector<std::uint8_t> readback(request.expected[i].size());
                    std::string readErr;
                    if (!allocator_->Read(readback.data(), entries[i].buffer.region.addr,
                                          readback.size(), readErr)) {
                        // 读回失败属工具链异常（内存访问链路不可用），后续观察不可信
                        err = "load entry " + std::to_string(i) + " readback failed: " + readErr;
                        return false;
                    }
                    if (std::memcmp(readback.data(), request.expected[i].data(),
                                    request.expected[i].size()) != 0) {
                        ++mismatch;
                        LOG_ERROR("Executor: load entry {} data mismatch with baseline", i);
                    }
                }
                continue;
            }
            if (IsNotFound(result.entryStatus[i])) {
                ++notFound;
            } else {
                ++failed;
            }
        }
        // entryStatus 数量不足的异常场景，缺失部分按失败计
        if (result.entryStatus.size() < entries.size()) {
            failed += entries.size() - result.entryStatus.size();
        }
    } else if (!result.status.ok()) {
        // entryStatus 为空（任务未到 entry 层）：按任务级状态整体归类
        if (IsNotFound(result.status)) {
            notFound = entries.size();
        } else {
            failed = entries.size();
            LogTaskResult(result, "load");
        }
    }
    notFoundCount_.fetch_add(notFound, std::memory_order_relaxed);
    mismatchCount_.fetch_add(mismatch, std::memory_order_relaxed);
    errorCount_.fetch_add(failed, std::memory_order_relaxed);
    if (failed != 0) {
        LogTaskResult(result, "load");
    }

    // 非性能模式输出 get 读回明细日志（key/offset/读回首段/尾部），供人工校验一致性；
    // 读回失败属工具链异常（内存访问链路不可用），与比对路径同一语义
    if (!WriteGetLog(request, result, err)) {
        return false;
    }

    LOG_INFO("Executor: load {} entries, {} not found, {} mismatch, {} failed", entries.size(),
             notFound, mismatch, failed);
    return true;
}

bool Executor::WriteGetLog(const Request &request, const kv::TaskResult &result, std::string &err)
{
    // get 读回日志未启用（perf=on）：不组装内容，直接返回（零开销）
    if (LoggerManager::GetGetLogger() == nullptr) {
        return true;
    }

    const std::vector<kv::KVBuffer> &entries = request.entries;

    // 批量返回码：优先逐 entry 状态码逗号拼接（wait 拿到的信息），entryStatus 缺失时退化为任务级状态码
    std::string codes;
    if (!result.entryStatus.empty()) {
        codes = std::to_string(static_cast<int>(result.entryStatus[0].code));
        for (std::size_t i = 1; i < result.entryStatus.size(); ++i) {
            codes += "," + std::to_string(static_cast<int>(result.entryStatus[i].code));
        }
    } else {
        codes = std::to_string(static_cast<int>(result.status.code));
    }

    // 每任务一行，字段名(值) 形式标注（裸值无法分辨字段）：
    // opcode(get) nsid(x) codes(y) key(k) offset(o) value(v) len(l) tail(t)——
    // 多 entry batch 时 codes 只出现一次、其后逐 entry 追加 key/offset/value/len/tail 组；
    // 行格式与全量日志一致（get logger 配置同款，pattern 沿用 log.pattern），业务代码不感知
    std::string line = "opcode(get) nsid(" + std::to_string(request.nsid) + ") codes(" + codes + ")";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        // key 字段：key 低 64 位与高 16 位十进制字符串直接拼接；value 生成种子串以
        // "低64 高16 offset" 开头，len + 2 恰好覆盖该前缀（两个分隔空格），供逐字段比对
        std::uint64_t keyLow = 0;
        if (i < request.keys.size()) {
            std::memcpy(&keyLow, request.keys[i].data(), sizeof(keyLow));
        }
        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string keyStr = std::to_string(keyLow) + std::to_string(keyHigh);
        const std::string offsetStr = std::to_string(entries[i].offset);
        const std::size_t len = keyStr.size() + offsetStr.size();

        // entry 完成状态：entryStatus 缺失时以任务级状态代表全部 entry
        const bool entryOk =
            (i < result.entryStatus.size()) ? result.entryStatus[i].ok() : result.status.ok();

        // 命中才读回数据：首段取 len + 2 字节、尾部固定 8 字节（均不超过块实际大小，
        // 经分配器 D2H 读回，device 语义缓冲不能直接解引用）；未命中或空块以 "-" 占位
        std::string value = "-";
        std::string tail = "-";
        if (entryOk) {
            const kv::MemoryRegion &region = entries[i].buffer.region;
            const std::size_t blockLen = static_cast<std::size_t>(region.size);
            const std::size_t headLen = std::min(len + 2, blockLen);
            const std::size_t tailLen = std::min<std::size_t>(8, blockLen);
            if (headLen != 0 || tailLen != 0) {
                if (allocator_ == nullptr) {
                    err = "get log readback requires a device allocator bound";
                    return false;
                }
            }
            std::string readErr;
            std::vector<std::uint8_t> head(headLen);
            if (headLen != 0 && !allocator_->Read(head.data(), region.addr, headLen, readErr)) {
                err = "get log entry " + std::to_string(i) + " head read failed: " + readErr;
                return false;
            }
            std::vector<std::uint8_t> tailBuf(tailLen);
            if (tailLen != 0 && !allocator_->Read(tailBuf.data(), region.addr + blockLen - tailLen,
                                                  tailLen, readErr)) {
                err = "get log entry " + std::to_string(i) + " tail read failed: " + readErr;
                return false;
            }
            if (headLen != 0) {
                value.assign(head.begin(), head.end());
            }
            if (tailLen != 0) {
                tail.assign(tailBuf.begin(), tailBuf.end());
            }
        }

        line += " key(" + keyStr + ") offset(" + offsetStr + ") value(" + value + ") len(" +
                std::to_string(len) + ") tail(" + tail + ")";
    }
    // 内容作为 fmt 参数而非格式串（"{}" 显式转义）：value/tail 为二进制转字符串，
    // 可能含 '{' '}' 字节，作为格式串会被 fmt 误解析
    LOG_INFO_GET("{}", line);
    return true;
}

bool Executor::DeleteAndRecord(const std::vector<kv::CacheKey> &keys, std::string &err)
{
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t beginNs = NowNs();
    if (!client_->DeleteAsync(keys, taskId, err)) {
        LOG_ERROR("Executor: delete submit failed: {}", err);
        RecordTask(keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: delete wait failed: {}", err);
        RecordTask(keys.size(), 0, false, 0);
        return false;
    }
    RecordTask(keys.size(), 0, true, NowNs() - beginNs);

    // entry 级分类优先：未命中系计 notFound，其他计 error；entryStatus 为空时按任务级整体归类
    std::uint64_t notFound = 0;
    std::uint64_t failed = 0;
    if (!result.entryStatus.empty()) {
        const std::size_t count = std::min(result.entryStatus.size(), keys.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (IsNotFound(result.entryStatus[i])) {
                ++notFound;
            } else if (!result.entryStatus[i].ok()) {
                ++failed;
            }
        }
        if (result.entryStatus.size() < keys.size()) {
            failed += keys.size() - result.entryStatus.size();
        }
    } else if (!result.status.ok()) {
        if (IsNotFound(result.status)) {
            notFound = keys.size();
        } else {
            failed = keys.size();
        }
    }
    notFoundCount_.fetch_add(notFound, std::memory_order_relaxed);
    errorCount_.fetch_add(failed, std::memory_order_relaxed);
    if (failed != 0) {
        LogTaskResult(result, "delete");
    }
    LOG_INFO("Executor: delete {} keys, {} not found, {} failed", keys.size(), notFound, failed);
    return true;
}

void Executor::RecordTask(std::size_t entryCount, std::uint64_t bytes, bool taskCompleted,
                          std::int64_t latencyNs)
{
    opCount_.fetch_add(entryCount, std::memory_order_relaxed);
    ioBytes_.fetch_add(bytes, std::memory_order_relaxed);
    if (taskCompleted) {
        taskCount_.fetch_add(1, std::memory_order_relaxed);
        latencyNsTotal_.fetch_add(static_cast<std::uint64_t>(latencyNs), std::memory_order_relaxed);
    }
}

bool Executor::IsNotFound(const kv::Status &status)
{
    // 全部“目标 key 无数据”语义的状态码：任务级 NOT_FOUND 与 entry 级各变体统一归未命中观察
    switch (status.code) {
        case kv::StatusCode::NOT_FOUND:
        case kv::StatusCode::ENTRY_KEY_NOT_FOUND:
        case kv::StatusCode::ENTRY_DATA_NOT_EXIST:
        case kv::StatusCode::ENTRY_KEY_NOT_EXIST:
        case kv::StatusCode::CQE_KEY_NOT_EXIST:
            return true;
        default:
            return false;
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
    if (result.entryStatus.size() < entryCount) {
        failed += entryCount - result.entryStatus.size();
    }
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
