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

#include "worker/executor.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <vector>
#include "log/logger_manager.h"
#include "request/request.h"
#include "worker/device_allocator.h"
#include "worker/kv_client_util.h"

namespace {

/** 当前时刻的纳秒数（steady_clock 单调时钟，埋点时延用） */
std::int64_t CurrentNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/** 构造 IO 提交的请求级选项：input 的 ns（namespace 名）直接填 IoOptions.kvns，
 *  由 kv client 哈希派生协议帧 namespace id（与对端 view 从 namespace 名派生同口径） */
kv::IoOptions MakeIoOptions(const Request &request)
{
    kv::IoOptions options;
    options.kvns = request.ns;
    return options;
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

/** 组装 exist/delete 聚合明细行的两个 key 字段：input 原文 key 逗号拼接与
 *  逐 key“input 原文 key 高16位”拼接 */
void BuildAggKeyFields(const Request &request, std::string &rawKeys, std::string &highs)
{
    for (std::size_t i = 0; i < request.keys.size(); ++i) {
        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string rawKey =
            (i < request.rawKeys.size()) ? request.rawKeys[i] : std::string("-");
        if (i != 0) {
            rawKeys += ',';
            highs += ',';
        }
        rawKeys += rawKey;
        highs += rawKey + " " + std::to_string(keyHigh);
    }
}

}  // namespace

Executor::Executor()
{
    // 分接口原子桶清零（atomic 数组不能就地默认初始化）；样本环形按容量展开
    for (std::size_t i = 0; i < kOpTypeCount; ++i) {
        opCountByOp_[i].store(0, std::memory_order_relaxed);
        ioBytesByOp_[i].store(0, std::memory_order_relaxed);
        latencyNsTotalByOp_[i].store(0, std::memory_order_relaxed);
        taskCountByOp_[i].store(0, std::memory_order_relaxed);
        latencySamples_[i].resize(kLatencySampleCapacityPerOp);
    }
}

void Executor::SetClient(KvClientUtil *client) { client_ = client; }

void Executor::SetAllocator(DeviceAllocator *allocator) { allocator_ = allocator; }

void Executor::SetPerfMode(bool perf) { perf_ = perf; }

void Executor::SetDetailValueLimit(std::size_t bytes) { detailValueLimit_ = bytes; }

bool Executor::Execute(OpType op, const Request &request, std::string &err)
{
    if (client_ == nullptr) {
        err = "executor has no kv client bound";
        return false;
    }
    switch (op) {
        case OpType::QUERY: return QueryAndRecord(request, err);
        case OpType::STORE: return StoreAndRecord(request, err);
        case OpType::LOAD: return LoadAndRecord(request, err);
        case OpType::DELETE: return DeleteAndRecord(request, err);
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
    metrics.ioBytes = ioBytes_.load(std::memory_order_relaxed);
    metrics.taskCount = taskCount_.load(std::memory_order_relaxed);
    metrics.latencyNsTotal = latencyNsTotal_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < kOpTypeCount; ++i) {
        metrics.opCountByOp[i] = opCountByOp_[i].load(std::memory_order_relaxed);
        metrics.ioBytesByOp[i] = ioBytesByOp_[i].load(std::memory_order_relaxed);
        metrics.latencyNsTotalByOp[i] = latencyNsTotalByOp_[i].load(std::memory_order_relaxed);
        metrics.taskCountByOp[i] = taskCountByOp_[i].load(std::memory_order_relaxed);
    }
    return metrics;
}

void Executor::AppendLatencySamplesUs(std::vector<std::uint32_t> (&out)[kOpTypeCount]) const
{
    const std::lock_guard<std::mutex> lock(samplesMutex_);
    for (std::size_t op = 0; op < kOpTypeCount; ++op) {
        const std::vector<std::uint32_t> &buf = latencySamples_[op];
        const std::size_t count = latencySampleCount_[op];
        if (count == 0) { continue; }
        // 未回绕：有效段 [0, count)（写游标恰在 count）；回绕后：[pos, cap) + [0, pos)（pos
        // 为最老样本）
        if (count < buf.size()) {
            out[op].insert(out[op].end(), buf.begin(),
                           buf.begin() + static_cast<std::ptrdiff_t>(count));
        } else {
            const std::size_t pos = latencySamplePos_[op];
            out[op].insert(out[op].end(), buf.begin() + static_cast<std::ptrdiff_t>(pos),
                           buf.end());
            out[op].insert(out[op].end(), buf.begin(),
                           buf.begin() + static_cast<std::ptrdiff_t>(pos));
        }
    }
}

bool Executor::QueryAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::CacheKey> &keys = request.keys;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->QueryAsync(keys, taskId, MakeIoOptions(request), err)) {
        LOG_ERROR("Executor: query submit failed: {}", err);
        RecordTask(OpType::QUERY, keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: query wait failed: {}", err);
        RecordTask(OpType::QUERY, keys.size(), 0, false, 0);
        return false;
    }
    RecordTask(OpType::QUERY, keys.size(), 0, true, CurrentNs() - startNs);

    // 任务级失败或结果缺失：观察记录，不算工具链失败（无可信逐 key 数据，不写明细）
    if (!result.status.ok() || !result.queryResult.has_value() ||
        result.queryResult->exists.size() != keys.size()) {
        LOG_ERROR("Executor: query result invalid, task status code={}",
                  static_cast<int>(result.status.code));
        errorCount_.fetch_add(keys.size(), std::memory_order_relaxed);
        return true;
    }

    // 逐 entry 记录 found / not found 分布（观察结果）
    std::uint64_t found = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (result.queryResult->exists[i] != 0) { ++found; }
    }
    notFoundCount_.fetch_add(keys.size() - found, std::memory_order_relaxed);

    // 输出 exist 任务明细日志（每任务一行聚合）
    WriteExistLog(request, result);
    const std::uint64_t notFound = keys.size() - found;
    // 未命中打 ERROR 仅为醒目提示（notfound 是观察指标，不算工具链失败）
    if (notFound != 0) {
        LOG_ERROR("Executor: query {} keys, {} found, {} not found", keys.size(), found, notFound);
    } else {
        LOG_INFO("Executor: query {} keys, {} found, {} not found", keys.size(), found, notFound);
    }
    return true;
}

bool Executor::StoreAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::KVBuffer> &entries = request.entries;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->StoreAsync(entries, taskId, MakeIoOptions(request), err)) {
        LOG_ERROR("Executor: store submit failed: {}", err);
        RecordTask(OpType::STORE, entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: store wait failed: {}", err);
        RecordTask(OpType::STORE, entries.size(), 0, false, 0);
        return false;
    }

    // entry 级失败计数（观察）；流量为成功 entry 口径（失败 entry 计 0 字节）
    const std::size_t failed = CountFailed(result, entries.size());
    const std::uint64_t successBytes = SumOkBytes(result, entries);
    RecordTask(OpType::STORE, entries.size(), successBytes, true, CurrentNs() - startNs);
    if (failed != 0) { LogTaskResult(result, "store"); }
    errorCount_.fetch_add(failed, std::memory_order_relaxed);

    // 输出 put 任务明细日志（写入后读回缓冲内容）；读回失败视为工具链失败
    if (!WritePutGetLog(request, result, "put", err)) { return false; }
    LOG_INFO("Executor: stored {} entries, {} failed", entries.size(), failed);
    return true;
}

bool Executor::LoadAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::KVBuffer> &entries = request.entries;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->LoadAsync(entries, taskId, MakeIoOptions(request), err)) {
        LOG_ERROR("Executor: load submit failed: {}", err);
        RecordTask(OpType::LOAD, entries.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: load wait failed: {}", err);
        RecordTask(OpType::LOAD, entries.size(), 0, false, 0);
        return false;
    }

    // entry 级分类优先（任务级 PARTIAL_FAILED 时 entryStatus 仍带逐项详情）：
    // 未命中计 notFound、其他计 error；流量为成功 entry 口径；读回内容不做自动
    // 比对，由明细日志人工核对
    std::uint64_t notFound = 0;
    std::uint64_t failed = 0;
    std::uint64_t successBytes = 0;
    if (!result.entryStatus.empty()) {
        const std::size_t count = std::min(result.entryStatus.size(), entries.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (result.entryStatus[i].ok()) {
                successBytes += entries[i].buffer.region.size;
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
    } else {
        // entryStatus 为空且任务级成功：全部 entry 视为成功
        successBytes = SumBytes(entries);
    }
    RecordTask(OpType::LOAD, entries.size(), successBytes, true, CurrentNs() - startNs);
    notFoundCount_.fetch_add(notFound, std::memory_order_relaxed);
    errorCount_.fetch_add(failed, std::memory_order_relaxed);
    if (failed != 0) { LogTaskResult(result, "load"); }

    // 输出 get 任务明细日志（读回缓冲实际内容）；读回失败视为工具链失败
    if (!WritePutGetLog(request, result, "get", err)) { return false; }

    // 未命中打 ERROR 仅为醒目提示（notfound 是观察指标，不算工具链失败）
    if (notFound != 0) {
        LOG_ERROR("Executor: load {} entries, {} not found, {} failed", entries.size(), notFound,
                  failed);
    } else {
        LOG_INFO("Executor: load {} entries, {} not found, {} failed", entries.size(), notFound,
                 failed);
    }
    return true;
}

bool Executor::WritePutGetLog(const Request &request, const kv::TaskResult &result,
                              const char *opcode, std::string &err)
{
    const bool isGet = std::string(opcode) == "get";
    // 任务级返回码（entryStatus 缺失时代表整批）
    const int batchCode = static_cast<int>(result.status.code);

    const std::vector<kv::KVBuffer> &entries = request.entries;

    // 整批返回码串：全部 entry 状态码逗号拼接（任务内逐行重复，保持行自包含）
    std::string batchCodes;
    if (result.entryStatus.empty()) {
        batchCodes = std::to_string(batchCode);
    } else {
        batchCodes.reserve(result.entryStatus.size() * 4);
        for (std::size_t i = 0; i < result.entryStatus.size(); ++i) {
            if (i != 0) { batchCodes += ','; }
            batchCodes += std::to_string(static_cast<int>(result.entryStatus[i].code));
        }
    }

    // 每 entry 一行，字段名(值) 形式，字段间 ", " 分隔；inputkey 为 input 原文 key，
    // key+offset 为 key 低 64 位 / 高 16 位 / 块内 offset 三数十进制
    for (std::size_t i = 0; i < entries.size(); ++i) {
        // 该 entry 自身返回码（entryStatus 缺失时以任务级代表）
        const int entryCode = (i < result.entryStatus.size())
                                  ? static_cast<int>(result.entryStatus[i].code)
                                  : batchCode;

        // perf 模式仅打印失败请求（任务级或该 entry 返回码非 0，含未命中），成功跳过
        if (perf_ && batchCode == 0 && entryCode == 0) { continue; }

        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string rawKey =
            (i < request.rawKeys.size()) ? request.rawKeys[i] : std::string("-");
        const std::string keyHighStr = std::to_string(keyHigh);
        const std::string offsetStr = std::to_string(entries[i].offset);
        // len：key 与 offset 两个十进制串的位数和（纯显示用途）
        const std::size_t len = rawKey.size() + offsetStr.size();

        // entry 完成状态（entryStatus 缺失时以任务级代表）
        const bool entryOk =
            (i < result.entryStatus.size()) ? result.entryStatus[i].ok() : result.status.ok();

        // 读回组装：get 无论 entry 状态总是读回缓冲（预填哨兵串即“没读到数据”标记），
        // put 仅成功 entry 读回、失败以 "-" 占位；首段长 prefixLens[i]（超过
        // detail_value_bytes 时截断，尾部标记 ...+未显示字节数）、尾段固定 8 字节，
        // 均经分配器 D2H 读回
        std::string value = "-";
        std::string tail = "-";
        const kv::MemoryRegion &region = entries[i].buffer.region;
        const std::size_t blockLen = static_cast<std::size_t>(region.size);
        const std::size_t prefixLen = (i < request.prefixLens.size()) ? request.prefixLens[i] : 0;
        const std::size_t valueBytes = std::min(prefixLen, blockLen);
        std::size_t headLen = valueBytes;
        if (detailValueLimit_ != 0) { headLen = std::min(headLen, detailValueLimit_); }
        const std::size_t tailLen = std::min<std::size_t>(8, blockLen);
        if ((isGet || entryOk) && (headLen != 0 || tailLen != 0)) {
            if (allocator_ == nullptr) {
                err = "req log readback requires a device allocator bound";
                return false;
            }
            std::string readErr;
            std::vector<std::uint8_t> head(headLen);
            if (headLen != 0 && !allocator_->Read(head.data(), region.addr, headLen, readErr)) {
                err = "req log entry " + std::to_string(i) + " head read failed: " + readErr;
                return false;
            }
            std::vector<std::uint8_t> tailBuf(tailLen);
            if (tailLen != 0 && !allocator_->Read(tailBuf.data(), region.addr + blockLen - tailLen,
                                                  tailLen, readErr)) {
                err = "req log entry " + std::to_string(i) + " tail read failed: " + readErr;
                return false;
            }
            if (headLen != 0) {
                value.assign(head.begin(), head.end());
                // 截断标记：...+N（N 为未显示字节数），避免误认完整读回内容
                if (valueBytes > headLen) { value += fmt::format("...+{}", valueBytes - headLen); }
            }
            if (tailLen != 0) { tail.assign(tailBuf.begin(), tailBuf.end()); }
        }

        // value/tail 为二进制转字符串，可能含 '{' '}' 字节，经 "{}" 作参数而非格式串
        const std::string line = "(" + std::string(opcode) + "), ns (" + request.ns +
                                 "), return batch code (" + batchCodes + "), key code (" +
                                 std::to_string(entryCode) + "), inputkey (" + rawKey +
                                 "), key+offset (" + rawKey + " " + keyHighStr + " " + offsetStr +
                                 "), offset (" + offsetStr + "), len (" + std::to_string(len) +
                                 "), tail (" + tail + "), value (" + value + ")";
        if (isGet) {
            LOG_INFO_GET("{}", line);
        } else {
            LOG_INFO_OTHER("{}", line);
        }
    }
    return true;
}

void Executor::WriteExistLog(const Request &request, const kv::TaskResult &result)
{
    // perf 模式仅打印失败请求（任一 key 未存在），全部命中跳过
    if (perf_) {
        bool allHit = true;
        for (const auto hit : result.queryResult->exists) {
            if (hit == 0) {
                allHit = false;
                break;
            }
        }
        if (allHit) { return; }
    }

    // 每任务一行聚合：ns 为请求 namespace，all_block_ids 为 input 原文 key 逗号拼接，
    // key_high 为逐 key “原文key 高16位”拼接，prefix_hit_len 为 kv client 返回的前缀命中 key 数
    std::string rawKeys;
    std::string highs;
    BuildAggKeyFields(request, rawKeys, highs);
    const std::string line = "(exist), ns (" + request.ns + "), all_block_ids[" + rawKeys +
                             "] key_high[" + highs + "] prefix_hit_len (" +
                             std::to_string(result.queryResult->prefixHitKeys) + ")";
    LOG_INFO_OTHER("{}", line);
}

void Executor::WriteDeleteLog(const Request &request, std::uint64_t notFound, std::uint64_t failed)
{
    // perf 模式仅打印失败请求（任一 key 未命中或失败），全部成功跳过
    if (perf_ && notFound == 0 && failed == 0) { return; }

    // 每任务一行聚合：字段同 exist；delete 无前缀命中信息，prefix_hit_len 以 "-" 占位
    std::string rawKeys;
    std::string highs;
    BuildAggKeyFields(request, rawKeys, highs);
    const std::string line = "(delete), ns (" + request.ns + "), all_block_ids[" + rawKeys +
                             "] key_high[" + highs + "] prefix_hit_len (-)";
    LOG_INFO_OTHER("{}", line);
}

bool Executor::DeleteAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::CacheKey> &keys = request.keys;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->DeleteAsync(keys, taskId, MakeIoOptions(request), err)) {
        LOG_ERROR("Executor: delete submit failed: {}", err);
        RecordTask(OpType::DELETE, keys.size(), 0, false, 0);
        return false;
    }
    kv::TaskResult result;
    if (!client_->Wait(taskId, 0, result, err)) {
        LOG_ERROR("Executor: delete wait failed: {}", err);
        RecordTask(OpType::DELETE, keys.size(), 0, false, 0);
        return false;
    }
    RecordTask(OpType::DELETE, keys.size(), 0, true, CurrentNs() - startNs);

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
    if (failed != 0) { LogTaskResult(result, "delete"); }

    // 请求明细日志（每任务一行聚合）
    WriteDeleteLog(request, notFound, failed);
    LOG_INFO("Executor: delete {} keys, {} not found, {} failed", keys.size(), notFound, failed);
    return true;
}

void Executor::RecordTask(OpType op, std::size_t entryCount, std::uint64_t bytes,
                          bool taskCompleted, std::int64_t latencyNs)
{
    // 全量口径（请求发出即计，含未命中与失败）
    opCount_.fetch_add(entryCount, std::memory_order_relaxed);
    // 流量为成功 entry 口径
    ioBytes_.fetch_add(bytes, std::memory_order_relaxed);
    // 分接口桶同款口径
    const std::size_t bucket = static_cast<std::size_t>(op);
    opCountByOp_[bucket].fetch_add(entryCount, std::memory_order_relaxed);
    ioBytesByOp_[bucket].fetch_add(bytes, std::memory_order_relaxed);
    if (taskCompleted) {
        taskCount_.fetch_add(1, std::memory_order_relaxed);
        latencyNsTotal_.fetch_add(static_cast<std::uint64_t>(latencyNs), std::memory_order_relaxed);
        latencyNsTotalByOp_[bucket].fetch_add(static_cast<std::uint64_t>(latencyNs),
                                              std::memory_order_relaxed);
        taskCountByOp_[bucket].fetch_add(1, std::memory_order_relaxed);
        // 完成样本入环形（微秒），分位统计用；零时延样本跳过
        if (latencyNs > 0) {
            const std::lock_guard<std::mutex> lock(samplesMutex_);
            std::vector<std::uint32_t> &buf = latencySamples_[bucket];
            buf[latencySamplePos_[bucket]] = static_cast<std::uint32_t>(latencyNs / 1000);
            latencySamplePos_[bucket] = (latencySamplePos_[bucket] + 1) % buf.size();
            if (latencySampleCount_[bucket] < buf.size()) { ++latencySampleCount_[bucket]; }
        }
    }
}

bool Executor::IsNotFound(const kv::Status &status)
{
    // 全部“目标 key 无数据”语义的状态码统一归未命中观察
    switch (status.code) {
        case kv::StatusCode::NOT_FOUND:
        case kv::StatusCode::ENTRY_KEY_NOT_FOUND:
        case kv::StatusCode::ENTRY_DATA_NOT_EXIST:
        case kv::StatusCode::ENTRY_KEY_NOT_EXIST:
        case kv::StatusCode::CQE_KEY_NOT_EXIST: return true;
        default: return false;
    }
}

std::size_t Executor::CountFailed(const kv::TaskResult &result, std::size_t entryCount)
{
    // entry 级分类优先；entryStatus 为空（任务未到 entry 层）时按任务级状态整体归类
    if (!result.entryStatus.empty()) {
        std::size_t failed = 0;
        const std::size_t count = std::min(result.entryStatus.size(), entryCount);
        for (std::size_t i = 0; i < count; ++i) {
            if (!result.entryStatus[i].ok()) { ++failed; }
        }
        // entryStatus 数量不足的异常场景，缺失部分按失败计
        if (result.entryStatus.size() < entryCount) {
            failed += entryCount - result.entryStatus.size();
        }
        return std::min(failed, entryCount);
    }
    if (!result.status.ok()) { return entryCount; }
    return 0;
}

std::uint64_t Executor::SumBytes(const std::vector<kv::KVBuffer> &entries)
{
    std::uint64_t bytes = 0;
    for (const auto &entry : entries) { bytes += entry.buffer.region.size; }
    return bytes;
}

std::uint64_t Executor::SumOkBytes(const kv::TaskResult &result,
                                   const std::vector<kv::KVBuffer> &entries)
{
    // entryStatus 缺失：任务级成功视为全部成功，失败视为全部失败
    if (result.entryStatus.empty()) { return result.status.ok() ? SumBytes(entries) : 0; }
    std::uint64_t bytes = 0;
    const std::size_t count = std::min(result.entryStatus.size(), entries.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (result.entryStatus[i].ok()) { bytes += entries[i].buffer.region.size; }
    }
    return bytes;
}
