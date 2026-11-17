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
std::int64_t CurrentNs()
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

}  // namespace

Executor::Executor()
{
    // 分接口原子桶清零（atomic 数组不能就地默认初始化）
    for (std::size_t i = 0; i < kOpTypeCount; ++i) {
        opCountByOp_[i].store(0, std::memory_order_relaxed);
        ioBytesByOp_[i].store(0, std::memory_order_relaxed);
    }
}

void Executor::SetClient(KvClientUtil *client) { client_ = client; }

void Executor::SetAllocator(DeviceAllocator *allocator) { allocator_ = allocator; }

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
    metrics.mismatchCount = mismatchCount_.load(std::memory_order_relaxed);
    metrics.ioBytes = ioBytes_.load(std::memory_order_relaxed);
    metrics.taskCount = taskCount_.load(std::memory_order_relaxed);
    metrics.latencyNsTotal = latencyNsTotal_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < kOpTypeCount; ++i) {
        metrics.opCountByOp[i] = opCountByOp_[i].load(std::memory_order_relaxed);
        metrics.ioBytesByOp[i] = ioBytesByOp_[i].load(std::memory_order_relaxed);
    }
    return metrics;
}

bool Executor::QueryAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::CacheKey> &keys = request.keys;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->QueryAsync(keys, taskId, err)) {
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

    // 任务级失败或结果缺失：观察记录（errorCount），不算工具链失败（无可信逐 key 数据，不写明细）
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
        if (result.queryResult->exists[i] != 0) { ++found; }
    }
    notFoundCount_.fetch_add(keys.size() - found, std::memory_order_relaxed);

    // 输出 exist 任务明细日志（逐 key 存在性 + prefix_hit_len；perf=off 全量、perf=on 仅错误行）
    WriteExistLog(request, result);
    LOG_INFO("Executor: query {} keys, {} found, {} not found", keys.size(), found,
             keys.size() - found);
    return true;
}

bool Executor::StoreAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::KVBuffer> &entries = request.entries;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->StoreAsync(entries, taskId, err)) {
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

    // 输出 put 任务明细日志（写入后读回缓冲实际内容，顺带验证 H2D 写入生效；
    // perf=off 全量、perf=on 仅错误行）；读回失败属工具链异常（内存访问链路不可用），
    // 与 get 明细日志同一语义
    if (!WritePutGetLog(request, result, "put", err)) { return false; }
    LOG_INFO("Executor: stored {} entries, {} failed", entries.size(), failed);
    return true;
}

bool Executor::LoadAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::KVBuffer> &entries = request.entries;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->LoadAsync(entries, taskId, err)) {
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
    // 未命中系计 notFound、有基准逐字节比对不一致计 mismatch、其他计 error；
    // 流量为成功 entry 口径（ok 的 entry 累计 region.size，失败/未命中计 0 字节）
    std::uint64_t notFound = 0;
    std::uint64_t mismatch = 0;
    std::uint64_t failed = 0;
    std::uint64_t successBytes = 0;
    if (!result.entryStatus.empty()) {
        const std::size_t count = std::min(result.entryStatus.size(), entries.size());
        for (std::size_t i = 0; i < count; ++i) {
            if (result.entryStatus[i].ok()) {
                successBytes += entries[i].buffer.region.size;
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
    } else {
        // entryStatus 为空且任务级成功：全部 entry 视为成功
        successBytes = SumBytes(entries);
    }
    RecordTask(OpType::LOAD, entries.size(), successBytes, true, CurrentNs() - startNs);
    notFoundCount_.fetch_add(notFound, std::memory_order_relaxed);
    mismatchCount_.fetch_add(mismatch, std::memory_order_relaxed);
    errorCount_.fetch_add(failed, std::memory_order_relaxed);
    if (failed != 0) { LogTaskResult(result, "load"); }

    // 输出 get 任务明细日志（读回缓冲实际内容，供人工校验一致性；perf=off 全量、
    // perf=on 仅错误行）；读回失败属工具链异常（内存访问链路不可用），与比对路径同一语义
    if (!WritePutGetLog(request, result, "get", err)) { return false; }

    LOG_INFO("Executor: load {} entries, {} not found, {} mismatch, {} failed", entries.size(),
             notFound, mismatch, failed);
    return true;
}

bool Executor::WritePutGetLog(const Request &request, const kv::TaskResult &result,
                              const char *opcode, std::string &err)
{
    // 请求明细日志未初始化（异常场景）：不组装内容，直接返回
    if (LoggerManager::GetReqLogger() == nullptr) { return true; }
    // perf=on 仅记录错误行（entryCode 非 0 的 entry），全量明细留给 perf=off
    const bool errorOnly = LoggerManager::IsReqLogErrorOnly();
    // batchCode：任务级返回码（任务整体一个值，每行重复保持行自包含）
    const int batchCode = static_cast<int>(result.status.code);

    const std::vector<kv::KVBuffer> &entries = request.entries;

    // 每 entry 一行，字段名(值) 形式标注（裸值无法分辨字段）：
    // opcode(op) nsid(x) batchCode(b) entryCode(c) key(k) key_low(l) key_high(h) offset(o)
    // len(n) value(v) tail(t)——put / get 同款字段（opcode 区分）；key 为 input 原文 key，
    // key_low / key_high 为 128 位 key 低 64 位 / 高 16 位的十进制（分开显示）；
    // 行格式与全量日志一致（请求明细 logger 配置同款，pattern 沿用 log.pattern）
    for (std::size_t i = 0; i < entries.size(); ++i) {
        // entryCode：该 entry 自身返回码（与任务级 batchCode 可能不同，如 batch 部分失败
        // 时任务级为 PARTIAL_FAILED 而 entry 级各不相同）；entryStatus 缺失时以任务级代表
        const int entryCode = (i < result.entryStatus.size())
                                  ? static_cast<int>(result.entryStatus[i].code)
                                  : batchCode;
        if (errorOnly && entryCode == 0) { continue; }

        // key 分量显示：input 原文 key 与 key_low / key_high 十进制分开
        std::uint64_t keyLow = 0;
        if (i < request.keys.size()) {
            std::memcpy(&keyLow, request.keys[i].data(), sizeof(keyLow));
        }
        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string rawKey =
            (i < request.rawKeys.size()) ? request.rawKeys[i] : std::string("-");
        const std::string keyLowStr = std::to_string(keyLow);
        const std::string keyHighStr = std::to_string(keyHigh);
        const std::string offsetStr = std::to_string(entries[i].offset);
        // len：key_low.size + offset.size（两个十进制串的位数和，纯显示用途）
        const std::size_t len = keyLowStr.size() + offsetStr.size();

        // entry 完成状态：entryStatus 缺失时以任务级状态代表全部 entry
        const bool entryOk =
            (i < result.entryStatus.size()) ? result.entryStatus[i].ok() : result.status.ok();

        // 成功的 entry 才读回缓冲实际内容：首段长度按 prefixLens[i]（mode1 读生成前缀
        // 部分、mode2 读完整块，见 buildTaskRequest；不超过块实际大小）、尾部固定 8 字节，
        // 经分配器 D2H 读回（device 语义缓冲不能直接解引用；put 读回即验证 H2D 写入生效，
        // get 读回即读回校验）；失败或空块以 "-" 占位
        std::string value = "-";
        std::string tail = "-";
        if (entryOk) {
            const kv::MemoryRegion &region = entries[i].buffer.region;
            const std::size_t blockLen = static_cast<std::size_t>(region.size);
            const std::size_t prefixLen =
                (i < request.prefixLens.size()) ? request.prefixLens[i] : 0;
            const std::size_t headLen = std::min(prefixLen, blockLen);
            const std::size_t tailLen = std::min<std::size_t>(8, blockLen);
            if (headLen != 0 || tailLen != 0) {
                if (allocator_ == nullptr) {
                    err = "req log readback requires a device allocator bound";
                    return false;
                }
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
            if (headLen != 0) { value.assign(head.begin(), head.end()); }
            if (tailLen != 0) { tail.assign(tailBuf.begin(), tailBuf.end()); }
        }

        // 内容作为 fmt 参数而非格式串（"{}" 显式转义）：value/tail 为二进制转字符串，
        // 可能含 '{' '}' 字节，作为格式串会被 fmt 误解析
        const std::string line =
            "opcode(" + std::string(opcode) + ") nsid(" + std::to_string(request.nsid) +
            ") batchCode(" + std::to_string(batchCode) + ") entryCode(" +
            std::to_string(entryCode) + ") key(" + rawKey + ") key_low(" + keyLowStr +
            ") key_high(" + keyHighStr + ") offset(" + offsetStr + ") len(" + std::to_string(len) +
            ") value(" + value + ") tail(" + tail + ")";
        LOG_INFO_REQ("{}", line);
    }
    return true;
}

void Executor::WriteExistLog(const Request &request, const kv::TaskResult &result)
{
    // 请求明细日志未初始化（异常场景）：不组装内容，直接返回
    if (LoggerManager::GetReqLogger() == nullptr) { return; }
    // perf=on 仅记录错误行（exist 无逐 entry 状态，按任务级返回码过滤），全量明细留给 perf=off
    const bool errorOnly = LoggerManager::IsReqLogErrorOnly();
    const int batchCode = static_cast<int>(result.status.code);
    if (errorOnly && batchCode == 0) { return; }

    // 每 key 一行，字段名(值) 形式标注：opcode(exist) nsid(x) batchCode(b) key(k) key_low(l)
    // key_high(h) exists(e) prefix_hit_len(n)——batchCode 为任务级返回码（exist 无逐
    // entry 状态，不显示 entryCode），key 为 input 原文 key，key_low / key_high 分开显示；
    // prefix_hit_len 为 kv client 返回的前缀命中 key 数（QueryResult::prefixHitKeys，
    // 任务级单值，逐行重复保持行自包含）；exist 无 value 载荷，不读回缓冲
    // （调用侧已保证 queryResult 有值且 exists 尺寸与 keys 匹配）
    for (std::size_t i = 0; i < request.keys.size(); ++i) {
        std::uint64_t keyLow = 0;
        std::memcpy(&keyLow, request.keys[i].data(), sizeof(keyLow));
        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string rawKey =
            (i < request.rawKeys.size()) ? request.rawKeys[i] : std::string("-");
        const std::uint8_t exists =
            (i < result.queryResult->exists.size()) ? result.queryResult->exists[i] : 0;
        const std::string line = "opcode(exist) nsid(" + std::to_string(request.nsid) +
                                 ") batchCode(" + std::to_string(batchCode) + ") key(" + rawKey +
                                 ") key_low(" + std::to_string(keyLow) + ") key_high(" +
                                 std::to_string(keyHigh) + ") exists(" +
                                 std::to_string(static_cast<int>(exists)) + ") prefix_hit_len(" +
                                 std::to_string(result.queryResult->prefixHitKeys) + ")";
        LOG_INFO_REQ("{}", line);
    }
}

void Executor::WriteDeleteLog(const Request &request, const kv::TaskResult &result)
{
    // 请求明细日志未初始化（异常场景）：不组装内容，直接返回
    if (LoggerManager::GetReqLogger() == nullptr) { return; }
    // perf=on 仅记录错误行（entryCode 非 0 的 key），全量明细留给 perf=off
    const bool errorOnly = LoggerManager::IsReqLogErrorOnly();
    const int batchCode = static_cast<int>(result.status.code);

    // 每 key 一行，字段名(值) 形式标注：opcode(delete) nsid(x) batchCode(b) entryCode(c)
    // key(k) key_low(l) key_high(h)——batchCode 为任务级返回码、entryCode 为该 key 自身
    // 返回码；delete 无 value 载荷（无 offset/len/value/tail 字段）
    for (std::size_t i = 0; i < request.keys.size(); ++i) {
        const int entryCode = (i < result.entryStatus.size())
                                  ? static_cast<int>(result.entryStatus[i].code)
                                  : batchCode;
        if (errorOnly && entryCode == 0) { continue; }
        std::uint64_t keyLow = 0;
        std::memcpy(&keyLow, request.keys[i].data(), sizeof(keyLow));
        const std::uint16_t keyHigh = (i < request.keyHighs.size()) ? request.keyHighs[i] : 0;
        const std::string rawKey =
            (i < request.rawKeys.size()) ? request.rawKeys[i] : std::string("-");
        const std::string line = "opcode(delete) nsid(" + std::to_string(request.nsid) +
                                 ") batchCode(" + std::to_string(batchCode) + ") entryCode(" +
                                 std::to_string(entryCode) + ") key(" + rawKey + ") key_low(" +
                                 std::to_string(keyLow) + ") key_high(" + std::to_string(keyHigh) +
                                 ")";
        LOG_INFO_REQ("{}", line);
    }
}

bool Executor::DeleteAndRecord(const Request &request, std::string &err)
{
    const std::vector<kv::CacheKey> &keys = request.keys;
    kv::TaskId taskId = kv::kInvalidTaskId;
    const std::int64_t startNs = CurrentNs();
    if (!client_->DeleteAsync(keys, taskId, err)) {
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

    // 请求明细日志（delete 无 value 载荷，每 key 一行返回码；perf=on 仅错误行）
    WriteDeleteLog(request, result);
    LOG_INFO("Executor: delete {} keys, {} not found, {} failed", keys.size(), notFound, failed);
    return true;
}

void Executor::RecordTask(OpType op, std::size_t entryCount, std::uint64_t bytes,
                          bool taskCompleted, std::int64_t latencyNs)
{
    // 全量口径（请求发出即计，含未命中与失败；error rate 分母依赖此口径）
    opCount_.fetch_add(entryCount, std::memory_order_relaxed);
    // 流量为成功 entry 口径（调用方已按成功过滤，失败/未命中传 0）
    ioBytes_.fetch_add(bytes, std::memory_order_relaxed);
    // 分接口桶（opCount 全量、ioBytes 成功口径，与总量同款）
    const std::size_t bucket = static_cast<std::size_t>(op);
    opCountByOp_[bucket].fetch_add(entryCount, std::memory_order_relaxed);
    ioBytesByOp_[bucket].fetch_add(bytes, std::memory_order_relaxed);
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
        case kv::StatusCode::CQE_KEY_NOT_EXIST: return true;
        default: return false;
    }
}

std::size_t Executor::CountFailed(const kv::TaskResult &result, std::size_t entryCount)
{
    if (!result.status.ok()) { return entryCount; }
    std::size_t failed = 0;
    for (const auto &status : result.entryStatus) {
        if (!status.ok()) { ++failed; }
    }
    if (result.entryStatus.size() < entryCount) {
        failed += entryCount - result.entryStatus.size();
    }
    return std::min(failed, entryCount);
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
    // entryStatus 缺失：任务级成功视为全部 entry 成功，失败视为全部 0 字节
    if (result.entryStatus.empty()) { return result.status.ok() ? SumBytes(entries) : 0; }
    std::uint64_t bytes = 0;
    const std::size_t count = std::min(result.entryStatus.size(), entries.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (result.entryStatus[i].ok()) { bytes += entries[i].buffer.region.size; }
    }
    return bytes;
}
