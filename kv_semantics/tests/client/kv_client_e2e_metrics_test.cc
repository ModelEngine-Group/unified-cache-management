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
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "kv_client_impl.h"

namespace UC::ASU {
namespace {

CacheKey MakeCacheKey(std::string_view text)
{
    CacheKey key{};
    if (text.size() <= key.size()) {
        if (!text.empty()) { std::memcpy(key.data(), text.data(), text.size()); }
        return key;
    }
    const auto hash = std::hash<std::string_view>{}(text);
    std::memcpy(key.data(), &hash, key.size());
    return key;
}

struct CacheKeyHasher {
    std::size_t operator()(const CacheKey& key) const
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (auto byte : key) {
            hash ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(byte));
            hash *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

enum class OperationKind {
    QUERY,
    LOAD,
    STORE,
    DELETE,
};

const char* OperationName(OperationKind kind)
{
    switch (kind) {
        case OperationKind::QUERY: return "Query";
        case OperationKind::LOAD: return "Load";
        case OperationKind::STORE: return "Store";
        case OperationKind::DELETE: return "Delete";
        default: return "Unknown";
    }
}

struct OperationSample {
    OperationKind kind{OperationKind::QUERY};
    std::uint64_t latencyNs{0};
    std::uint64_t bytes{0};
    bool success{false};
    bool correct{false};
};

struct OperationSummary {
    std::size_t total{0};
    std::size_t succeeded{0};
    std::size_t correct{0};
    std::uint64_t bytes{0};
    std::uint64_t latencyAvgNs{0};
    std::uint64_t latencyP50Ns{0};
    std::uint64_t latencyP95Ns{0};
    std::uint64_t latencyP99Ns{0};
    std::uint64_t latencyMaxNs{0};
    double iops{0.0};
    double bandwidthBytesPerSec{0.0};
    double successRate{0.0};
    double correctnessRate{0.0};
};

class MetricsRecorder {
public:
    void Record(OperationKind kind, std::uint64_t latencyNs, std::uint64_t bytes, bool success,
                bool correct)
    {
        samples_.push_back(
            OperationSample{kind, std::max<std::uint64_t>(latencyNs, 1), bytes, success, correct});
    }

    OperationSummary Summarize(OperationKind kind) const
    {
        OperationSummary summary;
        std::vector<std::uint64_t> latencies;
        std::uint64_t totalLatencyNs = 0;
        for (const auto& sample : samples_) {
            if (sample.kind != kind) { continue; }
            ++summary.total;
            if (sample.success) { ++summary.succeeded; }
            if (sample.correct) { ++summary.correct; }
            summary.bytes += sample.bytes;
            totalLatencyNs += sample.latencyNs;
            latencies.emplace_back(sample.latencyNs);
        }

        if (summary.total == 0) { return summary; }

        std::sort(latencies.begin(), latencies.end());
        summary.latencyAvgNs = totalLatencyNs / summary.total;
        summary.latencyP50Ns = Percentile(latencies, 50);
        summary.latencyP95Ns = Percentile(latencies, 95);
        summary.latencyP99Ns = Percentile(latencies, 99);
        summary.latencyMaxNs = latencies.back();
        summary.iops = static_cast<double>(summary.succeeded) * 1000000000.0 /
                       static_cast<double>(totalLatencyNs);
        summary.bandwidthBytesPerSec =
            static_cast<double>(summary.bytes) * 1000000000.0 / static_cast<double>(totalLatencyNs);
        summary.successRate =
            static_cast<double>(summary.succeeded) / static_cast<double>(summary.total);
        summary.correctnessRate =
            static_cast<double>(summary.correct) / static_cast<double>(summary.total);
        return summary;
    }

    void PrintReport(const std::string& name) const
    {
        std::cout << "[AsuClientE2E] " << name << '\n';
        for (auto kind : {OperationKind::STORE, OperationKind::QUERY, OperationKind::LOAD,
                          OperationKind::DELETE}) {
            const auto summary = Summarize(kind);
            if (summary.total == 0) { continue; }
            std::cout << "  " << OperationName(kind) << ": total=" << summary.total
                      << " success_rate=" << summary.successRate
                      << " correctness_rate=" << summary.correctnessRate << " iops=" << summary.iops
                      << " bandwidth_Bps=" << summary.bandwidthBytesPerSec
                      << " latency_avg_ns=" << summary.latencyAvgNs
                      << " latency_p50_ns=" << summary.latencyP50Ns
                      << " latency_p95_ns=" << summary.latencyP95Ns
                      << " latency_p99_ns=" << summary.latencyP99Ns
                      << " latency_max_ns=" << summary.latencyMaxNs << '\n';
        }
    }

private:
    static std::uint64_t Percentile(const std::vector<std::uint64_t>& latencies,
                                    std::uint32_t percentile)
    {
        if (latencies.empty()) { return 0; }
        const auto rank = (static_cast<std::uint64_t>(latencies.size()) * percentile + 99U) / 100U;
        const auto index = static_cast<std::size_t>(std::max<std::uint64_t>(rank, 1) - 1U);
        return latencies[std::min(index, latencies.size() - 1)];
    }

    std::vector<OperationSample> samples_;
};

struct ResourceSnapshot {
    std::size_t storedKeyCount{0};
    std::uint64_t storedBytes{0};
    std::size_t taskCount{0};
    std::size_t createdTransports{0};
    std::size_t shutdownTransports{0};
    std::clock_t cpuTicks{0};
};

struct ClusterState {
    mutable std::mutex mutex;
    std::unordered_set<AsuId> activeAsus;
    std::unordered_map<AsuId,
                       std::unordered_map<CacheKey, std::vector<std::uint8_t>, CacheKeyHasher>>
        stores;
    std::unordered_map<TaskId, TaskResult> tasks;
    std::unordered_map<AsuId, std::size_t> queryCalls;
    std::unordered_map<AsuId, std::size_t> loadCalls;
    std::unordered_map<AsuId, std::size_t> storeCalls;
    std::unordered_map<AsuId, std::size_t> deleteCalls;
    TaskId nextTaskId{1};
    std::size_t createdTransports{0};
    std::size_t shutdownTransports{0};
    std::size_t forcedQueryFailures{0};

    void SetActiveAsus(const std::vector<AsuId>& asuIds)
    {
        std::lock_guard<std::mutex> lock{mutex};
        activeAsus.clear();
        activeAsus.insert(asuIds.begin(), asuIds.end());
    }

    void ForceNextQueryFailure()
    {
        std::lock_guard<std::mutex> lock{mutex};
        ++forcedQueryFailures;
    }

    void ClearOperationCalls()
    {
        std::lock_guard<std::mutex> lock{mutex};
        queryCalls.clear();
        loadCalls.clear();
        storeCalls.clear();
        deleteCalls.clear();
    }

    ResourceSnapshot Snapshot() const
    {
        std::lock_guard<std::mutex> lock{mutex};
        ResourceSnapshot snapshot;
        snapshot.taskCount = tasks.size();
        snapshot.createdTransports = createdTransports;
        snapshot.shutdownTransports = shutdownTransports;
        snapshot.cpuTicks = std::clock();
        for (const auto& asuStore : stores) {
            snapshot.storedKeyCount += asuStore.second.size();
            for (const auto& item : asuStore.second) { snapshot.storedBytes += item.second.size(); }
        }
        return snapshot;
    }

    std::size_t TouchedAsuCount() const
    {
        std::lock_guard<std::mutex> lock{mutex};
        std::unordered_set<AsuId> touched;
        for (const auto& item : queryCalls) { touched.insert(item.first); }
        for (const auto& item : loadCalls) { touched.insert(item.first); }
        for (const auto& item : storeCalls) { touched.insert(item.first); }
        for (const auto& item : deleteCalls) { touched.insert(item.first); }
        return touched.size();
    }

    std::size_t OperationCalls(AsuId asuId) const
    {
        std::lock_guard<std::mutex> lock{mutex};
        return CountCalls(queryCalls, asuId) + CountCalls(loadCalls, asuId) +
               CountCalls(storeCalls, asuId) + CountCalls(deleteCalls, asuId);
    }

private:
    static std::size_t CountCalls(const std::unordered_map<AsuId, std::size_t>& calls, AsuId asuId)
    {
        auto iter = calls.find(asuId);
        return iter == calls.end() ? 0 : iter->second;
    }
};

TaskResult BuildTaskResult(const std::vector<Status>& entryStatus)
{
    const bool anyFailed = std::any_of(entryStatus.begin(), entryStatus.end(),
                                       [](const Status& status) { return !status.ok(); });
    TaskResult result;
    result.status =
        anyFailed ? Status::Error(StatusCode::PARTIAL_FAILED, "one or more fake ASU entries failed")
                  : Status::OK();
    result.entryStatus = entryStatus;
    return result;
}

Status SubmitCompletedTask(ClusterState& state, TaskResult result, TaskId& taskId)
{
    taskId = state.nextTaskId++;
    state.tasks.emplace(taskId, std::move(result));
    return Status::OK();
}

bool IsBufferUsable(const MemoryRegion& region) { return region.size == 0 || region.addr != 0; }

class InMemoryAsuTransport final : public AsuTransport {
public:
    explicit InMemoryAsuTransport(std::shared_ptr<ClusterState> state) : state_(std::move(state)) {}

    Status Init(const TransportConfig& config,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)transProvider;
        std::lock_guard<std::mutex> lock{state_->mutex};
        config_ = config;
        initialized_ = true;
        ++state_->createdTransports;
        return Status::OK();
    }

    Status Init(const std::string& configPath,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)configPath;
        (void)transProvider;
        return Status::Error(StatusCode::UNSUPPORTED,
                             "in-memory ASU transport config path is unsupported");
    }

    Status Shutdown() override
    {
        std::lock_guard<std::mutex> lock{state_->mutex};
        if (initialized_) { ++state_->shutdownTransports; }
        initialized_ = false;
        return Status::OK();
    }

    Status CheckHealth() override
    {
        return initialized_ ? Status::OK()
                            : Status::Error(StatusCode::NOT_INITIALIZED,
                                            "fake ASU transport is not initialized");
    }

    Status RunQuery(BatchView<CacheKey> keys, QueryResult& result)
    {
        std::lock_guard<std::mutex> lock{state_->mutex};
        auto status = CheckReadyLocked();
        if (!status.ok()) { return status; }
        if (state_->forcedQueryFailures > 0) {
            --state_->forcedQueryFailures;
            return Status::Error(StatusCode::CONNECTION_ERROR, "forced stale-view query failure");
        }

        ++state_->queryCalls[config_.asuId];
        const auto& store = state_->stores[config_.asuId];
        result.exists.assign(keys.size, 0);
        result.prefixHitKeys = 0;
        for (std::size_t index = 0; index < keys.size; ++index) {
            result.exists[index] = store.find(keys[index]) == store.end() ? 0 : 1;
        }
        return Status::OK();
    }

    Status Submit(const TransportTaskPtr& task) override
    {
        if (!task) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "in-memory transport task is null");
        }
        switch (task->opType) {
            case AsuOpType::QUERY:
                return SubmitQuery({task->keys.data(), task->keys.size()}, task->taskId,
                                   std::move(task->onComplete));
            case AsuOpType::LOAD:
            case AsuOpType::BATCH_LOAD:
                return SubmitLoad({task->entries.data(), task->entries.size()}, task->taskId,
                                  std::move(task->onComplete));
            case AsuOpType::STORE:
            case AsuOpType::BATCH_STORE:
                return SubmitStore({task->entries.data(), task->entries.size()}, task->taskId,
                                   std::move(task->onComplete));
            case AsuOpType::DELETE:
                return SubmitDelete({task->keys.data(), task->keys.size()}, task->taskId,
                                    std::move(task->onComplete));
            default:
                task->taskId = kInvalidTaskId;
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "unsupported in-memory transport operation");
        }
    }

    Status SubmitQuery(BatchView<CacheKey> keys, TaskId& taskId, TaskCompletionCallback onComplete)
    {
        QueryResult queryResult;
        auto status = RunQuery(keys, queryResult);
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }
        std::lock_guard<std::mutex> lock{state_->mutex};
        TaskResult result;
        result.status = Status::OK();
        result.entryStatus.assign(keys.size, Status::OK());
        result.queryResult = queryResult;
        status = SubmitCompletedTask(*state_, result, taskId);
        if (status.ok() && onComplete) { onComplete(std::move(result)); }
        return status;
    }

    Status SubmitLoad(BatchView<KVBuffer> entries, TaskId& taskId,
                      TaskCompletionCallback onComplete)
    {
        std::lock_guard<std::mutex> lock{state_->mutex};
        auto status = CheckReadyLocked();
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }

        ++state_->loadCalls[config_.asuId];
        std::vector<Status> entryStatus;
        entryStatus.reserve(entries.size);
        auto& store = state_->stores[config_.asuId];
        for (std::size_t index = 0; index < entries.size; ++index) {
            const auto& entry = entries[index];
            auto iter = store.find(entry.key);
            if (iter == store.end()) {
                entryStatus.emplace_back(
                    Status::Error(StatusCode::NOT_FOUND, "fake ASU key not found"));
                continue;
            }
            if (!IsBufferUsable(entry.buffer.region) ||
                entry.buffer.region.size < iter->second.size()) {
                entryStatus.emplace_back(Status::Error(StatusCode::INVALID_ARGUMENT,
                                                       "fake ASU load buffer is too small"));
                continue;
            }

            auto* data = reinterpret_cast<std::uint8_t*>(entry.buffer.region.addr);
            std::copy(iter->second.begin(), iter->second.end(), data);
            entryStatus.emplace_back(Status::OK());
        }
        auto result = BuildTaskResult(entryStatus);
        status = SubmitCompletedTask(*state_, result, taskId);
        if (status.ok() && onComplete) { onComplete(std::move(result)); }
        return status;
    }

    Status SubmitStore(BatchView<KVBuffer> entries, TaskId& taskId,
                       TaskCompletionCallback onComplete)
    {
        std::lock_guard<std::mutex> lock{state_->mutex};
        auto status = CheckReadyLocked();
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }

        ++state_->storeCalls[config_.asuId];
        std::vector<Status> entryStatus;
        entryStatus.reserve(entries.size);
        auto& store = state_->stores[config_.asuId];
        for (std::size_t index = 0; index < entries.size; ++index) {
            const auto& entry = entries[index];
            if (!IsBufferUsable(entry.buffer.region)) {
                entryStatus.emplace_back(Status::Error(StatusCode::INVALID_ARGUMENT,
                                                       "fake ASU store buffer is invalid"));
                continue;
            }

            const auto* data = reinterpret_cast<const std::uint8_t*>(entry.buffer.region.addr);
            store[entry.key] = std::vector<std::uint8_t>(data, data + entry.buffer.region.size);
            entryStatus.emplace_back(Status::OK());
        }
        auto result = BuildTaskResult(entryStatus);
        status = SubmitCompletedTask(*state_, result, taskId);
        if (status.ok() && onComplete) { onComplete(std::move(result)); }
        return status;
    }

    Status SubmitDelete(BatchView<CacheKey> keys, TaskId& taskId, TaskCompletionCallback onComplete)
    {
        std::lock_guard<std::mutex> lock{state_->mutex};
        auto status = CheckReadyLocked();
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }

        ++state_->deleteCalls[config_.asuId];
        std::vector<Status> entryStatus;
        entryStatus.reserve(keys.size);
        auto& store = state_->stores[config_.asuId];
        for (std::size_t index = 0; index < keys.size; ++index) {
            const auto& key = keys[index];
            if (store.erase(key) == 0) {
                entryStatus.emplace_back(
                    Status::Error(StatusCode::NOT_FOUND, "fake ASU key not found"));
            } else {
                entryStatus.emplace_back(Status::OK());
            }
        }
        auto result = BuildTaskResult(entryStatus);
        status = SubmitCompletedTask(*state_, result, taskId);
        if (status.ok() && onComplete) { onComplete(std::move(result)); }
        return status;
    }

    Status Cancel(TaskId) override { return Status::Error(StatusCode::UNSUPPORTED, "unsupported"); }

private:
    Status CheckReadyLocked() const
    {
        if (!initialized_) {
            return Status::Error(StatusCode::NOT_INITIALIZED,
                                 "fake ASU transport is not initialized");
        }
        if (state_->activeAsus.find(config_.asuId) == state_->activeAsus.end()) {
            return Status::Error(StatusCode::CONNECTION_ERROR,
                                 "fake ASU has been removed from active view");
        }
        return Status::OK();
    }

    std::shared_ptr<ClusterState> state_;
    TransportConfig config_;
    bool initialized_{false};
};

class DynamicViewServer final : public ViewServer {
public:
    DynamicViewServer(std::shared_ptr<ClusterState> state, std::vector<AsuId> asuIds)
        : state_(std::move(state)), asuIds_(std::move(asuIds))
    {
        state_->SetActiveAsus(asuIds_);
    }

    Status GetGlobalView(GlobalView& view) override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        view = GlobalView{};
        view.viewEpoch = epoch_;
        for (auto asuId : asuIds_) { view.asuMap.emplace(asuId, AsuInfo{}); }
        ++fetchCount_;
        return Status::OK();
    }

    void Publish(std::vector<AsuId> asuIds)
    {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            asuIds_ = std::move(asuIds);
            ++epoch_;
        }
        state_->SetActiveAsus(AsuIds());
    }

    std::size_t FetchCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return fetchCount_;
    }

private:
    std::vector<AsuId> AsuIds() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return asuIds_;
    }

    std::shared_ptr<ClusterState> state_;
    mutable std::mutex mutex_;
    std::vector<AsuId> asuIds_;
    std::uint64_t epoch_{1};
    std::size_t fetchCount_{0};
};

AsuClientConfig MakeClientConfig(const std::vector<AsuId>& allAsuIds)
{
    AsuClientConfig config;
    config.clientId = "asu-client-e2e-metrics-test";
    config.defaultWaitTimeoutMs = 1000;
    for (auto asuId : allAsuIds) {
        TransportConfig transportConfig;
        transportConfig.asuId = asuId;
        transportConfig.asuName = "fake-asu-" + std::to_string(asuId);
        transportConfig.providerType = TransProviderType::FAKE;
        transportConfig.maxInflightTasks = 1024;
        config.transportConfigs.emplace_back(std::move(transportConfig));
    }
    return config;
}

ViewServerFactory MakeViewServerFactory(const std::shared_ptr<ViewServer>& viewServer)
{
    return [viewServer](const AsuClientConfig&) { return viewServer; };
}

TransportFactory MakeFactory(const std::shared_ptr<ClusterState>& state)
{
    return [state] { return std::make_unique<InMemoryAsuTransport>(state); };
}

std::vector<std::uint8_t> MakePayload(std::size_t size, std::uint8_t seed)
{
    std::vector<std::uint8_t> payload(size);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::uint8_t>(seed + static_cast<std::uint8_t>(index % 251U));
    }
    return payload;
}

KVBuffer MakeBuffer(const CacheKey& key, std::vector<std::uint8_t>& payload)
{
    MemoryRegion region;
    region.memoryType = MemoryType::HOST;
    region.addr = payload.empty() ? 0 : reinterpret_cast<std::uint64_t>(payload.data());
    region.size = payload.size();
    return KVBuffer{
        key, Buffer{region, kInvalidMRHandle}
    };
}

std::vector<KVBuffer> MakeStoreEntries(const std::string& prefix, std::size_t count,
                                       std::size_t baseSize,
                                       std::vector<std::vector<std::uint8_t>>& payloads)
{
    payloads.clear();
    payloads.reserve(count);
    std::vector<KVBuffer> entries;
    entries.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        payloads.emplace_back(
            MakePayload(baseSize + (index % 5U) * 17U, static_cast<std::uint8_t>(index + 1U)));
        entries.emplace_back(
            MakeBuffer(MakeCacheKey(prefix + std::to_string(index)), payloads.back()));
    }
    return entries;
}

std::vector<KVBuffer> MakeLoadEntries(const std::vector<KVBuffer>& storeEntries,
                                      const std::vector<std::vector<std::uint8_t>>& expected,
                                      std::vector<std::vector<std::uint8_t>>& buffers)
{
    buffers.clear();
    buffers.reserve(expected.size());
    std::vector<KVBuffer> entries;
    entries.reserve(storeEntries.size());
    for (std::size_t index = 0; index < storeEntries.size(); ++index) {
        buffers.emplace_back(expected[index].size(), 0);
        entries.emplace_back(MakeBuffer(storeEntries[index].key, buffers.back()));
    }
    return entries;
}

std::vector<CacheKey> ExtractKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) { keys.emplace_back(entry.key); }
    return keys;
}

std::uint64_t SumPayloadBytes(const std::vector<std::vector<std::uint8_t>>& payloads)
{
    return std::accumulate(
        payloads.begin(), payloads.end(), std::uint64_t{0},
        [](std::uint64_t sum, const auto& payload) { return sum + payload.size(); });
}

std::uint64_t ElapsedNs(std::chrono::steady_clock::time_point start)
{
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
}

bool EntryStatusesOk(const TaskResult& result, std::size_t expectedCount)
{
    return result.status.ok() && result.entryStatus.size() == expectedCount &&
           std::all_of(result.entryStatus.begin(), result.entryStatus.end(),
                       [](const Status& status) { return status.ok(); });
}

Status QueryAndWait(AsuClient& client, const std::vector<CacheKey>& keys, QueryResult& result,
                    std::uint64_t timeoutMs = 0)
{
    TaskId taskId{kInvalidTaskId};
    auto status = client.QueryAsync(keys, taskId);
    if (!status.ok()) { return status; }

    TaskResult taskResult;
    status = client.Wait(taskId, timeoutMs, taskResult);
    if (taskResult.queryResult.has_value()) {
        result = std::move(*taskResult.queryResult);
    } else if (status.ok()) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "client query result is missing");
    }
    return status;
}

bool StoreAndMeasure(AsuClient& client, const std::vector<KVBuffer>& entries, std::uint64_t bytes,
                     MetricsRecorder& metrics)
{
    TaskId taskId{kInvalidTaskId};
    const auto start = std::chrono::steady_clock::now();
    auto status = client.StoreAsync(entries, taskId);
    TaskResult result;
    if (status.ok()) { status = client.Wait(taskId, 1000, result); }
    const bool correct = status.ok() && EntryStatusesOk(result, entries.size());
    metrics.Record(OperationKind::STORE, ElapsedNs(start), correct ? bytes : 0, status.ok(),
                   correct);
    return correct;
}

bool LoadAndMeasure(AsuClient& client, const std::vector<KVBuffer>& entries,
                    const std::vector<std::vector<std::uint8_t>>& expected,
                    const std::vector<std::vector<std::uint8_t>>& actual, std::uint64_t bytes,
                    MetricsRecorder& metrics)
{
    TaskId taskId{kInvalidTaskId};
    const auto start = std::chrono::steady_clock::now();
    auto status = client.LoadAsync(entries, taskId);
    TaskResult result;
    if (status.ok()) { status = client.Wait(taskId, 1000, result); }
    const bool dataMatches = expected == actual;
    const bool correct = status.ok() && EntryStatusesOk(result, entries.size()) && dataMatches;
    metrics.Record(OperationKind::LOAD, ElapsedNs(start), correct ? bytes : 0, status.ok(),
                   correct);
    return correct;
}

bool QueryAndMeasure(AsuClient& client, const std::vector<CacheKey>& keys,
                     const std::vector<std::uint8_t>& expected, MetricsRecorder& metrics)
{
    QueryResult result;
    const auto start = std::chrono::steady_clock::now();
    auto status = QueryAndWait(client, keys, result, 1000);
    const bool correct = status.ok() && result.exists == expected;
    metrics.Record(OperationKind::QUERY, ElapsedNs(start), 0, status.ok(), correct);
    return correct;
}

bool DeleteAndMeasure(AsuClient& client, const std::vector<CacheKey>& keys,
                      MetricsRecorder& metrics)
{
    TaskId taskId{kInvalidTaskId};
    const auto start = std::chrono::steady_clock::now();
    auto status = client.DeleteAsync(keys, taskId);
    TaskResult result;
    if (status.ok()) { status = client.Wait(taskId, 1000, result); }
    const bool correct = status.ok() && EntryStatusesOk(result, keys.size());
    metrics.Record(OperationKind::DELETE, ElapsedNs(start), 0, status.ok(), correct);
    return correct;
}

bool WaitUntil(const std::function<bool()>& condition)
{
    for (std::uint32_t attempt = 0; attempt < 200; ++attempt) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

std::vector<CacheKey> MakeProbeKeys(std::size_t count)
{
    std::vector<CacheKey> keys;
    keys.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        keys.emplace_back(MakeCacheKey("refresh-probe-" + std::to_string(index)));
    }
    return keys;
}

void ExpectStoreAndLoadSummaries(const MetricsRecorder& metrics)
{
    const auto storeSummary = metrics.Summarize(OperationKind::STORE);
    ASSERT_GT(storeSummary.total, 0U);
    EXPECT_DOUBLE_EQ(storeSummary.successRate, 1.0);
    EXPECT_DOUBLE_EQ(storeSummary.correctnessRate, 1.0);
    EXPECT_GT(storeSummary.iops, 0.0);
    EXPECT_GT(storeSummary.bandwidthBytesPerSec, 0.0);
    EXPECT_GT(storeSummary.latencyP50Ns, 0U);

    const auto loadSummary = metrics.Summarize(OperationKind::LOAD);
    ASSERT_GT(loadSummary.total, 0U);
    EXPECT_DOUBLE_EQ(loadSummary.successRate, 1.0);
    EXPECT_DOUBLE_EQ(loadSummary.correctnessRate, 1.0);
    EXPECT_GT(loadSummary.iops, 0.0);
    EXPECT_GT(loadSummary.bandwidthBytesPerSec, 0.0);
    EXPECT_GT(loadSummary.latencyP50Ns, 0U);
}

}  // namespace

TEST(KvClientE2EMetricsTest, NormalWorkloadReportsPrecisionPerformanceAndResources)
{
    auto state = std::make_shared<ClusterState>();
    auto viewServer = std::make_shared<DynamicViewServer>(state, std::vector<AsuId>{1, 2, 3});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(MakeClientConfig({1, 2, 3})).ok());

    MetricsRecorder metrics;
    std::vector<std::vector<std::uint8_t>> payloads;
    auto storeEntries = MakeStoreEntries("normal-key-", 48, 256, payloads);
    const auto keys = ExtractKeys(storeEntries);
    const auto bytes = SumPayloadBytes(payloads);

    ASSERT_TRUE(StoreAndMeasure(*client, storeEntries, bytes, metrics));
    const auto afterStore = state->Snapshot();
    EXPECT_EQ(afterStore.storedKeyCount, storeEntries.size());
    EXPECT_EQ(afterStore.storedBytes, bytes);

    ASSERT_TRUE(QueryAndMeasure(*client, keys, std::vector<std::uint8_t>(keys.size(), 1), metrics));

    std::vector<std::vector<std::uint8_t>> loadBuffers;
    auto loadEntries = MakeLoadEntries(storeEntries, payloads, loadBuffers);
    ASSERT_TRUE(LoadAndMeasure(*client, loadEntries, payloads, loadBuffers, bytes, metrics));

    std::vector<std::vector<std::uint8_t>> overwritePayloads;
    auto overwriteEntries = MakeStoreEntries("normal-key-", 8, 512, overwritePayloads);
    const auto overwriteBytes = SumPayloadBytes(overwritePayloads);
    ASSERT_TRUE(StoreAndMeasure(*client, overwriteEntries, overwriteBytes, metrics));
    std::vector<std::vector<std::uint8_t>> overwriteLoadBuffers;
    auto overwriteLoadEntries =
        MakeLoadEntries(overwriteEntries, overwritePayloads, overwriteLoadBuffers);
    ASSERT_TRUE(LoadAndMeasure(*client, overwriteLoadEntries, overwritePayloads,
                               overwriteLoadBuffers, overwriteBytes, metrics));

    ASSERT_TRUE(DeleteAndMeasure(*client, keys, metrics));
    ASSERT_TRUE(QueryAndMeasure(*client, keys, std::vector<std::uint8_t>(keys.size(), 0), metrics));

    const auto afterDelete = state->Snapshot();
    EXPECT_EQ(afterDelete.storedKeyCount, 0U);
    EXPECT_EQ(afterDelete.storedBytes, 0U);
    EXPECT_GE(state->TouchedAsuCount(), 2U);

    ExpectStoreAndLoadSummaries(metrics);
    const auto querySummary = metrics.Summarize(OperationKind::QUERY);
    EXPECT_DOUBLE_EQ(querySummary.successRate, 1.0);
    EXPECT_DOUBLE_EQ(querySummary.correctnessRate, 1.0);
    const auto deleteSummary = metrics.Summarize(OperationKind::DELETE);
    EXPECT_DOUBLE_EQ(deleteSummary.successRate, 1.0);
    EXPECT_DOUBLE_EQ(deleteSummary.correctnessRate, 1.0);
    metrics.PrintReport("normal workload");

    auto status = client->Shutdown();
    EXPECT_TRUE(status.ok()) << status.message;
}

TEST(KvClientE2EMetricsTest, DiskMembershipChangesRefreshAndContinueWorkload)
{
    auto state = std::make_shared<ClusterState>();
    auto viewServer = std::make_shared<DynamicViewServer>(state, std::vector<AsuId>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(MakeClientConfig({1, 2, 3})).ok());

    MetricsRecorder metrics;
    std::vector<std::vector<std::uint8_t>> payloads;
    auto storeEntries = MakeStoreEntries("membership-key-", 64, 128, payloads);
    ASSERT_TRUE(StoreAndMeasure(*client, storeEntries, SumPayloadBytes(payloads), metrics));

    viewServer->Publish({1, 2, 3});
    state->ForceNextQueryFailure();
    QueryResult ignoredResult;
    auto status = QueryAndWait(*client, {MakeCacheKey("membership-refresh-add")}, ignoredResult);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    const auto addProbeKeys = MakeProbeKeys(512);
    ASSERT_TRUE(WaitUntil([&] {
        state->ClearOperationCalls();
        QueryResult result;
        const auto retryStatus = QueryAndWait(*client, addProbeKeys, result);
        return retryStatus.ok() && state->OperationCalls(3) > 0;
    }));

    std::vector<std::vector<std::uint8_t>> addedPayloads;
    auto addedEntries = MakeStoreEntries("membership-added-key-", 96, 96, addedPayloads);
    ASSERT_TRUE(StoreAndMeasure(*client, addedEntries, SumPayloadBytes(addedPayloads), metrics));
    std::vector<std::vector<std::uint8_t>> addedLoadBuffers;
    auto addedLoadEntries = MakeLoadEntries(addedEntries, addedPayloads, addedLoadBuffers);
    ASSERT_TRUE(LoadAndMeasure(*client, addedLoadEntries, addedPayloads, addedLoadBuffers,
                               SumPayloadBytes(addedPayloads), metrics));

    viewServer->Publish({1, 3});
    state->ForceNextQueryFailure();
    const auto probeKeys = MakeProbeKeys(512);
    status = QueryAndWait(*client, probeKeys, ignoredResult);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitUntil([&] {
        state->ClearOperationCalls();
        QueryResult result;
        const auto retryStatus = QueryAndWait(*client, probeKeys, result);
        return retryStatus.ok() && state->OperationCalls(2) == 0 && state->OperationCalls(3) > 0;
    }));

    state->ClearOperationCalls();
    std::vector<std::vector<std::uint8_t>> removedPayloads;
    auto removedEntries = MakeStoreEntries("membership-removed-key-", 96, 96, removedPayloads);
    ASSERT_TRUE(
        StoreAndMeasure(*client, removedEntries, SumPayloadBytes(removedPayloads), metrics));
    std::vector<std::vector<std::uint8_t>> removedLoadBuffers;
    auto removedLoadEntries = MakeLoadEntries(removedEntries, removedPayloads, removedLoadBuffers);
    ASSERT_TRUE(LoadAndMeasure(*client, removedLoadEntries, removedPayloads, removedLoadBuffers,
                               SumPayloadBytes(removedPayloads), metrics));
    EXPECT_EQ(state->OperationCalls(2), 0U);

    const auto snapshot = state->Snapshot();
    EXPECT_GE(snapshot.createdTransports, 3U);
    ExpectStoreAndLoadSummaries(metrics);
    metrics.PrintReport("disk add and remove");

    status = client->Shutdown();
    EXPECT_TRUE(status.ok()) << status.message;
}

}  // namespace UC::ASU
