/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "shard_gc.h"
#include <algorithm>
#include <unordered_map>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

namespace {

// FinishTask use atomic operation to avoid success result conflict with timeout,
// as they may occur at the same time, and the waiter->Done should only be called once
void FinishTask(const ShardTaskContext& ctx, const std::function<void(ShardTaskResult&)>& write)
{
    if (!ctx.result->finished.exchange(true)) {
        write(*ctx.result);
        ctx.waiter->Done();
    }
}

}  // namespace

ShardGarbageCollector::~ShardGarbageCollector() { StopBackgroundCheck(); }

Status ShardGarbageCollector::ValidateAndInitCapacity()
{
    capacityBytes_ = config_.posixCapacityGb * 1024ULL * 1024ULL * 1024ULL;
    maxFileCount_ = capacityBytes_ / config_.blockSize;
    size_t thresholdFilesPerShard = static_cast<size_t>(
        maxFileCount_ / layout_->SampleShards(1.0).size() * config_.posixGcTriggerThresholdRatio);
    size_t recycleNum = static_cast<size_t>(thresholdFilesPerShard * config_.posixGcRecyclePercent);
    if (recycleNum == 0) {
        size_t minFilesPerShard = static_cast<size_t>(1.0 / (config_.posixGcTriggerThresholdRatio *
                                                             config_.posixGcRecyclePercent)) +
                                  1;
        size_t minCapacityBytes =
            minFilesPerShard * layout_->SampleShards(1.0).size() * config_.blockSize;
        size_t minCapacityGb =
            (minCapacityBytes + 1024ULL * 1024ULL * 1024ULL - 1) / (1024ULL * 1024ULL * 1024ULL);
        return Status::InvalidParam(
            "posix_capacity_gb({}) is too small, GC cannot recycle any files. "
            "Minimum recommended: {}GB",
            config_.posixCapacityGb, minCapacityGb);
    }

    return Status::OK();
}

Status ShardGarbageCollector::Setup(const SpaceLayout* layout, const BackendManager* backendMgr,
                                    const Config& config)
{
    layout_ = layout;
    backendMgr_ = backendMgr;
    config_ = config;
    auto s = ValidateAndInitCapacity();
    if (s.Failure()) { return s; }
    leaseEnable_ = config_.posixGcCrossInstanceLock;
    if (leaseEnable_) { lease_.Setup(config_, backendMgr_); }
    auto success = gcPool_.SetWorkerFn([this](ShardTaskContext& ctx, auto&) { ProcessTask(ctx); })
                       .SetWorkerTimeoutFn(
                           [this](ShardTaskContext& ctx, ssize_t tid) { OnTaskTimeout(ctx, tid); },
                           config_.posixGcTaskTimeoutMs)
                       .SetNWorker(config_.posixGcConcurrency)
                       .Run();
    if (!success) { return Status::Error("failed to start gc thread pool"); }
    try {
        gcCheckWorker_ = std::thread(&ShardGarbageCollector::GCCheckLoop, this);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to create gc check worker thread.", e.what());
        return Status::OutOfMemory();
    }
    return Status::OK();
}

void ShardGarbageCollector::StopBackgroundCheck()
{
    {
        std::lock_guard<std::mutex> lock(gcCheckMtx_);
        stop_ = true;
    }
    gcCheckCv_.notify_all();
    if (gcCheckWorker_.joinable()) { gcCheckWorker_.join(); }
    if (leaseEnable_) { lease_.RequestStop(); }
}

void ShardGarbageCollector::GCCheckLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_gc");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM posix GC check worker name.", nameStatus);
    }
    while (!stop_.load()) {
        if (!leaseEnable_) {
            RunGcCycle();
        } else {
            switch (lease_.TryAcquire()) {
                case GcLease::Acquisition::Acquired:
                    RunGcCycle();
                    lease_.Release();
                    break;
                case GcLease::Acquisition::HeldByPeer:
                    UC_INFO("Another instance holds the GC lock; skipping this cycle.");
                    break;
                case GcLease::Acquisition::Unavailable:
                    UC_WARN(
                        "GC lock is unusable; skipping this cycle. Capacity will not be "
                        "reclaimed until this is resolved.");
                    break;
            }
        }
        {
            std::unique_lock<std::mutex> lock(gcCheckMtx_);
            gcCheckCv_.wait_for(lock, std::chrono::seconds(config_.posixGcCheckIntervalSec),
                                [this] { return stop_.load(); });
        }
        if (stop_.load()) { break; }
    }
}

void ShardGarbageCollector::RunGcCycle()
{
    auto [trigger, avgFilesPerShard, threshold] = ShouldTrigger();
    UC_INFO("GC sampling: avgFiles/shard={}, threshold={}, trigger={}", avgFilesPerShard, threshold,
            trigger);
    int rounds = 0;
    const bool gcRunning = trigger;
    if (gcRunning) { UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_gc_running"), 1.0); }
    while (!stop_.load() && trigger) {
        if (leaseEnable_ && !lease_.HoldsLock()) {
            UC_WARN(
                "Lost the GC lock after {} round(s); a peer has taken over. Stopping this "
                "drain to avoid concurrent reclamation.",
                rounds);
            break;
        }
        bool gcLimited = Execute();
        rounds++;
        if (gcLimited) { continue; }
        std::tie(trigger, avgFilesPerShard, threshold) = ShouldTrigger();
    }
    if (gcRunning) { UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_gc_running"), 0.0); }
    if (rounds > 0) {
        UC_INFO("GC completed: rounds={}, avgFiles/shard={}, threshold={}", rounds,
                avgFilesPerShard, threshold);
    }
}

std::vector<std::shared_ptr<ShardTaskResult>> ShardGarbageCollector::RunShardTasks(
    std::vector<ShardTaskContext> tasks)
{
    auto waiter = std::make_shared<Latch>();
    waiter->Set(tasks.size());
    std::vector<std::shared_ptr<ShardTaskResult>> results;
    results.reserve(tasks.size());
    for (auto& task : tasks) {
        task.waiter = waiter;
        task.result = std::make_shared<ShardTaskResult>();
        results.push_back(task.result);
        gcPool_.Push(std::move(task));
    }
    waiter->Wait();
    return results;
}

bool ShardGarbageCollector::Execute()
{
    if (config_.posixGcPreciseMode) { return ExecutePrecise(); }
    std::vector<ShardTaskContext> tasks;
    for (const auto& shard : layout_->SampleShards(1.0)) {
        tasks.push_back({ShardTaskContext::Type::GC, shard});
    }
    bool gcLimited = false;
    for (const auto& result : RunShardTasks(std::move(tasks))) {
        gcLimited = gcLimited || result->limited;
    }
    return gcLimited;
}

bool ShardGarbageCollector::ExecutePrecise()
{
    std::vector<ShardTaskContext> collects;
    for (const auto& shard : layout_->SampleShards(1.0)) {
        collects.push_back({ShardTaskContext::Type::COLLECT, shard});
    }
    const auto collected = RunShardTasks(std::move(collects));
    bool gcLimited = false;
    size_t candidateCount = 0;
    for (const auto& result : collected) {
        gcLimited = gcLimited || result->limited;
        candidateCount += result->candidates.size();
    }
    if (candidateCount == 0) { return false; }

    std::vector<FileInfo> merged;
    merged.reserve(candidateCount);
    for (const auto& result : collected) {
        merged.insert(merged.end(), result->candidates.begin(), result->candidates.end());
        result->candidates.clear();
        result->candidates.shrink_to_fit();
    }

    const auto candidatePercent =
        config_.posixGcRecyclePercent + config_.posixGcCandidateExtraPercent;
    size_t victimCount = static_cast<size_t>(static_cast<double>(candidateCount) *
                                             config_.posixGcRecyclePercent / candidatePercent);
    if (victimCount == 0) { return false; }
    victimCount = std::min(victimCount, merged.size());
    std::nth_element(
        merged.begin(), merged.begin() + victimCount - 1, merged.end(),
        [](const FileInfo& lhs, const FileInfo& rhs) { return lhs.mtime < rhs.mtime; });

    std::unordered_map<std::string, std::vector<Detail::BlockId>> victimsByShard;
    for (size_t i = 0; i < victimCount; i++) {
        victimsByShard[layout_->ShardOf(merged[i].blockId)].push_back(merged[i].blockId);
    }
    UC_DEBUG("GC precise: candidates={}, deleting={} across {} shard(s)", candidateCount,
             victimCount, victimsByShard.size());

    std::vector<ShardTaskContext> deletes;
    for (auto& [shard, victims] : victimsByShard) {
        deletes.push_back(
            {ShardTaskContext::Type::DELETE, shard, nullptr, nullptr,
             std::make_shared<const std::vector<Detail::BlockId>>(std::move(victims))});
    }
    RunShardTasks(std::move(deletes));
    return gcLimited;
}

std::tuple<bool, size_t, size_t> ShardGarbageCollector::ShouldTrigger()
{
    std::vector<ShardTaskContext> samples;
    for (const auto& shard : layout_->SampleShards(config_.posixGcShardSampleRatio)) {
        samples.push_back({ShardTaskContext::Type::SAMPLE, shard});
    }
    size_t sampledFiles = 0;
    size_t sampledShards = 0;
    for (const auto& result : RunShardTasks(std::move(samples))) {
        if (result->scanned) {
            sampledFiles += result->files;
            ++sampledShards;
        }
    }
    const auto shardCount = layout_->SampleShards(1.0).size();
    size_t thresholdFilesPerShard = maxFileCount_ / shardCount;
    size_t threshold =
        static_cast<size_t>(thresholdFilesPerShard * config_.posixGcTriggerThresholdRatio);
    if (sampledShards == 0) {
        UC_WARN("GC sampling got no result from any shard; skipping this check.");
        return {false, 0, threshold};
    }
    size_t avgFilesPerShard = sampledFiles / sampledShards;
    const auto estimatedFiles = static_cast<double>(sampledFiles) *
                                static_cast<double>(shardCount) /
                                static_cast<double>(sampledShards);
    const auto usedBytes = estimatedFiles * static_cast<double>(config_.blockSize);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_store_used_bytes"), usedBytes);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_store_capacity_bytes"),
                             static_cast<double>(capacityBytes_));
    UC::Metrics::UpdateStats(
        NAME_TO_METRIC_ID("posix_store_usage_ratio"),
        capacityBytes_ == 0 ? 0.0 : usedBytes / static_cast<double>(capacityBytes_));
    return {avgFilesPerShard >= threshold, avgFilesPerShard, threshold};
}

void ShardGarbageCollector::OnTaskTimeout(const ShardTaskContext& ctx, ssize_t tid)
{
    UC_WARN(
        "GC {} task on shard({}) exceeded {}ms (tid={}); storage may be unresponsive. "
        "Giving up this shard for this round; a replacement worker is started.",
        ctx.type == ShardTaskContext::Type::SAMPLE ? "sample" : "recycle", ctx.shard,
        config_.posixGcTaskTimeoutMs, tid);
    std::optional<BackendManager::BackendAttempt> attempt;
    {
        std::lock_guard<std::mutex> lock(ctx.result->attemptMutex);
        attempt = ctx.result->attempt;
    }
    if (attempt) { backendMgr_->RecordResult(attempt.value(), Status::Timeout()); }
    FinishTask(ctx, [](ShardTaskResult&) {});
}

void ShardGarbageCollector::RemoveFiles(const std::vector<Detail::BlockId>& blocks,
                                        const std::string& backend, const ShardTaskResult& result)
{
    for (const auto& blockId : blocks) {
        if (result.finished.load()) { break; }
        layout_->RemoveFile(backend, blockId);
    }
}

void ShardGarbageCollector::ProcessTask(ShardTaskContext& ctx)
{
    std::vector<std::string> attempted;
    auto attempt = backendMgr_->SelectNextBackend({}, attempted);
    if (!attempt) {
        FinishTask(ctx, [](ShardTaskResult&) {});
        return;
    }
    {
        std::lock_guard<std::mutex> lock(ctx.result->attemptMutex);
        ctx.result->attempt = attempt.Value();
    }
    const auto& backend = backendMgr_->Backends()[attempt.Value().index];
    const auto maxRecycle = config_.posixGcMaxRecycleCountPerShard;
    size_t files = 0;
    std::vector<FileInfo> candidates;
    std::vector<Detail::BlockId> oldest;
    switch (ctx.type) {
        case ShardTaskContext::Type::SAMPLE: {
            files = layout_->CountFilesInShard(backend, ctx.shard);
            break;
        }
        case ShardTaskContext::Type::COLLECT: {
            const auto candidatePercent =
                config_.posixGcRecyclePercent + config_.posixGcCandidateExtraPercent;
            candidates =
                layout_->GetColdestCandidates(backend, ctx.shard, candidatePercent, maxRecycle);
            break;
        }
        case ShardTaskContext::Type::DELETE: {
            RemoveFiles(*ctx.victims, backend, *ctx.result);
            break;
        }
        case ShardTaskContext::Type::GC: {
            oldest = layout_->GetOldestFiles(backend, ctx.shard, config_.posixGcRecyclePercent,
                                             maxRecycle);
            RemoveFiles(oldest, backend, *ctx.result);
            break;
        }
    }
    const auto limited = candidates.size() >= maxRecycle || oldest.size() >= maxRecycle;
    FinishTask(ctx, [&](ShardTaskResult& result) {
        result.scanned = true;
        result.files = files;
        result.limited = limited;
        result.candidates = std::move(candidates);
    });
}

}  // namespace UC::PosixStore
