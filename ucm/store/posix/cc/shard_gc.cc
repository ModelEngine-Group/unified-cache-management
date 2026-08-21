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
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

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

Status ShardGarbageCollector::Setup(const SpaceLayout* layout, const Config& config)
{
    layout_ = layout;
    config_ = config;
    auto s = ValidateAndInitCapacity();
    if (s.Failure()) { return s; }
    leaseEnable_ = config_.posixGcCrossInstanceLock;
    if (leaseEnable_) { lease_.Setup(config_); }
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
                    UC_ERROR(
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

bool ShardGarbageCollector::Execute()
{
    auto waiter = std::make_shared<Latch>();
    auto shards = layout_->SampleShards(1.0);
    waiter->Set(shards.size());
    std::atomic<bool> gcLimited{false};
    for (const auto& shard : shards) {
        gcPool_.Push({ShardTaskContext::Type::GC, shard, waiter, nullptr, &gcLimited});
    }
    waiter->Wait();
    return gcLimited.load();
}

std::tuple<bool, size_t, size_t> ShardGarbageCollector::ShouldTrigger()
{
    auto sampleShards = layout_->SampleShards(config_.posixGcShardSampleRatio);
    auto waiter = std::make_shared<Latch>();
    std::atomic<size_t> sampledFiles{0};
    waiter->Set(sampleShards.size());
    for (const auto& shard : sampleShards) {
        gcPool_.Push({ShardTaskContext::Type::SAMPLE, shard, waiter, &sampledFiles});
    }
    waiter->Wait();
    size_t avgFilesPerShard = sampledFiles.load() / sampleShards.size();
    const auto shardCount = layout_->SampleShards(1.0).size();
    size_t thresholdFilesPerShard = maxFileCount_ / shardCount;
    size_t threshold =
        static_cast<size_t>(thresholdFilesPerShard * config_.posixGcTriggerThresholdRatio);
    const auto estimatedFiles = static_cast<double>(sampledFiles.load()) *
                                static_cast<double>(shardCount) /
                                static_cast<double>(sampleShards.size());
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
        "This round keeps waiting; a replacement worker is started.",
        ctx.type == ShardTaskContext::Type::SAMPLE ? "sample" : "recycle", ctx.shard,
        config_.posixGcTaskTimeoutMs, tid);
}

void ShardGarbageCollector::ProcessTask(ShardTaskContext& ctx)
{
    if (ctx.type == ShardTaskContext::Type::SAMPLE) {
        size_t count = layout_->CountFilesInShard(ctx.shard);
        ctx.sampledFiles->fetch_add(count, std::memory_order_relaxed);
    } else {
        auto filesToDelete = layout_->GetOldestFiles(ctx.shard, config_.posixGcRecyclePercent,
                                                     config_.posixGcMaxRecycleCountPerShard);
        for (const auto& blockId : filesToDelete) { layout_->RemoveFile(blockId); }
        if (filesToDelete.size() >= config_.posixGcMaxRecycleCountPerShard) {
            ctx.gcLimited->store(true, std::memory_order_relaxed);
        }
    }
    ctx.waiter->Done();
}

}  // namespace UC::PosixStore
