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
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

ShardGarbageCollector::~ShardGarbageCollector() { StopBackgroundCheck(); }

Status ShardGarbageCollector::ValidateAndInitCapacity()
{
    size_t storageCapacityBytes = config_.posixCapacityGb * 1024ULL * 1024ULL * 1024ULL;
    maxFileCount_ = storageCapacityBytes / config_.blockSize;
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
    auto success = gcPool_.SetWorkerFn([this](ShardTaskContext& ctx, auto&) { ProcessTask(ctx); })
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
}

void ShardGarbageCollector::GCCheckLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_gc");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM posix GC check worker name.", nameStatus);
    }
    while (!stop_.load()) {
        auto sampleStart = std::chrono::steady_clock::now();
        auto [trigger, avgFilesPerShard, scannedFiles, scannedShards, threshold] = ShouldTrigger();
        auto sampleEnd = std::chrono::steady_clock::now();
        auto sampleMs = std::chrono::duration_cast<std::chrono::milliseconds>(sampleEnd - sampleStart).count();
        UC_INFO("GC sampling: avgFiles/shard={}, scanned_files={}, scanned_shards={}, "
                "threshold={}, trigger={}, sample_time_ms={}",
                avgFilesPerShard, scannedFiles, scannedShards, threshold, trigger, sampleMs);
        
        int rounds = 0;
        size_t totalDeletedFiles = 0;
        while (!stop_.load() && trigger) {
            auto roundStart = std::chrono::steady_clock::now();
            size_t deletedFiles = 0;
            bool gcLimited = Execute(deletedFiles);
            auto gcEnd = std::chrono::steady_clock::now();
            auto gcMs = std::chrono::duration_cast<std::chrono::milliseconds>(gcEnd - roundStart).count();
            
            rounds++;
            totalDeletedFiles += deletedFiles;
            UC_INFO("GC round {}: gc_time_ms={}, deleted_files={}", rounds, gcMs, deletedFiles);
            
            if (gcLimited) { continue; }
            
            sampleStart = std::chrono::steady_clock::now();
            std::tie(trigger, avgFilesPerShard, scannedFiles, scannedShards, threshold) = ShouldTrigger();
            sampleEnd = std::chrono::steady_clock::now();
            sampleMs = std::chrono::duration_cast<std::chrono::milliseconds>(sampleEnd - sampleStart).count();
            UC_INFO("GC resample: avgFiles/shard={}, scanned_files={}, scanned_shards={}, "
                    "threshold={}, trigger={}, sample_time_ms={}",
                    avgFilesPerShard, scannedFiles, scannedShards, threshold, trigger, sampleMs);
        }
        if (rounds > 0) {
            UC_INFO("GC completed: rounds={}, total_deleted_files={}", rounds, totalDeletedFiles);
        }
        {
            std::unique_lock<std::mutex> lock(gcCheckMtx_);
            gcCheckCv_.wait_for(lock, std::chrono::seconds(config_.posixGcCheckIntervalSec),
                                [this] { return stop_.load(); });
        }
        if (stop_.load()) { break; }
    }
}

bool ShardGarbageCollector::Execute(size_t& deletedFiles)
{
    auto waiter = std::make_shared<Latch>();
    auto shards = layout_->SampleShards(1.0);
    waiter->Set(shards.size());
    std::atomic<bool> gcLimited{false};
    std::atomic<size_t> deletedCount{0};
    for (const auto& shard : shards) {
        gcPool_.Push({ShardTaskContext::Type::GC, shard, waiter, nullptr, &gcLimited, &deletedCount});
    }
    waiter->Wait();
    deletedFiles = deletedCount.load();
    return gcLimited.load();
}

std::tuple<bool, size_t, size_t, size_t, size_t> ShardGarbageCollector::ShouldTrigger()
{
    auto sampleShards = layout_->SampleShards(config_.posixGcShardSampleRatio);
    auto waiter = std::make_shared<Latch>();
    std::atomic<size_t> sampledFiles{0};
    waiter->Set(sampleShards.size());
    for (const auto& shard : sampleShards) {
        gcPool_.Push({ShardTaskContext::Type::SAMPLE, shard, waiter, &sampledFiles});
    }
    waiter->Wait();
    size_t totalScannedFiles = sampledFiles.load();
    size_t avgFilesPerShard = totalScannedFiles / sampleShards.size();
    size_t thresholdFilesPerShard = maxFileCount_ / layout_->SampleShards(1.0).size();
    size_t threshold =
        static_cast<size_t>(thresholdFilesPerShard * config_.posixGcTriggerThresholdRatio);
    return {avgFilesPerShard >= threshold, avgFilesPerShard, totalScannedFiles, 
            sampleShards.size(), threshold};
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
        ctx.deletedFiles->fetch_add(filesToDelete.size(), std::memory_order_relaxed);
        if (filesToDelete.size() >= config_.posixGcMaxRecycleCountPerShard) {
            ctx.gcLimited->store(true, std::memory_order_relaxed);
        }
    }
    ctx.waiter->Done();
}

}  // namespace UC::PosixStore
