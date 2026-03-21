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
#include "posix_file.h"

namespace UC::PosixStore {

ShardGarbageCollector::~ShardGarbageCollector() { StopBackgroundCheck(); }

Status ShardGarbageCollector::Setup(const SpaceLayout* layout, const ShardGCConfig& config)
{
    layout_ = layout;
    config_ = config;
    shards_ = layout_->RelativeRoots();

    if (shards_.empty()) { return Status::InvalidParam("no shards available"); }
    if (config_.recyclePercent <= 0 || config_.recyclePercent > 1.0f) {
        return Status::InvalidParam("invalid recycle percent");
    }
    if (config_.gcConcurrency == 0) { return Status::InvalidParam("invalid gc concurrency"); }

    auto success = gcPool_.SetWorkerFn([this](ShardTaskContext& ctx, auto&) { ProcessTask(ctx); })
                       .SetNWorker(config_.gcConcurrency)
                       .Run();

    if (!success) { return Status::Error("failed to start gc thread pool"); }

    UC_INFO(
        "ShardGC setup: recyclePercent={}, concurrency={}, maxFileCount={}, "
        "thresholdRatio={}, checkIntervalSec={}",
        config_.recyclePercent, config_.gcConcurrency, config_.maxFileCount, config_.thresholdRatio,
        config_.gcCheckIntervalSec);

    if (config_.maxFileCount > 0) {
        gcCheckWorker_ = std::thread(&ShardGarbageCollector::GCCheckLoop, this);
        while (!stop_.load() && ShouldTrigger()) { Execute(); }
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
    while (!stop_.load()) {
        {
            std::unique_lock<std::mutex> lock(gcCheckMtx_);
            gcCheckCv_.wait_for(lock, std::chrono::seconds(config_.gcCheckIntervalSec),
                                [this] { return stop_.load(); });
        }
        if (stop_.load()) { break; }
        while (!stop_.load() && ShouldTrigger()) { Execute(); }
    }
}

void ShardGarbageCollector::Execute()
{
    std::shared_ptr<Latch> waiter;

    try {
        waiter = std::make_shared<Latch>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed to allocate latch: ", e.what());
        return;
    }

    waiter->Set(shards_.size());

    for (const auto& shard : shards_) {
        gcPool_.Push({ShardTaskContext::Type::GC, shard, waiter, nullptr});
    }

    waiter->Wait();
}

bool ShardGarbageCollector::ShouldTrigger()
{
    auto sampleShards = layout_->SampleShards(config_.shardSampleRatio);
    if (sampleShards.empty()) { return false; }

    std::shared_ptr<Latch> waiter;

    try {
        waiter = std::make_shared<Latch>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed to allocate latch: {}", e.what());
        return false;
    }

    std::atomic<size_t> sampledFiles{0};
    waiter->Set(sampleShards.size());

    for (const auto& shard : sampleShards) {
        gcPool_.Push({ShardTaskContext::Type::SAMPLE, shard, waiter, &sampledFiles});
    }

    waiter->Wait();

    size_t avgFilesPerShard = sampledFiles.load() / sampleShards.size();
    size_t thresholdFilesPerShard = config_.maxFileCount / shards_.size();

    UC_INFO("GC sampling: avgFiles/shard={}, threshold={}", avgFilesPerShard,
            static_cast<size_t>(thresholdFilesPerShard * config_.thresholdRatio));

    return avgFilesPerShard >= static_cast<size_t>(thresholdFilesPerShard * config_.thresholdRatio);
}

void ShardGarbageCollector::ProcessTask(ShardTaskContext& ctx)
{
    if (ctx.type == ShardTaskContext::Type::SAMPLE) {
        size_t count = layout_->CountFilesInShard(ctx.shard);
        ctx.sampledFiles->fetch_add(count, std::memory_order_relaxed);
    } else {
        auto filesToDelete = layout_->GetOldestFiles(ctx.shard, config_.recyclePercent,
                                                     config_.maxRecycleCountPerShard);
        for (const auto& blockId : filesToDelete) {
            PosixFile{layout_->DataFilePath(blockId, false)}.Remove();
        }
    }

    ctx.waiter->Done();
}

}  // namespace UC::PosixStore
