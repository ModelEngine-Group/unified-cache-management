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
#include "space_manager.h"
#include "logger/logger.h"
#include "posix_file.h"

namespace UC::PosixStore {

Status SpaceManager::Setup(const Config& config)
{
    gcEnable_ = config.posixGcEnable;

    auto s = layout_.Setup(config);
    if (s.Failure()) [[unlikely]] { return s; }

    if (gcEnable_) {
        s = hotnessTracker_.Setup(&layout_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to setup HotnessTracker: {}.", s);
            return s;
        }
    }

    auto prefixSuccess =
        prefixLookupSrv_
            .SetWorkerFn([this](PrefixLookupContext& ctx, auto&) { OnLookupPrefix(ctx); })
            .SetWorkerTimeoutFn(
                [this](PrefixLookupContext& ctx, auto) { OnLookupPrefixTimeout(ctx); },
                config.timeoutMs)
            .SetNWorker(config.lookupConcurrency)
            .Run();
    if (!prefixSuccess) [[unlikely]] {
        return Status::Error("failed to run prefix lookup service thread pool");
    }

    if (gcEnable_) { return SetupGC(config); }

    return Status::OK();
}

Status SpaceManager::SetupGC(const Config& config)
{
    size_t maxFileCount = 0;

    if (config.posixCapacityGb > 0) {
        size_t storageCapacityBytes = config.posixCapacityGb * 1024ULL * 1024ULL * 1024ULL;
        maxFileCount = storageCapacityBytes / config.blockSize;
        auto shards = layout_.RelativeRoots();
        size_t totalShards = shards.size();
        if (totalShards > 0) {
            size_t thresholdFilesPerShard = static_cast<size_t>(
                maxFileCount / totalShards * config.posixGcTriggerThresholdRatio);
            size_t recycleNum =
                static_cast<size_t>(thresholdFilesPerShard * config.posixGcRecyclePercent);

            if (recycleNum == 0) {
                size_t minFilesPerShard =
                    static_cast<size_t>(1.0 / (config.posixGcTriggerThresholdRatio *
                                               config.posixGcRecyclePercent)) +
                    1;
                size_t minCapacityBytes = minFilesPerShard * totalShards * config.blockSize;
                size_t minCapacityGb = (minCapacityBytes + 1024ULL * 1024ULL * 1024ULL - 1) /
                                       (1024ULL * 1024ULL * 1024ULL);
                return Status::InvalidParam(
                    "posix_capacity_gb({}) is too small, GC cannot recycle any files. "
                    "Minimum recommended: {}GB",
                    config.posixCapacityGb, minCapacityGb);
            }

            UC_INFO("GC enabled: capacityGb={}, thresholdFilesPerShard={}", config.posixCapacityGb,
                    thresholdFilesPerShard);
        }
    }

    ShardGCConfig gcConfig{
        config.posixGcRecyclePercent,
        config.posixGcConcurrency,
        maxFileCount,
        config.posixGcTriggerThresholdRatio,
        config.posixGcCheckIntervalSec,
        config.posixGcMaxRecycleCountPerShard,
        config.posixGcShardSampleRatio,
    };

    return gcMgr_.Setup(&layout_, gcConfig);
}

Expected<std::vector<uint8_t>> SpaceManager::Lookup(const Detail::BlockId* blocks, size_t num)
{
    std::vector<uint8_t> results(num, false);
    auto res = LookupOnPrefix(blocks, num);
    if (!res) [[unlikely]] { return res.Error(); }
    const auto index = res.Value();
    for (ssize_t i = 0; i <= index; ++i) { results[i] = true; }
    return results;
}

Expected<ssize_t> SpaceManager::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    if (num == 0) { return static_cast<ssize_t>(-1); }

    std::shared_ptr<std::atomic<ssize_t>> firstFail;
    std::shared_ptr<std::atomic<int32_t>> status;
    std::shared_ptr<Latch> waiter;

    const auto ok = Status::OK().Underlying();

    try {
        firstFail = std::make_shared<std::atomic<ssize_t>>(static_cast<ssize_t>(num));
        status = std::make_shared<std::atomic<int32_t>>(ok);
        waiter = std::make_shared<Latch>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to allocate prefix lookup context.", e.what());
        return Status::OutOfMemory();
    }

    const size_t nWorker = prefixLookupSrv_.NWorker();
    waiter->Set(nWorker);

    for (size_t begin = 0; begin < nWorker; begin++) {
        prefixLookupSrv_.Push({blocks, begin, num, nWorker, firstFail, status, waiter});
    }

    waiter->Wait();

    auto s = status->load();
    if (s != ok) [[unlikely]] { return Status{s, "failed to lookup some blocks"}; }

    return firstFail->load() - 1;
}

uint8_t SpaceManager::Lookup(const Detail::BlockId& block)
{
    const auto& path = layout_.DataFilePath(block, false);
    PosixFile file{path};
    constexpr auto mode =
        PosixFile::AccessMode::EXIST | PosixFile::AccessMode::READ | PosixFile::AccessMode::WRITE;
    auto s = file.Access(mode);
    if (s.Failure()) {
        if (s != Status::NotFound()) { UC_ERROR("Failed({}) to access file({}).", s, path); }
        return false;
    }
    return true;
}

void SpaceManager::OnLookupPrefix(PrefixLookupContext& ctx)
{
    for (size_t i = ctx.begin; i < ctx.end; i += ctx.nWorker) {
        if (ctx.status->load() != Status::OK().Underlying()) { break; }

        auto curFail = ctx.firstFail->load();
        if (curFail >= 0 && static_cast<size_t>(curFail) < i) { break; }

        if (!Lookup(ctx.blocks[i])) {
            ssize_t cur = ctx.firstFail->load();
            while (static_cast<ssize_t>(i) < cur) {
                if (ctx.firstFail->compare_exchange_weak(cur, static_cast<ssize_t>(i),
                                                         std::memory_order_acq_rel)) {
                    break;
                }
            }
            break;
        }
        if (gcEnable_) { hotnessTracker_.Touch(ctx.blocks[i]); }
    }
    ctx.waiter->Done();
}

void SpaceManager::OnLookupPrefixTimeout(PrefixLookupContext& ctx)
{
    auto ok = Status::OK().Underlying();
    auto timeout = Status::Timeout().Underlying();
    ctx.status->compare_exchange_weak(ok, timeout, std::memory_order_acq_rel);
    ctx.waiter->Done();
}
}  // namespace UC::PosixStore
