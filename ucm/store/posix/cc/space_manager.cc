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
#include <atomic>
#include "logger/logger.h"
#include "metrics_api.h"

namespace UC::PosixStore {

Status SpaceManager::Setup(const Config& config)
{
    hotnessTrackerEnable_ = config.deviceId == -1;
    gcEnable_ = config.posixGcEnable && config.posixCapacityGb > 0;
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_gc_running"), 0.0);
    auto s = layout_.Setup(config);
    if (s.Failure()) [[unlikely]] { return s; }
    s = backendMgr_.Setup(config, &layout_);
    if (s.Failure()) { return s; }
    if (hotnessTrackerEnable_) {
        s = hotnessTracker_.Setup(&layout_, &backendMgr_);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    if (gcEnable_) {
        s = gcMgr_.Setup(&layout_, &backendMgr_, config);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    auto backendConfig = config;
    backendConfig.timeoutMs = backendMgr_.IoTimeoutMs();
    for (const auto& backend : backendMgr_.Backends()) {
        auto manager = std::make_unique<LookupManager>();
        s = manager->Setup(backendConfig, &layout_, backend);
        if (s.Failure()) { return s; }
        lookupManagers_.push_back(std::move(manager));
    }
    auto success =
        prefixLookupSrv_
            .SetWorkerFn([this](PrefixLookupContext& ctx, auto&) { OnLookupPrefix(ctx); })
            .SetNWorker(config.lookupConcurrency)
            .SetCpuAffinity(config.cpuAffinityCores)
            .Run();
    if (!success) { return Status::Error("failed to run prefix lookup service thread pool"); }
    return Status::OK();
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
    std::shared_ptr<Latch> waiter;

    try {
        firstFail = std::make_shared<std::atomic<ssize_t>>(static_cast<ssize_t>(num));
        waiter = std::make_shared<Latch>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to allocate prefix lookup context.", e.what());
        return Status::OutOfMemory();
    }

    const size_t nWorker = prefixLookupSrv_.NWorker();
    waiter->Set(nWorker);

    for (size_t begin = 0; begin < nWorker; begin++) {
        prefixLookupSrv_.Push({blocks, begin, num, nWorker, firstFail, waiter});
    }

    waiter->Wait();

    return firstFail->load() - 1;
}

Expected<ssize_t> SpaceManager::LookupOnReverse(const Detail::BlockId* blocks, size_t num)
{
    if (num == 0) { return static_cast<ssize_t>(-1); }
    for (ssize_t i = static_cast<ssize_t>(num) - 1; i >= 0; --i) {
        if (Lookup(blocks + i)) {
            if (hotnessTrackerEnable_) { hotnessTracker_.Touch(*(blocks + i)); }
            return i;
        }
    }
    return static_cast<ssize_t>(-1);
}

uint8_t SpaceManager::Lookup(const Detail::BlockId* block)
{
    const auto status = backendMgr_.RunOnAvailableBackend(
        *block, [&](size_t backendIndex) { return lookupManagers_[backendIndex]->Lookup(*block); });
    return status.Success();
}

void SpaceManager::Prefetch(const Detail::BlockId* blocks, size_t num)
{
    if (!hotnessTrackerEnable_) { return; }

    for (size_t i = 0; i < num; i++) { hotnessTracker_.Touch(blocks[i]); }
}

void SpaceManager::OnLookupPrefix(PrefixLookupContext& ctx)
{
    for (size_t i = ctx.begin; i < ctx.end; i += ctx.nWorker) {
        if (i >= static_cast<size_t>(ctx.firstFail->load())) { break; }
        if (!Lookup(ctx.blocks + i)) {
            ssize_t cur = ctx.firstFail->load();
            while (static_cast<ssize_t>(i) < cur) {
                if (ctx.firstFail->compare_exchange_weak(cur, static_cast<ssize_t>(i),
                                                         std::memory_order_acq_rel)) {
                    break;
                }
            }
            break;
        }
        if (hotnessTrackerEnable_) { hotnessTracker_.Touch(*(ctx.blocks + i)); }
    }
    ctx.waiter->Done();
}
}  // namespace UC::PosixStore
