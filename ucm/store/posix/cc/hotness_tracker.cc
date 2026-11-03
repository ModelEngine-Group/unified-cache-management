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
#include "hotness_tracker.h"
#include <chrono>
#include <string>
#include <utility>
#include <utime.h>
#include "logger/logger.h"
#include "thread/cpu_affinity.h"

namespace UC::PosixStore {

namespace {

// Keeps memory bounded when a backend hangs. Touches over this limit are dropped.
constexpr size_t kMaxPendingBlocks = 65536;

}  // namespace

HotnessTracker::~HotnessTracker()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
}

Status HotnessTracker::Setup(const SpaceLayout* layout, const BackendManager* backendMgr)
{
    layout_ = layout;
    backendMgr_ = backendMgr;
    const auto success =
        utimeSrv_.SetWorkerFn([this](Batch& batch, auto&) { TouchBatch(batch); })
            .SetWorkerTimeoutFn(
                [this](Batch& batch, auto) {
                    backendMgr_->RecordResult(batch.attempt, Status::Timeout());
                    UC_WARN(
                        "Hotness batch of {} block(s) in backend({}:{}) timed out; a new worker "
                        "takes over.",
                        batch.blocks.size(), batch.attempt.index,
                        backendMgr_->Backends()[batch.attempt.index]);
                },
                backendMgr_->IoTimeoutMs())
            .SetNWorker(backendMgr_->BackendCount())
            .Run();
    if (!success) { return Status::Error("failed to run hotness utime thread pool"); }
    try {
        dispatcher_ = std::thread(&HotnessTracker::DispatchLoop, this);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to create hotness dispatcher thread.", e.what());
        return Status::OutOfMemory();
    }
    return Status::OK();
}

void HotnessTracker::Touch(const Detail::BlockId& blockId)
{
    if (pendingBlocks_.fetch_add(1, std::memory_order_relaxed) < kMaxPendingBlocks) {
        std::lock_guard<std::mutex> lock(queueMtx_);
        produceQueue_.push_back(blockId);
    } else {
        pendingBlocks_.fetch_sub(1, std::memory_order_relaxed);
        droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
    }
}

void HotnessTracker::DispatchLoop()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_hot");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM posix hotness worker name.", nameStatus);
    }
    std::deque<Detail::BlockId> consumeQueue;
    constexpr size_t kSpinLimit = 16;
    size_t spinCount = 0;
    while (!stop_.load()) {
        const auto dropped = droppedBlocks_.exchange(0, std::memory_order_relaxed);
        if (dropped > 0) {
            UC_WARN("Dropped {} hotness touch(es) because the utime queue is full.", dropped);
        }
        {
            std::lock_guard<std::mutex> lock(queueMtx_);
            consumeQueue.swap(produceQueue_);
        }
        if (consumeQueue.empty()) {
            if (++spinCount < kSpinLimit) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                spinCount = 0;
            }
            continue;
        }
        spinCount = 0;
        // Group by backend so a hung backend only holds up its own batch.
        std::vector<Batch> batches(backendMgr_->BackendCount());
        size_t noBackend = 0;
        for (const auto& block : consumeQueue) {
            std::vector<std::string> attempted;
            auto attempt = backendMgr_->SelectNextBackend(block, attempted);
            if (attempt) {
                auto& batch = batches[attempt.Value().index];
                batch.attempt = attempt.Value();
                batch.blocks.push_back(block);
            } else {
                ++noBackend;
            }
        }
        for (auto& batch : batches) {
            if (!batch.blocks.empty()) { utimeSrv_.Push(std::move(batch)); }
        }
        consumeQueue.clear();
        if (noBackend > 0) {
            pendingBlocks_.fetch_sub(noBackend, std::memory_order_relaxed);
            UC_WARN("Skipped {} hotness touch(es) because no backend is available.", noBackend);
        }
    }
}

// Touches every block on the batch's backend and records each result against it. A failed touch
// is not retried on another backend, since hotness is best effort.
void HotnessTracker::TouchBatch(const Batch& batch)
{
    const auto& backend = backendMgr_->Backends()[batch.attempt.index];
    for (const auto& block : batch.blocks) {
        const auto path = layout_->ArchivedFilePath(backend, block);
        utime(path.c_str(), nullptr);
    }
    pendingBlocks_.fetch_sub(batch.blocks.size(), std::memory_order_relaxed);
}

}  // namespace UC::PosixStore
