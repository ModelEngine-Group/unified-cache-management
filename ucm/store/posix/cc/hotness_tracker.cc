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
#include <utime.h>
#include "logger/logger.h"
#include "posix_file.h"
#include "shard_gc.h"

namespace UC::PosixStore {

HotnessTracker::~HotnessTracker()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_.store(true);
        cv_.notify_all();
    }
    if (worker_.joinable()) { worker_.join(); }
}

Status HotnessTracker::Setup(const SpaceLayout* layout, size_t intervalSeconds)
{
    if (!layout) { return Status::InvalidParam("layout is null"); }
    layout_ = layout;
    intervalSeconds_ = intervalSeconds;
    stop_.store(false);
    worker_ = std::thread(&HotnessTracker::UpdateLoop, this);
    return Status::OK();
}

void HotnessTracker::SetGCTrigger(ShardGarbageCollector* gc, size_t maxFileCount,
                                  double thresholdRatio)
{
    gc_ = gc;
    maxFileCount_ = maxFileCount;
    thresholdRatio_ = thresholdRatio;
}

void HotnessTracker::Touch(const Detail::BlockId& blockId)
{
    auto blockStr = BlockIdToHex(blockId);
    std::lock_guard<std::mutex> lock(mtx_);
    pendingBlocks_.insert(std::move(blockStr));
}

void HotnessTracker::UpdateLoop()
{
    running_.store(true);
    while (!stop_.load()) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait_for(lock, std::chrono::seconds(intervalSeconds_),
                         [this] { return stop_.load(); });
        }
        if (stop_.load()) { break; }
        FlushPendingBlocks();
        if (gc_ && gc_->ShouldTrigger(maxFileCount_, thresholdRatio_)) { gc_->Trigger(); }
    }
    FlushPendingBlocks();
    running_.store(false);
}

void HotnessTracker::FlushPendingBlocks()
{
    std::unordered_set<std::string> blocksToUpdate;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pendingBlocks_.empty()) { return; }
        blocksToUpdate.swap(pendingBlocks_);
    }

    for (const auto& blockStr : blocksToUpdate) {
        Detail::BlockId blockId{};
        if (blockStr.size() != 32) { continue; }
        for (size_t i = 0; i < 16; i++) {
            auto byte = std::stoul(blockStr.substr(i * 2, 2), nullptr, 16);
            blockId[i] = static_cast<std::byte>(byte);
        }
        auto filePath = layout_->DataFilePath(blockId, false);
        utime(filePath.c_str(), nullptr);
    }
}

}  // namespace UC::PosixStore
