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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_HOTNESS_TRACKER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_HOTNESS_TRACKER_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include "space_layout.h"
#include "type/types.h"

namespace UC::PosixStore {

class ShardGarbageCollector;

class HotnessTracker {
public:
    HotnessTracker() = default;
    HotnessTracker(const HotnessTracker&) = delete;
    HotnessTracker& operator=(const HotnessTracker&) = delete;
    ~HotnessTracker();

    Status Setup(const SpaceLayout* layout, size_t intervalSeconds = 60);
    void SetGCTrigger(ShardGarbageCollector* gc, size_t maxFileCount, double thresholdRatio);
    void Touch(const Detail::BlockId& blockId);

private:
    void UpdateLoop();
    void FlushPendingBlocks();

    const SpaceLayout* layout_{nullptr};
    size_t intervalSeconds_{60};

    std::mutex mtx_;
    std::unordered_set<std::string> pendingBlocks_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::thread worker_;
    std::condition_variable cv_;

    ShardGarbageCollector* gc_{nullptr};
    size_t maxFileCount_{0};
    double thresholdRatio_{0.0};
};

}  // namespace UC::PosixStore

#endif
