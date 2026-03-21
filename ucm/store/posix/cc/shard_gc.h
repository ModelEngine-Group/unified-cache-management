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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_SHARD_GC_H
#define UNIFIEDCACHE_POSIX_STORE_CC_SHARD_GC_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "space_layout.h"
#include "status/status.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"

namespace UC::PosixStore {

struct ShardGCConfig {
    double recyclePercent;
    size_t gcConcurrency;
    size_t maxFileCount;
    double thresholdRatio;
    size_t gcCheckIntervalSec;
    size_t maxRecycleCountPerShard;
    double shardSampleRatio;
};

struct ShardTaskContext {
    enum class Type { GC, SAMPLE };
    Type type;
    std::string shard;
    std::shared_ptr<Latch> waiter;
    std::atomic<size_t>* sampledFiles{nullptr};
};

class ShardGarbageCollector {
public:
    ShardGarbageCollector() = default;
    ShardGarbageCollector(const ShardGarbageCollector&) = delete;
    ShardGarbageCollector& operator=(const ShardGarbageCollector&) = delete;
    ~ShardGarbageCollector();

    Status Setup(const SpaceLayout* layout, const ShardGCConfig& config);

private:
    void Execute();
    bool ShouldTrigger();
    void ProcessTask(ShardTaskContext& ctx);
    void GCCheckLoop();
    void StopBackgroundCheck();

    const SpaceLayout* layout_{nullptr};
    ShardGCConfig config_;
    std::vector<std::string> shards_;
    ThreadPool<ShardTaskContext> gcPool_;
    std::thread gcCheckWorker_;
    std::mutex gcCheckMtx_;
    std::condition_variable gcCheckCv_;
    std::atomic<bool> stop_{false};
};

}  // namespace UC::PosixStore

#endif
