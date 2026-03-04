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
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "space_layout.h"
#include "status/status.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"

namespace UC::PosixStore {

struct FileInfo {
    Detail::BlockId blockId;
    time_t mtime;
};

struct MtimeComparator {
    bool operator()(const FileInfo& lhs, const FileInfo& rhs) const
    {
        return lhs.mtime > rhs.mtime;
    }
};

Detail::BlockId HexToBlockId(const char* hexStr);

std::string BlockIdToHex(const Detail::BlockId& blockId);

struct ShardGCConfig {
    double recyclePercent;
    size_t gcConcurrency;
};

struct GCTaskContext {
    std::string shardPath;
    double recyclePercent;
    std::shared_ptr<std::atomic<size_t>> deletedCount;
    std::shared_ptr<Latch> waiter;
};

class ShardGarbageCollector {
public:
    ShardGarbageCollector() = default;
    ShardGarbageCollector(const ShardGarbageCollector&) = delete;
    ShardGarbageCollector& operator=(const ShardGarbageCollector&) = delete;
    ~ShardGarbageCollector();

    Status Setup(const SpaceLayout* layout, const std::vector<std::string>& backends,
                 const ShardGCConfig& config);
    void Trigger();
    void SetGCThreshold(size_t maxFileCount, double thresholdRatio);
    Expected<size_t> Execute();
    bool IsRunning() const;
    bool ShouldTrigger(size_t maxFileCount, double thresholdRatio) const;

private:
    void ProcessShard(GCTaskContext& ctx);
    size_t ScanAndCollectOldestFiles(const std::string& shardPath, double recyclePercent,
                                     std::vector<FileInfo>& filesToDelete);

    const SpaceLayout* layout_{nullptr};
    std::vector<std::string> backends_;
    ShardGCConfig config_;
    ThreadPool<GCTaskContext> gcPool_;
    std::atomic<bool> running_{false};
    std::mutex mtx_;
    std::thread asyncWorker_;
    bool stop_{false};
    size_t maxFileCount_{0};
    double thresholdRatio_{0.0};
};

}  // namespace UC::PosixStore

#endif
