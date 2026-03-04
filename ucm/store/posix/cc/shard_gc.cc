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
#include <algorithm>
#include <dirent.h>
#include <random>
#include <sys/stat.h>
#include "logger/logger.h"
#include "posix_file.h"
#include "template/topn_heap.h"

namespace UC::PosixStore {

static constexpr size_t kMinRecycleNum = 1;

static constexpr char HEX_CHARS[] = "0123456789abcdef";

Detail::BlockId HexToBlockId(const char* hexStr)
{
    Detail::BlockId blockId;
    for (size_t i = 0; i < 16; ++i) {
        uint8_t high = static_cast<uint8_t>(hexStr[i * 2]);
        uint8_t low = static_cast<uint8_t>(hexStr[i * 2 + 1]);

        high = (high <= '9') ? (high - '0') : (high - 'a' + 10);
        low = (low <= '9') ? (low - '0') : (low - 'a' + 10);

        blockId[i] = static_cast<std::byte>((high << 4) | low);
    }
    return blockId;
}

std::string BlockIdToHex(const Detail::BlockId& blockId)
{
    std::string result;
    result.reserve(32);
    for (size_t i = 0; i < 16; ++i) {
        uint8_t byte = static_cast<uint8_t>(blockId[i]);
        result.push_back(HEX_CHARS[byte >> 4]);
        result.push_back(HEX_CHARS[byte & 0x0F]);
    }
    return result;
}

ShardGarbageCollector::~ShardGarbageCollector()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    if (asyncWorker_.joinable()) { asyncWorker_.join(); }
}

Status ShardGarbageCollector::Setup(const SpaceLayout* layout,
                                    const std::vector<std::string>& backends,
                                    const ShardGCConfig& config)
{
    if (!layout) { return Status::InvalidParam("layout is null"); }
    if (backends.empty()) { return Status::InvalidParam("backends is empty"); }
    if (config.recyclePercent <= 0 || config.recyclePercent > 1.0f) {
        return Status::InvalidParam("invalid recycle percent");
    }
    if (config.gcConcurrency == 0) { return Status::InvalidParam("invalid gc concurrency"); }

    layout_ = layout;
    backends_ = backends;
    config_ = config;

    auto success = gcPool_.SetWorkerFn([this](GCTaskContext& ctx, auto&) { ProcessShard(ctx); })
                       .SetNWorker(config.gcConcurrency)
                       .Run();

    if (!success) { return Status::Error("failed to start gc thread pool"); }

    UC_DEBUG("ShardGC setup: recyclePercent={}, concurrency=", config.recyclePercent,
             config.gcConcurrency);
    return Status::OK();
}

void ShardGarbageCollector::SetGCThreshold(size_t maxFileCount, double thresholdRatio)
{
    maxFileCount_ = maxFileCount;
    thresholdRatio_ = thresholdRatio;
}

void ShardGarbageCollector::Trigger()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (running_.load()) { return; }
    if (asyncWorker_.joinable()) { asyncWorker_.join(); }
    asyncWorker_ = std::thread([this] {
        size_t totalDeleted = 0;
        while (true) {
            auto result = Execute();
            if (result) {
                size_t deleted = result.Value();
                totalDeleted += deleted;
                UC_DEBUG("GC deleted {} files", deleted);
                UC_DEBUG("GC total deleted {} files", totalDeleted);
                if (maxFileCount_ > 0 && ShouldTrigger(maxFileCount_, thresholdRatio_)) {
                    UC_DEBUG("GC continuing, file count still exceeds threshold");
                    continue;
                } else {
                    UC_DEBUG("GC finished, file count within threshold");
                    break;
                }
            } else {
                break;
            }
        }
    });
}

Expected<size_t> ShardGarbageCollector::Execute()
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return Status::Error("GC is already running");
    }

    auto shards = layout_->RelativeRoots();
    if (shards.empty()) {
        running_.store(false);
        return Status::Error("no shard directories found");
    }

    size_t totalTasks = shards.size() * backends_.size();

    std::shared_ptr<std::atomic<size_t>> deletedCount;
    std::shared_ptr<Latch> waiter;

    try {
        deletedCount = std::make_shared<std::atomic<size_t>>(0);
        waiter = std::make_shared<Latch>();
    } catch (const std::exception& e) {
        running_.store(false);
        return Status::OutOfMemory();
    }

    waiter->Set(totalTasks);

    for (const auto& backend : backends_) {
        for (const auto& shard : shards) {
            std::string shardPath = backend;
            if (shardPath.back() != '/') { shardPath += '/'; }
            shardPath += shard;
            gcPool_.Push({shardPath, config_.recyclePercent, deletedCount, waiter});
        }
    }

    waiter->Wait();
    running_.store(false);

    size_t deleted = deletedCount->load();
    return deleted;
}

bool ShardGarbageCollector::IsRunning() const { return running_.load(); }

bool ShardGarbageCollector::ShouldTrigger(size_t maxFileCount, double thresholdRatio) const
{
    if (!layout_ || maxFileCount == 0) { return false; }

    auto shards = layout_->RelativeRoots();
    if (shards.empty()) { return false; }

    size_t totalShards = shards.size();
    size_t sampleCount = std::max(static_cast<size_t>(1), static_cast<size_t>(totalShards * 0.1));

    std::vector<size_t> shardIndices(totalShards);
    for (size_t i = 0; i < totalShards; ++i) { shardIndices[i] = i; }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shardIndices.begin(), shardIndices.end(), gen);
    shardIndices.resize(sampleCount);

    std::atomic<size_t> sampledFiles{0};
    std::vector<std::thread> threads;

    size_t threadsCount = std::min(static_cast<size_t>(16), sampleCount);
    size_t shardsPerThread = (sampleCount + threadsCount - 1) / threadsCount;

    for (size_t t = 0; t < threadsCount; ++t) {
        threads.emplace_back([&, t, shardsPerThread]() {
            size_t start = t * shardsPerThread;
            size_t end = std::min(start + shardsPerThread, sampleCount);
            size_t localCount = 0;

            for (size_t i = start; i < end; ++i) {
                size_t shardIdx = shardIndices[i];
                for (const auto& backend : backends_) {
                    std::string shardPath = backend;
                    if (shardPath.back() != '/') { shardPath += '/'; }
                    shardPath += shards[shardIdx];

                    DIR* dir = opendir(shardPath.c_str());
                    if (!dir) { continue; }
                    struct dirent* entry;
                    while ((entry = readdir(dir)) != nullptr) {
                        if (entry->d_name[0] == '.') { continue; }
                        ++localCount;
                    }
                    closedir(dir);
                }
            }
            sampledFiles.fetch_add(localCount, std::memory_order_relaxed);
        });
    }

    for (auto& th : threads) {
        if (th.joinable()) { th.join(); }
    }

    size_t sampledFileCount = sampledFiles.load();
    size_t estimatedFiles = sampledFileCount * totalShards / sampleCount;
    size_t avgFilesPerShard = estimatedFiles / totalShards;
    size_t thresholdFilesPerShard = maxFileCount / totalShards;

    UC_DEBUG("GC sampling: sampled {}/{} shards, sampledFiles={}, estimatedFiles={}", sampleCount,
             totalShards, sampledFileCount, estimatedFiles);
    UC_INFO("GC avgFilesPerShard={}, trigger={}", avgFilesPerShard,
            avgFilesPerShard >= static_cast<size_t>(thresholdFilesPerShard * thresholdRatio));

    return avgFilesPerShard >= static_cast<size_t>(thresholdFilesPerShard * thresholdRatio);
}

void ShardGarbageCollector::ProcessShard(GCTaskContext& ctx)
{
    std::vector<FileInfo> filesToDelete;
    size_t scanned = ScanAndCollectOldestFiles(ctx.shardPath, ctx.recyclePercent, filesToDelete);

    if (scanned == 0 || filesToDelete.empty()) {
        ctx.waiter->Done();
        return;
    }

    size_t deleted = 0;
    for (const auto& file : filesToDelete) {
        std::string filePath = ctx.shardPath + "/" + BlockIdToHex(file.blockId);
        PosixFile f{filePath};
        f.Remove();
        deleted++;
    }

    ctx.deletedCount->fetch_add(deleted, std::memory_order_relaxed);
    ctx.waiter->Done();
}

size_t ShardGarbageCollector::ScanAndCollectOldestFiles(const std::string& shardPath,
                                                        double recyclePercent,
                                                        std::vector<FileInfo>& filesToDelete)
{
    DIR* dir = opendir(shardPath.c_str());
    if (!dir) { return 0; }

    std::vector<FileInfo> allFiles;
    allFiles.reserve(1024);

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') { continue; }
        std::string filePath = shardPath + "/" + entry->d_name;
        struct stat st;
        if (stat(filePath.c_str(), &st) != 0) { continue; }
        if (!S_ISREG(st.st_mode)) { continue; }
        allFiles.push_back({HexToBlockId(entry->d_name), st.st_mtime});
    }
    closedir(dir);

    size_t totalFiles = allFiles.size();
    if (totalFiles == 0) { return totalFiles; }

    size_t recycleNum = static_cast<size_t>(totalFiles * recyclePercent);
    if (recycleNum < kMinRecycleNum) { recycleNum = kMinRecycleNum; }
    if (recycleNum > totalFiles) { recycleNum = totalFiles; }

    auto heap = std::make_unique<TopNHeap<FileInfo, MtimeComparator>>(recycleNum);
    for (const auto& file : allFiles) { heap->Push(file); }

    filesToDelete.reserve(recycleNum);
    while (!heap->Empty()) {
        filesToDelete.push_back(heap->Top());
        heap->Pop();
    }

    return totalFiles;
}

}  // namespace UC::PosixStore
