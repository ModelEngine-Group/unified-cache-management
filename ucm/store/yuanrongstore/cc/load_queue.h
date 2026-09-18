/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_LOAD_QUEUE_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_LOAD_QUEUE_H

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include "backfill_queue.h"
#include "copy_stream.h"
#include "datasystem/hetero_client.h"
#include "datasystem/kv_client.h"
#include "host_buffer_pool.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_task.h"
#include "ucmstore_v1.h"
#include "yuanrong_config.h"

namespace UC::YuanRongStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    enum class RecoverySource : uint8_t { LOOKUP_MISS, LOAD_FALLBACK };
    struct LoadStats {
        size_t requested{0};
        size_t yuanrongSuccess{0};
        size_t lookupMissPosixSuccess{0};
        size_t loadFallbackPosixSuccess{0};

        void Record() const;
    };
    struct HostBatch {
        std::vector<size_t> indexes;
        std::vector<std::string> keys;
        std::vector<HostBufferPool::Handle> hostBuffers;
        Detail::TaskHandle backendTaskHandle{0};
        Status status{Status::OK()};
    };

    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    Config config_{};
    std::shared_ptr<datasystem::HeteroClient> heteroClient_;
    std::shared_ptr<datasystem::KVClient> kvClient_;
    StoreV1* backend_{nullptr};
    SpscRingQueue<TaskPair> waiting_;
    std::vector<std::unique_ptr<SpscRingQueue<TaskPair>>> running_;
    std::mutex submitMutex_;
    std::thread dispatcher_;
    std::vector<std::thread> workers_;
    size_t nextWorker_{0};
    HostBufferPool hostBufferPool_;
    BackfillQueue backfillQueue_;

public:
    ~LoadQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet,
                 std::shared_ptr<datasystem::HeteroClient> heteroClient,
                 std::shared_ptr<datasystem::KVClient> kvClient);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void Close();
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void WorkerStage(size_t workerIndex, std::promise<Status> started);
    void RunOne(CopyStream& stream, TaskPair&& pair);
    Status LoadOne(CopyStream& stream, TaskPtr task, LoadStats& stats);
    Status LoadThenRecover(CopyStream& stream, TaskPtr task, const std::vector<std::string>& keys,
                           const std::vector<datasystem::DeviceBlobList>& blobLists,
                           double taskStart, LoadStats& stats);
    Status RecoverFromBackend(CopyStream& stream, TaskPtr task,
                              const std::vector<std::string>& keys,
                              const std::vector<datasystem::DeviceBlobList>& blobLists,
                              const std::vector<size_t>& missIndexes, RecoverySource source,
                              LoadStats& stats);
    HostBatch PrepareHostBatch(TaskPtr task, const std::vector<std::string>& keys,
                               const std::vector<size_t>& missIndexes, size_t begin, size_t end);
    Status FinalizeHostBatch(CopyStream& stream,
                             const std::vector<datasystem::DeviceBlobList>& blobLists,
                             HostBatch& batch);
    Status HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                    const datasystem::DeviceBlobList& blobList);
};

}  // namespace UC::YuanRongStore

#endif
