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
#ifndef UNIFIEDCACHE_YUANRONG_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_YUANRONG_STORE_CC_DUMP_QUEUE_H

#include <atomic>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "datasystem/hetero_client.h"
#include "datasystem/kv/read_only_buffer.h"
#include "datasystem/kv_client.h"
#include "datasystem/utils/optional.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans/device.h"
#include "trans/stream.h"
#include "trans_task.h"
#include "ucmstore_v1.h"
#include "yuanrong_config.h"

namespace UC::YuanRongStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    struct DumpTaskContext {
        TaskPtr task;
        WaiterPtr waiter;
        double submitTime{0.0};
        double prerequisiteStart{0.0};
        double prerequisiteEnd{0.0};
    };

    struct WorkerContext {
        Status status{Status::OK()};
        Trans::Device device;
    };

    struct PersistenceTask {
        Detail::TaskHandle ownerTaskId{0};
        std::vector<std::string> keys;
        Detail::TaskDesc desc;
        double enqueueTime{0.0};
    };

    struct PersistenceContext {
        Detail::TaskHandle ownerTaskId{0};
        Detail::TaskHandle backendTaskId{0};
        size_t bytes{0};
        std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> buffers;
    };

    alignas(64) std::atomic_bool closing_{false};
    alignas(64) std::atomic_bool stopD2H_{false};
    alignas(64) std::atomic_bool stopPersistence_{false};
    alignas(64) std::atomic_size_t pendingCount_{0};
    TaskIdSet* failureSet_{nullptr};
    Config config_{};
    std::shared_ptr<datasystem::HeteroClient> heteroClient_;
    std::shared_ptr<datasystem::KVClient> kvClient_;
    StoreV1* backend_{nullptr};
    SpscRingQueue<DumpTaskContext> ready_;
    SpscRingQueue<PersistenceTask> persistence_;
    std::mutex submitMutex_;
    std::mutex readySubmitMutex_;
    std::unique_ptr<ThreadPool<DumpTaskContext, std::shared_ptr<WorkerContext>>> prerequisitePool_;
    std::thread d2hWorker_;
    std::thread persistenceWorker_;

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet,
                 std::shared_ptr<datasystem::HeteroClient> heteroClient,
                 std::shared_ptr<datasystem::KVClient> kvClient);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void Close();
    void D2HStage(std::promise<Status>& started);
    void PersistenceStage();
    void RunPrerequisite(DumpTaskContext& context, const std::shared_ptr<WorkerContext>& worker);
    void RunD2H(DumpTaskContext&& context);
    void Finish(DumpTaskContext& context, const Status& status);
    Status DumpReadyTask(TaskPtr task, double prerequisiteQueueWaitMs, double prerequisiteMs,
                         double d2hQueueWaitMs, double pipelineStart);
    void Persist(const PersistenceTask& task, size_t& inflightBytes,
                 std::list<PersistenceContext>& inflight);
    Status PersistBatch(const PersistenceTask& task, size_t begin, size_t end,
                        size_t& inflightBytes, std::list<PersistenceContext>& inflight);
    void ReclaimCompletedInflight(size_t& inflightBytes, std::list<PersistenceContext>& inflight);
    void ReleasePersistenceContext(PersistenceContext& context, size_t& inflightBytes);
    static void ReleaseBuffers(
        std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>>& buffers);
};

}  // namespace UC::YuanRongStore

#endif
