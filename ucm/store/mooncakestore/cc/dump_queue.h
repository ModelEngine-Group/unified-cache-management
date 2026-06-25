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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_DUMP_QUEUE_H

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include "copy_stream.h"
#include "global_config.h"
#include "host_buffer_pool.h"
#include "real_client.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_task.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace UC::MooncakeStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    struct DumpCtx {
        Detail::TaskHandle taskHandle;
        Detail::TaskHandle backendTaskHandle{0};
        std::vector<HostBufferPool::Handle> hostBufs;
    };

public:
    ~DumpQueue();

    Status Setup(const Config& config, TaskIdSet* failureSet,
                 std::shared_ptr<mooncake::RealClient> realClient, StoreV1* backend,
                 HostBufferPool* bufPool);
    void Close();
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage(std::promise<Status>& started);
    void DispatchOneTask(CopyStream& stream, TaskPair&& pair);
    Status DumpOneTask(CopyStream& stream, TaskPtr task);
    Status WaitPrerequisite(TaskPtr task);
    Status PrepareBackendDump(CopyStream& stream, TaskPtr task, DumpCtx& dumpCtx,
                              Detail::TaskDesc& backendTaskDesc);
    Status PutToMooncake(TaskPtr task, const std::vector<std::string>& keys,
                         const std::vector<std::vector<void*>>& allBuffers,
                         const std::vector<std::vector<size_t>>& allSizes);
    Status SubmitBackendDump(CopyStream& stream, TaskPtr task, DumpCtx&& dumpCtx,
                             Detail::TaskDesc&& backendTaskDesc);
    void BackendDumpStage();

    Status DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                   void* host);

private:
    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    TaskIdSet* failureSet_{nullptr};
    std::shared_ptr<mooncake::RealClient> realClient_;
    StoreV1* backend_{nullptr};
    HostBufferPool* bufPool_{nullptr};
    uint32_t replicaNum_{1};
    bool withSoftPin_{false};
    std::vector<size_t> tensorSizes_;
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    std::vector<ssize_t> cpuAffinityCores_;

    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<DumpCtx> dumping_;

    std::thread dispatcher_;
    std::thread dumper_;
};

}  // namespace UC::MooncakeStore

#endif
