/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the project is
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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_TRANS_MANAGER_H

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "global_config.h"
#include "host_buffer_pool.h"
#include "real_client.h"
#include "thread/thread_pool.h"
#include "trans_task.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace UC::MooncakeStore {

/**
 * Dump pipeline (4 stages):
 *   ProcessDump → GetBuffer → Submit → Wait
 *
 * Load pipeline (4 stages):
 *   ProcessLoad → Wait → Put → Get
 */
class TransManager {
public:
    TransManager();
    ~TransManager();

    Status Setup(const Config& config);
    void Close();

    void ProcessTask(Detail::TaskHandle handle, TransTask& task, std::shared_ptr<TaskState> state,
                     HostBufferPool& bufPool);

    std::shared_ptr<mooncake::RealClient> GetRealClient() const;

private:
    Status SetupRealClient(const Config& config);
    Status SetupBackendPipeline(const Config& config);
    void ProcessDump(TransTask& task, std::string& err);
    void ProcessLoad(TransTask& task, std::shared_ptr<TaskState> state, HostBufferPool& bufPool);

    struct DumpCtx {
        std::vector<std::string> keys;
        std::vector<TransShard> shards;
    };
    struct DumpSubmitCtx {
        std::vector<std::string> keys;
        std::vector<TransShard> shards;
        std::vector<void*> bufferPtrs;
    };
    struct DumpWaitCtx {
        Detail::TaskHandle posixHandle;
        size_t shardCount;
    };

    void EnqueueBackendDump(DumpCtx ctx);
    void OnDumpGetBuffer(DumpCtx& ctx, auto&);
    void OnDumpSubmit(DumpSubmitCtx& ctx, auto&);
    void OnDumpWait(DumpWaitCtx& ctx, auto&);

    struct LoadWaitCtx {
        std::shared_ptr<TaskState> state;
        Detail::TaskHandle posixHandle;
        std::vector<std::string> keys;
        std::vector<TransShard> shards;
        std::vector<void*> hostBufs;
        std::vector<size_t> hostBufSizes;
        HostBufferPool* bufPool;
    };
    struct LoadPutCtx {
        std::shared_ptr<TaskState> state;
        std::vector<std::string> keys;
        std::vector<TransShard> shards;
        std::vector<void*> hostBufs;
        std::vector<size_t> hostBufSizes;
        HostBufferPool* bufPool;
    };
    struct LoadGetCtx {
        std::shared_ptr<TaskState> state;
        std::vector<std::string> keys;
        std::vector<TransShard> shards;
        std::vector<void*> hostBufs;
        HostBufferPool* bufPool;
    };

    void EnqueueLoadTransfer(LoadWaitCtx ctx);
    void OnLoadWait(LoadWaitCtx& ctx, auto&);
    void OnLoadPut(LoadPutCtx& ctx, auto&);
    void OnLoadGet(LoadGetCtx& ctx, auto&);

    void BuildBatchFromShards(const std::vector<TransShard>& shards, std::vector<std::string>& keys,
                              std::vector<std::vector<void*>>& buffers,
                              std::vector<std::vector<size_t>>& sizes);

    bool CheckMooncakeHit(const std::vector<int>& results);
    Status BuildMissShardBatch(const std::vector<TransShard>& shards,
                               const std::vector<int>& results, HostBufferPool& bufPool,
                               LoadWaitCtx& ctx, Detail::TaskDesc& desc);
    void SubmitPosixLoad(LoadWaitCtx& ctx, const Detail::TaskDesc& desc,
                         std::shared_ptr<TaskState> state, HostBufferPool& bufPool);

    void BuildPutBatch(const LoadPutCtx& ctx, std::vector<std::span<const char>>& putValues);
    void DoPutBatch(const LoadPutCtx& ctx, const std::vector<std::span<const char>>& putValues,
                    LoadGetCtx& gctx);

    void BuildGetBatch(const LoadGetCtx& ctx, std::vector<std::vector<void*>>& getBuffers,
                       std::vector<std::vector<size_t>>& getSizes);
    void DoGetInto(LoadGetCtx& ctx, const std::vector<std::vector<void*>>& getBuffers,
                   const std::vector<std::vector<size_t>>& getSizes);

    std::shared_ptr<mooncake::RealClient> realClient_;
    StoreV1* backend_{nullptr};
    Config config_;

    std::unique_ptr<ThreadPool<DumpCtx>> dumpGetBufferPool_;
    std::unique_ptr<ThreadPool<DumpSubmitCtx>> dumpSubmitPool_;
    std::unique_ptr<ThreadPool<DumpWaitCtx>> dumpWaitPool_;

    std::unique_ptr<ThreadPool<LoadWaitCtx>> loadWaitPool_;
    std::unique_ptr<ThreadPool<LoadPutCtx>> loadPutPool_;
    std::unique_ptr<ThreadPool<LoadGetCtx>> loadGetPool_;

    std::atomic<bool> stopFlag_{false};
    std::atomic<bool> closed_{false};
};

}  // namespace UC::MooncakeStore

#endif
