#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <unistd.h> 
#include "global_config.h"
#include "trans_task.h"
#include "thread/latch.h"
#include "ucmstore_v1.h"
#include "thread/thread_pool.h"
#include "compress_lib/huf.h"  // HUF_compress_float_fixRatio, HUF_decompress_float_fixRatio
#include "memory_pool.h"
#include "template/spsc_ring_queue.h"
#include <condition_variable>


#include "compress_lib/tunstall_bf16.h"  // HUF_compress_float_fixRatio, HUF_decompress_float_fixRatio

namespace UC::Compressor {

#define USE_C_COMPRESS

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    // 2
    struct ShardTask {
        Detail::TaskHandle taskHandle;
        Detail::Shard* shard;
        Detail::TaskHandle backendTaskHandle;
        WaiterPtr waiter;
    };

private:
    StoreV1* backend_{nullptr};
    size_t shardSize_{0};
    FixedRatio ratio{R145};
    DataType dataType{DT_INVALID};
    
    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
    };
    ThreadPool<CompressTask> dump_pool_;
    ThreadPool<CompressTask> load_pool_;
    std::unique_ptr<uint8_t[]> threadBuf_{0};
    // 2
    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    // 3
    std::mutex waiterMtx_; // waiter的锁
    std::mutex backendMtx_; // backend->wait锁
    std::atomic<Detail::TaskHandle> backendTaskHandle_{0};
    std::condition_variable cv_; 

public:
    ~CompressorAction();
    Status Setup(const Config& config);
    void Push(TaskPtr task, WaiterPtr waiter);
    void Compress_Load(TaskPtr t, WaiterPtr w);

private:
    // void Compress_Load(CompressTask& ios);
    void Compress_Dump(CompressTask& ios);
    // void LoadWorker(ComputeBatchTask& cbt);
    // 2
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    // void TransferStage(std::promise<Status>& started);
    void TransferOneTask(ShardTask&& task);
    // 3


};

}

#endif