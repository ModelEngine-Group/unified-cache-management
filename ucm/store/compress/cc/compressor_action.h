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

namespace UC::Compressor {

#define USE_C_COMPRESS

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;

private:
    StoreV1* backend_{nullptr};
    size_t shardSize_{0};
    FixedRatio ratio{R145};
    
    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
    };
    ThreadPool<CompressTask> dump_pool_;
    ThreadPool<CompressTask> load_pool_;

public:
    ~CompressorAction();
    Status Setup(const Config& config);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void Compress_Load(CompressTask& ios);
    void Compress_Dump(CompressTask& ios);

};

}

#endif
