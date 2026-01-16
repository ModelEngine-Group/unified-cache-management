#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <unistd.h> 
#include "global_config.h"
#include "trans_task.h"
#include "thread/latch.h"
#include "ucmstore_v1.h"
#include "thread/thread_pool.h"

namespace UC::Compressor {


// #define USE_C_COMPRESS

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;

private:
    StoreV1* backend_{nullptr};
    size_t shardSize_{0};
    static constexpr std::size_t pageSize = 4096;
    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
    };
    ThreadPool<CompressTask> dump_pool_;
    ThreadPool<CompressTask> load_pool_;
public:
    ~CompressorAction();
    Status Setup(const Config& config);
    // void Compress_Load(TaskPtr task, WaiterPtr waiter);
    // void Compress_Dump(TaskPtr task, WaiterPtr waiter);
    void Push(TaskPtr task, WaiterPtr waiter);

    void print_byte_binary(uint8_t b) {
        for (int i = 7; i >= 0; i--) {
            putchar((b & (1 << i)) ? '1' : '0');
        }
    }
    // 打印前 n 个字节的二进制
    void print_binary_block(const void* buf, size_t n) {
        const uint8_t* p = (const uint8_t*)buf;
        for (size_t i = 0; i < n; i++) {
            print_byte_binary(p[i]);
            putchar(' ');
            if ((i+1) % 8 == 0) putchar('\n'); // 每 8 个字节换行
        }
    }

private:
    void Compress_Load(CompressTask& ios);
    void Compress_Dump(CompressTask& ios);

};

}

#endif
