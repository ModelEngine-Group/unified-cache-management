#include "logger/logger.h"
#include "compressor_action.h"
#include <chrono>
#include <pthread.h>
#include <queue>

namespace UC::Compressor {

CompressorAction::~CompressorAction()
{
    // 后续改多线程需要在这销毁线程
}

Status CompressorAction::Setup(const Config& config) 
{
    backend_ = config.storeBackend;
    shardSize_ = config.shardSize;
    switch (config.compressRatio) {
        case 32: ratio = R1; break;
        case 24: ratio = R133; break;
        case 23: ratio = R139; break;
        case 22: ratio = R145; break;
        case 21: ratio = R152; break;
        case 16: ratio = R200; break;
        default: return Status::InvalidParam("invalid compressRatio({})", config.compressRatio);
    }

    switch (config.dataType) {
        case 0: dataType = DT_BF16; break;
        default: return Status::InvalidParam("invalid compress dataType({})", config.dataType);
    }
    
    // init thread pool
    dump_pool_.SetNWorker(config.streamNumber/2)
              .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
              .Run();
    
    threadBuf_ = std::make_unique<uint8_t[]>(shardSize_);
    return Status::OK();
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    UC_DEBUG("task {}, push size is {}", task->id, task->desc.size());
    // waiter->Set(1);
    if (task->type == TransTask::Type::DUMP) {
        dump_pool_.Push(CompressTask {
            task,
            waiter
        });
    } else {
        // load_pool_.Push(CompressTask {
        //     task,
        //     waiter
        // });
    }
}

void CompressorAction::Compress_Load(TaskPtr t, WaiterPtr w)
{
    #ifdef USE_C_COMPRESS
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = backend_->Load(t->desc);
    if (result.Value() > 0) {
        backend_->Wait(result.Value());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    UC_INFO("task id {} backend load and wait time: {} us", t->id, t1 - t0);

    auto t3 = std::chrono::high_resolution_clock::now();
    const auto& shards = t->desc;
    const size_t sz = shards.size();
    const size_t numThreads = 4;   // 线程数量
    std::vector<std::thread> threads;
    const size_t batch_sz = (sz + numThreads - 1) / numThreads;
    for (size_t t_id = 0; t_id < numThreads; ++t_id) {
        size_t start = t_id * batch_sz;
        size_t end = std::min(start + batch_sz, sz);
        if (start < end) {
            // 直接创建线程并执行逻辑
            threads.emplace_back([&, start, end]() {
                // 每个线程私有的缓冲区，防止竞争
                std::unique_ptr<uint8_t[]> localBuf(new uint8_t[shardSize_]); 
                for (size_t i = start; i < end; ++i) {
                    auto& s = shards[i];
                    uint8_t* src = static_cast<uint8_t*>(s.addrs[0]);
                    
                    if (ratio == R1) {
                        continue;
                    } else {
                        size_t n_bf16 = shardSize_ >> 1;
                        TunstallDecompressBF16((uint16_t*)localBuf.get(), src, n_bf16);
                        memcpy(src, localBuf.get(), shardSize_);
                    }
                }
            });
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    UC_INFO("create threads time: {}", t4 - t3);

    auto t5 = std::chrono::high_resolution_clock::now();
    for (auto& th : threads) th.join();
    auto t6 = std::chrono::high_resolution_clock::now();
    UC_INFO("threads finish time: {} thread num({})", t6- t5, numThreads);
#else
    // to posix load
    /* 原路径：直接调用 PosixStore */
    auto result = backend_->Load(std::move(t->desc));
    backend_->Wait(result.Value());
    UC_INFO("COMPRESS LOAD END.");
#endif
    // UC_INFO("COMPRESS LOAD END, task: {}", ct.task->id);
    w->Done();
}

void CompressorAction::Compress_Dump(CompressTask& ct)
{
#ifdef USE_C_COMPRESS
    UC_INFO("COMPRESS DUMP STARTING...");
    const auto& desc = ct.task->desc;
    if (desc.empty()) {
        UC_INFO("COMPRESS DUMP desc is empty...");
        return;
    }

    size_t srcSize = shardSize_;
    size_t compBufSize = srcSize + 4096;              // 压缩后缓冲区的可用大小
    
    Detail::TaskDesc backendDesc;
    backendDesc.brief = ct.task->desc.brief;
    std::vector<void*> blockToFree;
    std::unique_ptr<MemoryPool> dump_memoryPool_ = std::make_unique<MemoryPool>(compBufSize, ct.task->desc.size());
    if (!dump_memoryPool_) {
        UC_INFO("Out of memory: failed to allocate {} B", shardSize_ * ct.task->desc.size());
        Status::NoSpace();
    }

    for (const UC::Detail::Shard& s : desc) {
        UC_INFO("Task id: {} Shard index: {}  Compress start...", ct.task->id, s.index);

        uint8_t* compBuf = static_cast<uint8_t*>(dump_memoryPool_->allocate());
        uint16_t* src = static_cast<uint16_t*>(s.addrs[0]);

        size_t compBytes = 0;
        if (ratio == R1) {
            memcpy(compBuf, src, srcSize);
            compBytes = srcSize;
        } else {
            size_t n_bf16 = shardSize_ >> 1;
            compBytes = TunstallCompressBF16(compBuf, (uint16_t*)src, n_bf16);
        }
        
        std::vector<void*> _addrs{static_cast<void*>(compBuf)};

        backendDesc.push_back(Detail::Shard {
            s.owner,
            s.index,
            _addrs
        });

        UC_INFO("Shard index: {} compress end...  compBytes is {}", s.index, compBytes);
        blockToFree.push_back(static_cast<void*>(compBuf));
    }

    auto res = backend_->Dump(std::move(backendDesc));

    if (!blockToFree.empty() && res.Value() > 0) {
        backend_->Wait(res.Value());
        dump_memoryPool_->deallocate(blockToFree);
    }

    UC_INFO("COMPRESS DUMP END.");
#else
    // to posix dump
    const auto n = ct.task->desc.size();
    if (n > 0) 
    {
        backend_->Dump(std::move(ct.task->desc));
    }

    UC_INFO("COMPRESS DUMP END.");
#endif
    ct.waiter->Done();
}

}