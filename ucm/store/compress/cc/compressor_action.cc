#include "compressor_action.h"
#include <chrono>
#include <pthread.h>
#include <queue>
#include "logger/logger.h"

namespace UC::Compressor {

CompressorAction::~CompressorAction()
{
    // 后续改多线程需要在这销毁线程
}

Status CompressorAction::Setup(const Config& config)
{
    backend_ = config.storeBackend;
    shardSize_ = config.shardSize;
    decompressThreadNum = config.decompressThreadNum;

    switch (config.compressRatio) {
        case 32: ratio = R1; break;
        case 16: ratio = R200; break;
        default: return Status::InvalidParam("Invalid compressRatio({})", config.compressRatio);
    }

    switch (config.dataType) {
        case 0: dataType = DT_BF16; break;
        default: return Status::InvalidParam("Invalid compress dataType({})", config.dataType);
    }

    // init thread pool
    dump_pool_.SetNWorker(config.streamNumber >> 1)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
        .Run();

    load_pool_.SetNWorker(decompressThreadNum)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Load(ct); })
        .Run();

    UC_DEBUG("Compressor Setup OK | load_threads: {}, shard_size: {} B", decompressThreadNum,
             shardSize_);

    threadBuf_ = std::make_unique<uint8_t[]>(shardSize_);
    return Status::OK();
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    const char* type = (task->type == TransTask::Type::DUMP) ? "DUMP" : "LOAD";
    UC_DEBUG("Task Pushed | id: {}, type: {}, shards: {}", task->id, type, task->desc.size());

    waiter->Set(1);
    if (task->type == TransTask::Type::DUMP) {
        dump_pool_.Push(CompressTask{task, waiter});
    } else if (task->type == TransTask::Type::LOAD) {
        load_pool_.Push(CompressTask{task, waiter});
    }
}

void CompressorAction::Compress_Load(CompressTask& ct)
{
    UC_DEBUG("COMPRESS LOAD START | task_id: {}", ct.task->id);
    if (ratio == R1) {
        auto result = backend_->Load(std::move(ct.task->desc));
        backend_->Wait(result.Value());
    } else {
        auto result = backend_->Load(ct.task->desc);
        if (result.Value() > 0) { backend_->Wait(result.Value()); }

        const auto& shards = ct.task->desc;
        UC_DEBUG("COMPRESS LOAD | shards_count: {}", shards.size());

        for (size_t i = 0; i < shards.size(); ++i) {
            if (ratio == R200) {
                size_t n_bf16 = shardSize_ >> 1;
                int err = TunstallDecompressBF16Inplace(static_cast<uint8_t*>(shards[i].addrs[0]),
                                                        n_bf16);
                if (err != 0) {
                    UC_ERROR("COMPRESS LOAD FAILED | shard: {}, error: {}", shards[i].index, err);
                    continue;
                }
                UC_DEBUG("COMPRESS LOAD | shard: {}, done, decompressed_size: {}", shards[i].index,
                         shardSize_);
            }
        }
    }

    ct.waiter->Done();
    UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
}

void CompressorAction::Compress_Dump(CompressTask& ct)
{
    UC_DEBUG("COMPRESS DUMP START | task_id: {}", ct.task->id);

    if (ratio == R1) {
        const auto n = ct.task->desc.size();
        if (n > 0) { backend_->Dump(std::move(ct.task->desc)); }
    } else {
        const auto& desc = ct.task->desc;
        if (desc.empty()) {
            UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, desc is empty", ct.task->id);
            ct.waiter->Done();
            return;
        }

        size_t srcSize = shardSize_;
        size_t compBufSize = srcSize + 4096;

        Detail::TaskDesc backendDesc;
        backendDesc.brief = ct.task->desc.brief;
        std::vector<void*> blockToFree;
        std::unique_ptr<MemoryPool> dump_memoryPool_ =
            std::make_unique<MemoryPool>(compBufSize, ct.task->desc.size());
        if (!dump_memoryPool_) {
            UC_ERROR("COMPRESS DUMP OOM | task_id: {}, required: {} B", ct.task->id,
                     shardSize_ * desc.size());
            Status::NoSpace();
        }

        for (const UC::Detail::Shard& s : desc) {
            UC_DEBUG("COMPRESS DUMP | task_id: {}, shard: {}, compressing...", ct.task->id,
                     s.index);

            uint8_t* compBuf = static_cast<uint8_t*>(dump_memoryPool_->allocate());
            uint16_t* src = static_cast<uint16_t*>(s.addrs[0]);

            size_t compBytes = 0;
            if (ratio == R200) {
                size_t n_bf16 = shardSize_ >> 1;
                int err = TunstallCompressBF16(compBuf, src, n_bf16);
                if (err != 0) [[unlikely]] {
                    UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, shard: {}, error: {}",
                             ct.task->id, s.index, err);
                }
                compBytes = n_bf16;
            }

            std::vector<void*> _addrs{static_cast<void*>(compBuf)};

            backendDesc.push_back(Detail::Shard{s.owner, s.index, _addrs});

            UC_DEBUG("COMPRESS DUMP | shard: {}, done, compressed_size: {}", s.index, compBytes);
            blockToFree.push_back(static_cast<void*>(compBuf));
        }

        auto res = backend_->Dump(std::move(backendDesc));

        if (!blockToFree.empty() && res.Value() > 0) {
            backend_->Wait(res.Value());
            dump_memoryPool_->deallocate(blockToFree);
        }
    }

    ct.waiter->Done();
    UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
}

}  // namespace UC::Compressor