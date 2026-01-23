#include "logger/logger.h"
#include "compressor_action.h"




namespace UC::Compressor {

CompressorAction::~CompressorAction()
{
    // 后续改多线程需要在这销毁线程
}

Status CompressorAction::Setup(const Config& config) 
{
    backend_ = static_cast<StoreV1*>((void*)config.storeBackend);
    shardSize_ = config.shardSize;
    switch (config.compressRatio) {
        case 23: ratio = R139; break;
        case 22: ratio = R145; break;
        case 21: ratio = R152; break;
        default: return Status::InvalidParam("invalid compressRatio({})", config.compressRatio);
    }
    
    // init thread pool
    dump_pool_.SetNWorker(config.streamNumber/2)
              .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
              .Run();
    load_pool_.SetNWorker(config.streamNumber/2)
              .SetWorkerFn([this](auto& ct, auto&) { Compress_Load(ct); })
              .Run();

    memoryPool_ = std::make_unique<MemoryPool>(shardSize_, 32);

    return Status::OK();
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    waiter->Set(task->desc.size());
    if (task->type == TransTask::Type::DUMP) {
        dump_pool_.Push(CompressTask {
            task,
            waiter
        });
    } else {
        load_pool_.Push(CompressTask {
            task,
            waiter
        });
        // waiter->Wait();
        // UC_DEBUG("load wait.");
    }
}

void CompressorAction::Compress_Load(CompressTask& ct)
{
    UC_DEBUG("COMPRESS LOAD START, task {}", ct.task->id);
#ifdef USE_C_COMPRESS
    const auto desc = ct.task->desc;
    auto result = backend_->Load(std::move(ct.task->desc));
    UC_DEBUG("result {} desc size {}", result.Value(), desc.size());
    if (result.Value() > 0) {
        backend_->Wait(result.Value());
    }

    size_t totalDecompBytes = 0;             // 总解压字节数（日志用）

    size_t srcSize = (shardSize_ * (size_t)ratio / 32) / 4096 * 4096;;
    size_t decompBufSize = shardSize_;
    void* decompBuf = malloc(decompBufSize);

    for (const UC::Detail::Shard& s : desc) {
        uint8_t* src = static_cast<uint8_t*>(s.addrs[0]);
        UC_DEBUG("s.addrs {}  s.addrs.size {}", s.addrs[0], s.addrs.size());

        size_t decompBytes = HUF_decompress_float_fixRatio(decompBuf, decompBufSize, src, srcSize, NULL);

        memcpy(s.addrs[0], decompBuf, decompBytes);

        totalDecompBytes += decompBytes;

        // {
        //     FILE *fp = fopen("cpu_decompressed.bin", "ab");          // a=append, b=binary, 不存在则创建
        //     if (!fp) { perror("fopen"); return; }
        //     size_t written = fwrite(decompBuf, 1, decompBytes, fp);
        //     fclose(fp);
        //     if (written != decompBytes) { perror("fwrite"); }
        // }

        // {
        //     FILE *fp = fopen("npu_decompressed.bin", "ab");          // a=append, b=binary, 不存在则创建
        //     if (!fp) { perror("fopen"); return; }
        //     size_t written = fwrite(s.addrs[0], 1, decompBytes, fp);
        //     fclose(fp);
        //     if (written != decompBytes) { perror("fwrite"); }
        // }

        UC_DEBUG("COMPRESS LOAD END.... decompBytes {}", decompBytes);
    }

    if (decompBuf) free(decompBuf);

    UC_DEBUG("COMPRESS LOAD END. Total decompressed bytes: {}", totalDecompBytes);
#else
    // to posix load
    /* 原路径：直接调用 PosixStore */
    backend_->Load(std::move(ct.task->desc));
    UC_DEBUG("COMPRESS LOAD END.");
#endif
    UC_DEBUG("COMPRESS LOAD END, task: {}", ct.task->id);
    ct.waiter->Done();
}


void CompressorAction::Compress_Dump(CompressTask& ct)
{
    UC_DEBUG("COMPRESS DUMP START.");
#ifdef USE_C_COMPRESS
    UC_DEBUG("COMPRESS DUMP STARTING...");
    const auto& desc = ct.task->desc;
    if (desc.empty()) return;

    size_t srcSize = shardSize_;
    size_t compBufSize = srcSize + 4096;              // 压缩后缓冲区的可用大小
    uint8_t* compBuf = static_cast<uint8_t*>(memoryPool_->allocate());

    Detail::TaskDesc backendDesc;
    backendDesc.brief = ct.task->desc.brief;
    for (const UC::Detail::Shard& s : desc) {
        uint16_t* src = static_cast<uint16_t*>(s.addrs[0]);

        UC_DEBUG(" s.index {}  s.addrs.size {} s.addrs.data {}", s.index, s.addrs.size(), static_cast<const void*>(s.addrs.data()));

        size_t compBytes = HUF_compress_float_fixRatio (compBuf, compBufSize, src, srcSize, R145, DT_BF16);

        std::vector<void*> _addrs{static_cast<void*>(compBuf)};

        backendDesc.push_back(Detail::Shard {
            s.owner,
            s.index,
            _addrs
        });

        UC_DEBUG("COMPRESS DUMP ...  compBytes is {}", compBytes);
    }

    auto res = backend_->Dump(std::move(backendDesc));

    if (compBuf && res.Value() > 0) {
        backend_->Wait(res.Value());
        memoryPool_->deallocate(static_cast<void*>(compBuf));
    }

    UC_DEBUG("COMPRESS DUMP END.");
#else
    // to posix dump
    const auto n = ct.task->desc.size();
    if (n > 0) 
    {
        backend_->Dump(std::move(ct.task->desc));
    }

    UC_DEBUG("COMPRESS DUMP END.");
#endif
    ct.waiter->Done();
}

}
