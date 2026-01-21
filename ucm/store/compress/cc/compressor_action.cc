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
    uint8_t* compBuf = (uint8_t*)malloc(compBufSize);

    // 测试代码
    // void* decompBuf = malloc(srcSize);

    size_t totalCompBytes = 0;

    for (const UC::Detail::Shard& s : desc) {
        uint16_t* src = static_cast<uint16_t*>(s.addrs[0]);

        UC_DEBUG(" s.index {}  s.addrs.size {} s.addrs.data {}", s.index, s.addrs.size(), static_cast<const void*>(s.addrs.data()));

        size_t compBytes = HUF_compress_float_fixRatio (compBuf, compBufSize, src, srcSize, R145, DT_BF16);

        // {   // 测试代码
        //     size_t decompBytes = HUF_decompress_float_fixRatio(decompBuf, srcSize, compBuf, compBytes, NULL);
        //     if (memcmp(src, decompBuf, 1024) == 0) {
        //         UC_DEBUG("Consistency check passed ....");
        //     } else {
        //         UC_DEBUG("Data inconsistency detected...");
        //         printf("src 前64字节:\n");
        //         print_binary_block(src, 64);
        //         printf("compressed_buffer 前96字节:\n");
        //         print_binary_block(compBuf, 96);
        //         printf("decompBuf 前64字节:\n");
        //         print_binary_block(decompBuf, 64);
        //         return;
        //     }
        //     UC_DEBUG("shard {} comp {} B decompBytes {} B", s.index, compBytes, decompBytes);
        // }

        memcpy(s.addrs[0], compBuf, compBytes);   // 拷贝回原始地址，实现等效原地压缩的效果

        totalCompBytes += compBytes;
    }

    UC_DEBUG("COMPRESS DUMP END....");

    if (compBuf) free(compBuf);
    // if (decompBuf) free(decompBuf);

    /* 交给后端（PosixStore）*/
    backend_->Dump(std::move(ct.task->desc));

    UC_DEBUG("COMPRESS DUMP END. Total compressed bytes: {}", totalCompBytes);
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
