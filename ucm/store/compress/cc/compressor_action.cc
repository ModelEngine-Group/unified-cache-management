#include "compressor_action.h"
#include <chrono>
#include <pthread.h>
#include <queue>
#include <vector>
#include "logger/logger.h"

namespace UC::Compressor {
namespace {

struct R160ModeCounts {
    size_t highPrecision{0};
    size_t quantized{0};
};

struct R200ModeCounts {
    size_t tunstall{0};
    size_t fp8Fallback{0};
};

struct CodecModeCounts {
    R160ModeCounts r160;
    R200ModeCounts r200;

    void Add(CodecPayloadMode mode)
    {
        switch (mode) {
            case CodecPayloadMode::R160_HIGH_PRECISION: ++r160.highPrecision; break;
            case CodecPayloadMode::R160_QUANTIZED: ++r160.quantized; break;
            case CodecPayloadMode::R200_TUNSTALL: ++r200.tunstall; break;
            case CodecPayloadMode::R200_FP8_FALLBACK: ++r200.fp8Fallback; break;
            case CodecPayloadMode::INVALID: break;
            case CodecPayloadMode::NOT_APPLICABLE: break;
        }
    }
};

enum class CodecStatsStage {
    LOAD,
    DUMP,
};

void ReportR160ModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                         const R160ModeCounts& counts)
{
    const size_t valid = counts.highPrecision + counts.quantized;
    if (valid == 0) { return; }

    const double highRatio =
        100.0 * static_cast<double>(counts.highPrecision) / static_cast<double>(valid);
    const double quantizedRatio =
        100.0 * static_cast<double>(counts.quantized) / static_cast<double>(valid);
    UC_DEBUG(
        "R160 {} MODE | task_id: {}, high_precision: {}, quantized: {}, high_ratio: {:.2f}%, "
        "quantized_ratio: {:.2f}%",
        stage == CodecStatsStage::LOAD ? "LOAD" : "DUMP", taskId, counts.highPrecision,
        counts.quantized, highRatio, quantizedRatio);
}

void ReportR200ModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                         const R200ModeCounts& counts)
{
    const size_t valid = counts.tunstall + counts.fp8Fallback;
    if (valid == 0) { return; }

    const double tunstallRatio =
        100.0 * static_cast<double>(counts.tunstall) / static_cast<double>(valid);
    const double fallbackRatio =
        100.0 * static_cast<double>(counts.fp8Fallback) / static_cast<double>(valid);
    UC_DEBUG(
        "R200 {} MODE | task_id: {}, tunstall: {}, fp8_fallback: {}, tunstall_ratio: {:.2f}%, "
        "fallback_ratio: {:.2f}%",
        stage == CodecStatsStage::LOAD ? "LOAD" : "DUMP", taskId, counts.tunstall,
        counts.fp8Fallback, tunstallRatio, fallbackRatio);
}

void ReportCodecModeStats(Detail::TaskHandle taskId, CodecStatsStage stage,
                          const CodecModeCounts& counts)
{
    ReportR160ModeStats(taskId, stage, counts.r160);
    ReportR200ModeStats(taskId, stage, counts.r200);
}

}  // namespace

CompressorAction::~CompressorAction()
{
    // 后续改多线程需要在这销毁线程
}

Status CompressorAction::Setup(const Config& config, HashSet<Detail::TaskHandle>* failureSet)
{
    backend_ = config.storeBackend;
    failureSet_ = failureSet;
    shardSize_ = config.shardSize;
    compressedShardSize_ = config.compressedShardSize;
    decompressThreadNum = config.decompressThreadNum;

    codec_ = MakeCodec(static_cast<FixedRatio>(config.compressRatio),
                       static_cast<DataType>(config.dataType), compressedShardSize_);
    if (!codec_) {
        return Status::InvalidParam("Unsupported codec combo (ratio={}, dtype={})",
                                    config.compressRatio, config.dataType);
    }

    if ((shardSize_ & 1U) != 0) {
        return Status::InvalidParam("BF16 shardSize({}) must be even", shardSize_);
    }
    if (compressedShardSize_ == 0) {
        return Status::InvalidParam("compressed_shard_size must be provided by pipeline builder");
    }
    const size_t codecCompressedSize = codec_->CompressedSize(shardSize_);
    if (codecCompressedSize != compressedShardSize_) {
        return Status::InvalidParam(
            "compressed shard size({}) is invalid for shardSize({}), ratio({}) and dtype({})",
            compressedShardSize_, shardSize_, config.compressRatio, config.dataType);
    }
    if (codec_->NeedsCompress() && compressedShardSize_ % 4096 != 0) {
        return Status::InvalidParam(
            "compressed shard size({}) must be 4096-byte aligned for shardSize({}) and ratio({})",
            compressedShardSize_, shardSize_, config.compressRatio);
    }

    dump_pool_.SetNWorker(config.streamNumber >> 1)
        .SetCpuAffinity(config.cpuAffinityCores)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
        .Run();

    load_pool_.SetNWorker(decompressThreadNum)
        .SetCpuAffinity(config.cpuAffinityCores)
        .SetWorkerFn([this](auto& ct, auto&) { Compress_Load(ct); })
        .Run();

    UC_DEBUG("Compressor Setup OK | load_threads: {}, shard_size: {} B, stored_shard_size: {} B",
             decompressThreadNum, shardSize_, compressedShardSize_);

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
    auto fail = [this, &ct](const char* stage, const Status& status) {
        UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, stage: {}, status: {}", ct.task->id, stage,
                 status);
        failureSet_->Insert(ct.task->id);
    };
    if (!codec_->NeedsDecompress()) {
        auto result = backend_->Load(std::move(ct.task->desc));
        if (!result) {
            fail("backend submit", result.Error());
        } else {
            auto status = backend_->Wait(result.Value());
            if (status.Failure()) { fail("backend wait", status); }
        }
        ct.waiter->Done();
        UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
        return;
    }

    auto result = backend_->Load(ct.task->desc);
    if (!result) {
        fail("backend submit", result.Error());
        ct.waiter->Done();
        return;
    }
    if (result.Value() > 0) {
        auto status = backend_->Wait(result.Value());
        if (status.Failure()) {
            fail("backend wait", status);
            ct.waiter->Done();
            return;
        }
    }

    const auto& shards = ct.task->desc;
    UC_DEBUG("COMPRESS LOAD | shards_count: {}", shards.size());

    CodecModeCounts modeCounts;
    for (const auto& shard : shards) {
        const CodecPayloadMode payloadMode =
            codec_->GetPayloadMode(shard.addrs[0], compressedShardSize_, shardSize_);
        const int err = codec_->DecompressInplace(shard.addrs[0], shardSize_);
        if (err != 0) {
            UC_ERROR("COMPRESS LOAD FAILED | task_id: {}, shard: {}, error: {} ({})", ct.task->id,
                     shard.index, err, CodecErrorName(err));
            failureSet_->Insert(ct.task->id);
            continue;
        }
        modeCounts.Add(payloadMode);
        UC_DEBUG("COMPRESS LOAD | shard: {}, done, decompressed_size: {}", shard.index, shardSize_);
    }
    ReportCodecModeStats(ct.task->id, CodecStatsStage::LOAD, modeCounts);

    ct.waiter->Done();
    UC_DEBUG("COMPRESS LOAD END | task_id: {}", ct.task->id);
}

void CompressorAction::Compress_Dump(CompressTask& ct)
{
    UC_DEBUG("COMPRESS DUMP START | task_id: {}", ct.task->id);
    auto fail = [this, &ct](const char* stage, const Status& status) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, stage: {}, status: {}", ct.task->id, stage,
                 status);
        failureSet_->Insert(ct.task->id);
    };

    if (!codec_->NeedsCompress()) {
        const auto n = ct.task->desc.size();
        if (n > 0) {
            auto result = backend_->Dump(std::move(ct.task->desc));
            if (!result) {
                fail("backend submit", result.Error());
            } else {
                auto status = backend_->Wait(result.Value());
                if (status.Failure()) { fail("backend wait", status); }
            }
        }
        ct.waiter->Done();
        UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
        return;
    }

    const auto& desc = ct.task->desc;
    if (desc.empty()) {
        UC_ERROR("COMPRESS DUMP FAILED | task_id: {}, desc is empty", ct.task->id);
        failureSet_->Insert(ct.task->id);
        ct.waiter->Done();
        return;
    }

    const size_t scratchSize = codec_->CompressScratchSize(shardSize_);
    Detail::TaskDesc backendDesc;
    backendDesc.brief = ct.task->desc.brief;
    std::vector<void*> blockToFree;
    auto dumpMemoryPool = std::make_unique<MemoryPool>(scratchSize, desc.size());

    CodecModeCounts modeCounts;
    for (const UC::Detail::Shard& shard : desc) {
        UC_DEBUG("COMPRESS DUMP | task_id: {}, shard: {}, compressing...", ct.task->id,
                 shard.index);

        auto* compressed = static_cast<uint8_t*>(dumpMemoryPool->allocate());
        const size_t compressedBytes = codec_->Compress(compressed, shard.addrs[0], shardSize_);
        if (compressedBytes != compressedShardSize_) [[unlikely]] {
            UC_ERROR(
                "COMPRESS DUMP FAILED | task_id: {}, shard: {}, expected {} B but codec produced "
                "{} B",
                ct.task->id, shard.index, compressedShardSize_, compressedBytes);
            dumpMemoryPool->deallocate({compressed});
            continue;
        }
        modeCounts.Add(codec_->GetPayloadMode(compressed, compressedBytes, shardSize_));

        std::vector<void*> addrs{static_cast<void*>(compressed)};
        backendDesc.push_back(Detail::Shard{shard.owner, shard.index, addrs});
        blockToFree.push_back(static_cast<void*>(compressed));
        UC_DEBUG("COMPRESS DUMP | shard: {}, done, stored_size: {}", shard.index, compressedBytes);
    }
    if (backendDesc.size() != desc.size()) {
        UC_ERROR(
            "COMPRESS DUMP FAILED | task_id: {}, only {}/{} shards met the compression budget; "
            "the whole dump is aborted",
            ct.task->id, backendDesc.size(), desc.size());
        failureSet_->Insert(ct.task->id);
        if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }
        ct.waiter->Done();
        return;
    }

    bool backendDumpSucceeded = false;
    auto result = backend_->Dump(std::move(backendDesc));
    if (!result) {
        fail("backend submit", result.Error());
    } else if (result.Value() > 0) {
        auto status = backend_->Wait(result.Value());
        if (status.Failure()) {
            fail("backend wait", status);
        } else {
            backendDumpSucceeded = true;
        }
    } else {
        backendDumpSucceeded = true;
    }
    if (backendDumpSucceeded) {
        ReportCodecModeStats(ct.task->id, CodecStatsStage::DUMP, modeCounts);
    }
    if (!blockToFree.empty()) { dumpMemoryPool->deallocate(blockToFree); }

    ct.waiter->Done();
    UC_DEBUG("COMPRESS DUMP END | task_id: {}", ct.task->id);
}

}  // namespace UC::Compressor
