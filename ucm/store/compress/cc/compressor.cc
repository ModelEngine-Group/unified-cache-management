#include "compressor.h"
#include <shared_mutex>
#include "logger/logger.h"
#include "template/hashset.h"
#include "trans_manager.h"

namespace UC::Compressor {

class CompressorImpl {
public:
    std::shared_ptr<StoreV1> backend{nullptr};
    bool transEnable{false};
    TransManager transMgr;

public:
    Status Setup(const Config& config)
    {
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed to check config params: {}.", s);
            return s;
        }
        backend = config.storeBackend;
        transEnable = config.deviceId >= 0;
        if (transEnable) {
            s = transMgr.Setup(config);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }

private:
    Status CheckConfig(const Config& config)
    {
        if (!config.storeBackend) { return Status::InvalidParam("invalid store backend"); }
        if (config.deviceId < -1) {
            return Status::InvalidParam("invalid device({})", config.deviceId);
        }

        // TODO 参数校验
        return Status::OK();
    }
    void ShowConfig(const Config& config)
    {
        // TODO 参数打印
        constexpr const char* ns = "Compressor";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("Set {}::StoreBackend to {}.", ns, backend->Readme());
    }
};

Compressor::~Compressor() = default;

Status Compressor::Setup(const Detail::Dictionary& config)
{
    Config param;
    config.Get("store_backend", param.storeBackend);
    config.Get("unique_id", param.uniqueId);
    config.GetNumber("device_id", param.deviceId);
    config.GetNumber("tensor_size", param.tensorSize);
    config.GetNumber("shard_size", param.shardSize);
    config.GetNumber("block_size", param.blockSize);
    config.GetNumber("layer_size", param.layerSize);
    config.GetNumber("compress_ratio", param.compressRatio);
    config.GetNumber("data_type", param.dataType);
    config.GetNumber("timeout_ms", param.timeoutMs);
    config.GetNumber("stream_number", param.streamNumber);

    try {
        impl_ = std::make_shared<CompressorImpl>();
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make Compressor impl object.", e.what());
        return Status::Error(e.what());
    }
    return impl_->Setup(param);
}

std::string Compressor::Readme() const { return "Compressor"; }

Expected<std::vector<uint8_t>> Compressor::Lookup(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->Lookup(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

Expected<ssize_t> Compressor::LookupOnPrefix(const Detail::BlockId* blocks, size_t num)
{
    auto res = impl_->backend->LookupOnPrefix(blocks, num);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to lookup blocks({}).", res.Error(), num); }
    return res;
}

void Compressor::Prefetch(const Detail::BlockId*, size_t) {}

Expected<Detail::TaskHandle> Compressor::Load(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::LOAD, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit load task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<Detail::TaskHandle> Compressor::Dump(Detail::TaskDesc task)
{
    if (!impl_->transEnable) { return Status::Error("transfer is not enable"); }
    auto res = impl_->transMgr.Submit({TransTask::Type::DUMP, std::move(task)});
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}).", res.Error(), task.brief);
    }
    return res;
}

Expected<bool> Compressor::Check(Detail::TaskHandle taskId)
{
    auto res = impl_->transMgr.Check(taskId);
    if (!res) [[unlikely]] { UC_ERROR("Failed({}) to check task({}).", res.Error(), taskId); }
    return res;
}

Status Compressor::Wait(Detail::TaskHandle taskId)
{
    auto s = impl_->transMgr.Wait(taskId);
    if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to wait task({}).", s, taskId); }
    return s;
}
  
}

extern "C" UC::StoreV1* MakeCompressStore() { return new UC::Compressor::Compressor();}