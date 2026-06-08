/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache_io_executor.h"
#include <cstdint>
#include <limits>
#include <numeric>
#include "copy_stream.h"
#include "logger/logger.h"

#ifndef UCM_ENABLE_ASCEND_IO_AGGREGATION
#define UCM_ENABLE_ASCEND_IO_AGGREGATION 0
#endif

#if UCM_ENABLE_ASCEND_IO_AGGREGATION
#include "trans/ascend/ascend_shard_io_aggregator.h"
#endif

namespace UC::CacheStore {

class TensorIOExecutor : public CacheIOExecutor {
    CopyStream stream_;
    std::vector<size_t> tensorSizes_{};

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        return stream_.Setup(config.deviceId, config.streamNumber, config.useGdr);
    }
    Status WaitEvent(void* event) override { return stream_.WaitEvent(event); }
    Status HostToDevice(void* host, void** devices) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto pDevice = devices[i];
            auto size = tensorSizes_[i];
            auto s = stream_.NextStream()->HostToDeviceAsync(pHost, pDevice, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D({}) tensor({}/{}).", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }
    Status DeviceToHost(void** devices, void* host) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pDevice = devices[i];
            auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto size = tensorSizes_[i];
            auto s = stream_.NextStream()->DeviceToHostAsync(pDevice, pHost, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H({}) tensor({}/{}).", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }
    Status Synchronize() override { return stream_.Synchronize(); }
};

#if UCM_ENABLE_ASCEND_IO_AGGREGATION
class AggregatedIOExecutor : public CacheIOExecutor {
    Trans::AscendShardIOAggregator aggregator_;
    std::vector<size_t> tensorSizes_{};

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        const auto objectBytes =
            std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), static_cast<size_t>(0));
        if (config.ioAggregationMaxReadyLanes > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam("too many FFTS ready lanes({})",
                                        config.ioAggregationMaxReadyLanes);
        }
        Trans::AscendShardIOAggregatorConfig aggregatorConfig;
        aggregatorConfig.deviceId = config.deviceId;
        aggregatorConfig.streamNumber = config.streamNumber;
        aggregatorConfig.pipelineDepth = config.ioAggregationPipelineDepth;
        aggregatorConfig.maxReadyLanes = static_cast<uint16_t>(config.ioAggregationMaxReadyLanes);
        aggregatorConfig.objectBytes = objectBytes;
        aggregatorConfig.maxFragments = tensorSizes_.size();
        UC_INFO("Set Cache IO aggregation streams={}, objectBytes={}, tensors={}.",
                config.streamNumber, objectBytes, tensorSizes_.size());
        auto s = aggregator_.Setup(aggregatorConfig);
        if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to setup Cache IO aggregator.", s); }
        return s;
    }
    Status WaitEvent(void* event) override { return aggregator_.WaitEvent(event); }
    Status HostToDevice(void* host, void** devices) override
    {
        return aggregator_.SubmitLoadObject(host, devices, tensorSizes_);
    }
    Status DeviceToHost(void** devices, void* host) override
    {
        return aggregator_.SubmitDumpObject(devices, host, tensorSizes_);
    }
    Status Synchronize() override { return aggregator_.Synchronize(); }
};
#else
class UnavailableAggregatedIOExecutor : public CacheIOExecutor {
public:
    Status Setup(const Config&) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status WaitEvent(void*) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status HostToDevice(void*, void**) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status DeviceToHost(void**, void*) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status Synchronize() override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
};
#endif

std::unique_ptr<CacheIOExecutor> MakeCacheIOExecutor(const Config& config)
{
    if (!config.cacheIOAggregation) { return std::make_unique<TensorIOExecutor>(); }
#if UCM_ENABLE_ASCEND_IO_AGGREGATION
    return std::make_unique<AggregatedIOExecutor>();
#else
    return std::make_unique<UnavailableAggregatedIOExecutor>();
#endif
}

}  // namespace UC::CacheStore
