/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_io_aggregation_stream.h"
#include <cstdint>
#include <numeric>
#include "ascend_shard_io_aggregator.h"

namespace UC::Trans {

namespace {

Status Unsupported(const char* op)
{
    return Status::InvalidParam("Cache IO aggregation stream does not support {}", op);
}

constexpr size_t kIoAggregationLaneNumber = 4;
constexpr size_t kIoAggregationPipelineDepth = 2;
constexpr uint16_t kIoAggregationMaxReadyLanes = 8;

}  // namespace

AscendIoAggregationStream::AscendIoAggregationStream() = default;

AscendIoAggregationStream::~AscendIoAggregationStream() = default;

Status AscendIoAggregationStream::Setup()
{
    aggregator_.reset();
    pendingEvents_.clear();
    return Status::OK();
}

Status AscendIoAggregationStream::EnsureAggregator(const std::vector<size_t>& sizes)
{
    if (aggregator_) { return Status::OK(); }
    if (sizes.empty()) { return Status::InvalidParam("invalid Cache IO aggregation sizes"); }

    const auto objectBytes = std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
    if (objectBytes == 0) { return Status::InvalidParam("invalid Cache IO aggregation bytes"); }
    AscendShardIOAggregatorConfig aggregatorConfig;
    aggregatorConfig.streamNumber = kIoAggregationLaneNumber;
    aggregatorConfig.pipelineDepth = kIoAggregationPipelineDepth;
    aggregatorConfig.maxReadyLanes = kIoAggregationMaxReadyLanes;
    aggregatorConfig.objectBytes = objectBytes;
    aggregatorConfig.maxFragments = sizes.size();
    auto aggregator = std::make_unique<AscendShardIOAggregator>();
    auto s = aggregator->Setup(aggregatorConfig);
    if (s.Failure()) [[unlikely]] { return s; }
    aggregator_ = std::move(aggregator);
    for (auto* event : pendingEvents_) {
        s = aggregator_->WaitEvent(event);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    pendingEvents_.clear();
    return Status::OK();
}

Status AscendIoAggregationStream::DeviceToHost(void*, void*, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHost(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHost(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*, void*, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::HostToDevice(void*, void*, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDevice(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDevice(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*, void*, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void* host, void* device[],
                                                    const std::vector<size_t>& sizes)
{
    auto s = EnsureAggregator(sizes);
    if (s.Failure()) [[unlikely]] { return s; }
    return aggregator_->SubmitLoadObject(host, device, sizes);
}

Status AscendIoAggregationStream::DeviceToHostAsync(void* device[], void* host,
                                                    const std::vector<size_t>& sizes)
{
    auto s = EnsureAggregator(sizes);
    if (s.Failure()) [[unlikely]] { return s; }
    return aggregator_->SubmitDumpObject(device, host, sizes);
}

Status AscendIoAggregationStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Unsupported("AppendCallback");
}

Status AscendIoAggregationStream::Synchronized()
{
    if (!aggregator_) {
        pendingEvents_.clear();
        return Status::OK();
    }
    return aggregator_->Synchronize();
}

Status AscendIoAggregationStream::WaitEvent(void* event)
{
    if (event == nullptr) { return Status::OK(); }
    if (!aggregator_) {
        pendingEvents_.push_back(event);
        return Status::OK();
    }
    return aggregator_->WaitEvent(event);
}

}  // namespace UC::Trans
