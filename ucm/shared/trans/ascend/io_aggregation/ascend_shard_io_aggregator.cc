/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_shard_io_aggregator.h"
#include <numeric>
#include "logger/logger.h"

namespace UC::Trans {

namespace {
Status AclStatus(aclError ret, const char* expr)
{
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
    return Status{static_cast<int32_t>(ret), expr};
}
}  // namespace

AscendShardIOAggregator::~AscendShardIOAggregator() { Cleanup(); }

Status AscendShardIOAggregator::Setup(const AscendShardIOAggregatorConfig& config)
{
    Cleanup();
    if (config.streamNumber == 0) { return Status::InvalidParam("invalid stream number"); }
    if (config.pipelineDepth == 0) { return Status::InvalidParam("invalid pipeline depth"); }
    if (config.maxReadyLanes == 0) { return Status::InvalidParam("invalid max ready lanes"); }
    if (config.objectBytes == 0) { return Status::InvalidParam("invalid object bytes"); }
    if (config.maxFragments == 0) { return Status::InvalidParam("invalid max fragments"); }

    streamNumber_ = config.streamNumber;
    pipelineDepth_ = config.pipelineDepth;
    maxReadyLanes_ = config.maxReadyLanes;
    objectBytes_ = config.objectBytes;
    maxFragments_ = config.maxFragments;
    nextObjectIndex_ = 0;

    auto s = Status::OK();
    lanes_.resize(streamNumber_);
    for (auto& lane : lanes_) {
        s = AclStatus(aclrtCreateStream(&lane.copyStream), "aclrtCreateStream(copy)");
        if (s.Failure()) {
            Cleanup();
            return s;
        }
        s = AclStatus(aclrtCreateStream(&lane.fftsStream), "aclrtCreateStream(ffts)");
        if (s.Failure()) {
            Cleanup();
            return s;
        }

        lane.nextSlotIndex = 0;
        lane.stagingBuffers.assign(pipelineDepth_, nullptr);
        lane.slotReady.assign(pipelineDepth_, nullptr);
        lane.slotFree.assign(pipelineDepth_, nullptr);
        for (size_t slot = 0; slot < pipelineDepth_; ++slot) {
            s = AclStatus(
                aclrtMalloc(&lane.stagingBuffers[slot], objectBytes_, ACL_MEM_TYPE_HIGH_BAND_WIDTH),
                "aclrtMalloc(staging)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtCreateEvent(&lane.slotReady[slot]), "aclrtCreateEvent(slotReady)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtCreateEvent(&lane.slotFree[slot]), "aclrtCreateEvent(slotFree)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.copyStream),
                          "aclrtRecordEvent(slotFree)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
        }
    }

    setup_ = true;
    return Status::OK();
}

Status AscendShardIOAggregator::WaitEvent(void* event)
{
    if (!setup_) { return Status::OK(); }
    if (event == nullptr) { return Status::OK(); }
    auto aclEvent = static_cast<aclrtEvent>(event);
    for (auto& lane : lanes_) {
        auto s = AclStatus(aclrtStreamWaitEvent(lane.copyStream, aclEvent),
                           "aclrtStreamWaitEvent(copy prerequisite)");
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtStreamWaitEvent(lane.fftsStream, aclEvent),
                      "aclrtStreamWaitEvent(ffts prerequisite)");
        if (s.Failure()) { return s; }
    }
    return Status::OK();
}

Status AscendShardIOAggregator::BuildScatterSpecs(InFlightObject& object, void* staging,
                                                  void** devices,
                                                  const std::vector<size_t>& sizes) const
{
    if (staging == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid shard IO aggregation pointers");
    }
    if (sizes.empty() || sizes.size() > maxFragments_) {
        return Status::InvalidParam("invalid shard IO aggregation tensor number({})", sizes.size());
    }

    object.specs.clear();
    object.specs.reserve(sizes.size());
    auto* stagingBase = static_cast<uint8_t*>(staging);
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto size = sizes[i];
        if (offset > objectBytes_ || size > objectBytes_ - offset) {
            return Status::InvalidParam("shard IO aggregation object bytes overflow");
        }
        if (size != 0 && devices[i] != nullptr) {
            object.specs.push_back({devices[i], stagingBase + offset, size});
        }
        offset += size;
    }
    if (offset == 0 || offset > objectBytes_) {
        return Status::InvalidParam("invalid shard IO aggregation object bytes({})", offset);
    }
    return Status::OK();
}

Status AscendShardIOAggregator::BuildGatherSpecs(InFlightObject& object, void* staging,
                                                 void** devices,
                                                 const std::vector<size_t>& sizes) const
{
    if (staging == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid shard IO aggregation pointers");
    }
    if (sizes.empty() || sizes.size() > maxFragments_) {
        return Status::InvalidParam("invalid shard IO aggregation tensor number({})", sizes.size());
    }

    object.specs.clear();
    object.specs.reserve(sizes.size());
    auto* stagingBase = static_cast<uint8_t*>(staging);
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto size = sizes[i];
        if (offset > objectBytes_ || size > objectBytes_ - offset) {
            return Status::InvalidParam("shard IO aggregation object bytes overflow");
        }
        if (size != 0 && devices[i] != nullptr) {
            object.specs.push_back({stagingBase + offset, devices[i], size});
        }
        offset += size;
    }
    if (offset == 0 || offset > objectBytes_) {
        return Status::InvalidParam("invalid shard IO aggregation object bytes({})", offset);
    }
    return Status::OK();
}

Status AscendShardIOAggregator::LaunchFfts(InFlightObject& object, aclrtStream stream) const
{
    uint16_t readyCount = 0;
    auto s = object.dispatcher.BuildCopies(object.specs, maxReadyLanes_, readyCount);
    if (s.Failure()) { return s; }
    return object.dispatcher.Launch(stream, readyCount);
}

Status AscendShardIOAggregator::SubmitLoadObject(void* host, void** devices,
                                                 const std::vector<size_t>& sizes)
{
    if (!setup_) { return Status::Error("shard IO aggregator is not setup"); }
    if (host == nullptr) { return Status::InvalidParam("invalid load host pointer"); }

    auto& lane = lanes_[nextObjectIndex_ % streamNumber_];
    const auto slot = lane.nextSlotIndex % pipelineDepth_;
    auto* staging = lane.stagingBuffers[slot];
    auto object = std::make_unique<InFlightObject>();
    auto s = BuildScatterSpecs(*object, staging, devices, sizes);
    if (s.Failure()) { return s; }

    const auto objectBytes = std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
    s = AclStatus(aclrtStreamWaitEvent(lane.copyStream, lane.slotFree[slot]),
                  "aclrtStreamWaitEvent(slotFree)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtMemcpyAsync(staging, objectBytes_, host, objectBytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, lane.copyStream),
                  "aclrtMemcpyAsync(load staging)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtRecordEvent(lane.slotReady[slot], lane.copyStream),
                  "aclrtRecordEvent(slotReady)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtStreamWaitEvent(lane.fftsStream, lane.slotReady[slot]),
                  "aclrtStreamWaitEvent(slotReady)");
    if (s.Failure()) { return s; }

    s = LaunchFfts(*object, lane.fftsStream);
    if (s.Failure()) {
        auto release = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.fftsStream),
                                 "aclrtRecordEvent(slotFree after failed load launch)");
        return release.Failure() ? release : s;
    }
    s = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.fftsStream),
                  "aclrtRecordEvent(slotFree)");
    if (s.Failure()) { return s; }

    lane.inFlight.push_back(std::move(object));
    ++lane.nextSlotIndex;
    ++nextObjectIndex_;
    return Status::OK();
}

Status AscendShardIOAggregator::SubmitDumpObject(void** devices, void* host,
                                                 const std::vector<size_t>& sizes)
{
    if (!setup_) { return Status::Error("shard IO aggregator is not setup"); }
    if (host == nullptr) { return Status::InvalidParam("invalid dump host pointer"); }

    auto& lane = lanes_[nextObjectIndex_ % streamNumber_];
    const auto slot = lane.nextSlotIndex % pipelineDepth_;
    auto* staging = lane.stagingBuffers[slot];
    auto object = std::make_unique<InFlightObject>();
    auto s = BuildGatherSpecs(*object, staging, devices, sizes);
    if (s.Failure()) { return s; }

    const auto objectBytes = std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
    s = AclStatus(aclrtStreamWaitEvent(lane.fftsStream, lane.slotFree[slot]),
                  "aclrtStreamWaitEvent(slotFree)");
    if (s.Failure()) { return s; }
    s = LaunchFfts(*object, lane.fftsStream);
    if (s.Failure()) {
        auto release = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.fftsStream),
                                 "aclrtRecordEvent(slotFree after failed dump launch)");
        return release.Failure() ? release : s;
    }
    s = AclStatus(aclrtRecordEvent(lane.slotReady[slot], lane.fftsStream),
                  "aclrtRecordEvent(slotReady)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtStreamWaitEvent(lane.copyStream, lane.slotReady[slot]),
                  "aclrtStreamWaitEvent(slotReady)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtMemcpyAsync(host, objectBytes_, staging, objectBytes,
                                   ACL_MEMCPY_DEVICE_TO_HOST, lane.copyStream),
                  "aclrtMemcpyAsync(dump staging)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.copyStream),
                  "aclrtRecordEvent(slotFree)");
    if (s.Failure()) { return s; }

    lane.inFlight.push_back(std::move(object));
    ++lane.nextSlotIndex;
    ++nextObjectIndex_;
    return Status::OK();
}

Status AscendShardIOAggregator::Synchronize()
{
    if (!setup_) { return Status::OK(); }
    for (auto& lane : lanes_) {
        for (auto event : lane.slotFree) {
            auto s = AclStatus(aclrtStreamWaitEvent(lane.copyStream, event),
                               "aclrtStreamWaitEvent(slotFree)");
            if (s.Failure()) { return s; }
        }
        auto s = AclStatus(aclrtSynchronizeStream(lane.copyStream), "aclrtSynchronizeStream(copy)");
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtSynchronizeStream(lane.fftsStream), "aclrtSynchronizeStream(ffts)");
        if (s.Failure()) { return s; }
        lane.inFlight.clear();
    }
    return Status::OK();
}

void AscendShardIOAggregator::Cleanup() noexcept
{
    for (auto& lane : lanes_) {
        for (auto event : lane.slotReady) {
            if (event != nullptr) { (void)aclrtDestroyEvent(event); }
        }
        for (auto event : lane.slotFree) {
            if (event != nullptr) { (void)aclrtDestroyEvent(event); }
        }
        for (auto* buffer : lane.stagingBuffers) {
            if (buffer != nullptr) { (void)aclrtFree(buffer); }
        }
        if (lane.copyStream != nullptr) { (void)aclrtDestroyStream(lane.copyStream); }
        if (lane.fftsStream != nullptr) { (void)aclrtDestroyStream(lane.fftsStream); }
    }

    setup_ = false;
    streamNumber_ = 0;
    pipelineDepth_ = 0;
    maxReadyLanes_ = 0;
    objectBytes_ = 0;
    maxFragments_ = 0;
    nextObjectIndex_ = 0;
    lanes_.clear();
}

}  // namespace UC::Trans
