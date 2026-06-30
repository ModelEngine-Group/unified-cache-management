/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_sdma_direct_copier.h"
#include <cstddef>
#include <utility>
#include "logger/logger.h"

namespace UC::Trans {

namespace {
constexpr size_t kSdmaDirectLaneNumber = 1;
constexpr uint16_t kSdmaDirectMaxReadyLanes = 8;
}  // namespace

AscendSdmaDirectCopier::~AscendSdmaDirectCopier() { Cleanup(); }

Status AscendSdmaDirectCopier::Setup()
{
    Cleanup();

    auto s = Status::OK();
    lanes_.resize(kSdmaDirectLaneNumber);
    for (auto& lane : lanes_) {
        s = AclStatus(aclrtCreateStream(&lane.fftsStream), "aclrtCreateStream(sdma-direct)");
        if (s.Failure()) {
            Cleanup();
            return s;
        }
    }

    setup_ = true;
    return Status::OK();
}

Status AscendSdmaDirectCopier::WaitEvent(void* event)
{
    if (event == nullptr) { return Status::OK(); }
    if (!setup_) { return Status::OK(); }
    for (auto& lane : lanes_) {
        auto s = AclStatus(aclrtStreamWaitEvent(lane.fftsStream, static_cast<aclrtEvent>(event)),
                           "aclrtStreamWaitEvent(sdma-direct)");
        if (s.Failure()) { return s; }
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::SubmitLoadObject(const void* hostDevicePtr, void** devices,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildHostToDeviceSpecs(hostDevicePtr, devices, sizes, specs);
    if (s.Failure()) { return s; }
    return LaunchSpecs(std::move(specs), NextLane());
}

Status AscendSdmaDirectCopier::SubmitLoadTask(const std::vector<void*>& hostDevicePtrs,
                                              const std::vector<void**>& devices,
                                              const std::vector<size_t>& sizes)
{
    if (hostDevicePtrs.size() != devices.size()) {
        return Status::InvalidParam("invalid Cache SDMA Direct H2D task inputs");
    }
    std::vector<AscendFftsCopySpec> specs;
    specs.reserve(hostDevicePtrs.size() * sizes.size());
    for (size_t i = 0; i < hostDevicePtrs.size(); ++i) {
        auto s = BuildHostToDeviceSpecs(hostDevicePtrs[i], devices[i], sizes, specs);
        if (s.Failure()) { return s; }
    }
    return LaunchSpecs(std::move(specs), NextLane());
}

Status AscendSdmaDirectCopier::SubmitDumpObject(void** devices, void* hostDevicePtr,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildDeviceToHostSpecs(devices, hostDevicePtr, sizes, specs);
    if (s.Failure()) { return s; }
    return LaunchSpecs(std::move(specs), NextLane());
}

Status AscendSdmaDirectCopier::SubmitDumpTask(const std::vector<void**>& devices,
                                              const std::vector<void*>& hostDevicePtrs,
                                              const std::vector<size_t>& sizes)
{
    if (hostDevicePtrs.size() != devices.size()) {
        return Status::InvalidParam("invalid Cache SDMA Direct D2H task inputs");
    }
    std::vector<AscendFftsCopySpec> specs;
    specs.reserve(hostDevicePtrs.size() * sizes.size());
    for (size_t i = 0; i < hostDevicePtrs.size(); ++i) {
        auto s = BuildDeviceToHostSpecs(devices[i], hostDevicePtrs[i], sizes, specs);
        if (s.Failure()) { return s; }
    }
    return LaunchSpecs(std::move(specs), NextLane());
}

Status AscendSdmaDirectCopier::Synchronize()
{
    if (!setup_) { return Status::OK(); }
    for (auto& lane : lanes_) {
        auto s = AclStatus(aclrtSynchronizeStream(lane.fftsStream),
                           "aclrtSynchronizeStream(sdma-direct)");
        if (s.Failure()) { return s; }
        lane.inFlight.clear();
    }
    return Status::OK();
}

void AscendSdmaDirectCopier::Cleanup() noexcept
{
    for (auto& lane : lanes_) {
        if (lane.fftsStream != nullptr) {
            (void)aclrtDestroyStream(lane.fftsStream);
            lane.fftsStream = nullptr;
        }
        lane.inFlight.clear();
    }
    lanes_.clear();
    setup_ = false;
    nextLaneIndex_ = 0;
}

Status AscendSdmaDirectCopier::BuildHostToDeviceSpecs(const void* hostDevicePtr, void** devices,
                                                      const std::vector<size_t>& sizes,
                                                      std::vector<AscendFftsCopySpec>& specs) const
{
    if (!setup_) { return Status::Error("Cache SDMA Direct copier is not setup"); }
    if (hostDevicePtr == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid Cache SDMA Direct H2D pointers");
    }

    specs.reserve(specs.size() + sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0 || devices[i] == nullptr) {
            return Status::InvalidParam("invalid Cache SDMA Direct H2D tensor({})", i);
        }
        auto* src = static_cast<const std::byte*>(hostDevicePtr) + offset;
        specs.push_back({devices[i], src, sizes[i]});
        offset += sizes[i];
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::BuildDeviceToHostSpecs(void** devices, void* hostDevicePtr,
                                                      const std::vector<size_t>& sizes,
                                                      std::vector<AscendFftsCopySpec>& specs) const
{
    if (!setup_) { return Status::Error("Cache SDMA Direct copier is not setup"); }
    if (hostDevicePtr == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid Cache SDMA Direct D2H pointers");
    }

    specs.reserve(specs.size() + sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0 || devices[i] == nullptr) {
            return Status::InvalidParam("invalid Cache SDMA Direct D2H tensor({})", i);
        }
        auto* dst = static_cast<std::byte*>(hostDevicePtr) + offset;
        specs.push_back({dst, devices[i], sizes[i]});
        offset += sizes[i];
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::LaunchSpecs(std::vector<AscendFftsCopySpec>&& specs, Lane& lane)
{
    if (specs.empty()) { return Status::OK(); }
    auto object = std::make_unique<InFlightObject>();
    object->specs = std::move(specs);
    uint16_t readyCount = 0;
    auto s = object->dispatcher.BuildCopies(object->specs, kSdmaDirectMaxReadyLanes, readyCount);
    if (s.Failure()) { return s; }
    s = object->dispatcher.Launch(lane.fftsStream, readyCount);
    if (s.Failure()) { return s; }
    lane.inFlight.push_back(std::move(object));
    return Status::OK();
}

AscendSdmaDirectCopier::Lane& AscendSdmaDirectCopier::NextLane()
{
    auto& lane = lanes_[nextLaneIndex_ % lanes_.size()];
    ++nextLaneIndex_;
    return lane;
}

Status AscendSdmaDirectCopier::AclStatus(aclError ret, const char* expr)
{
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
    return Status{static_cast<int32_t>(ret), expr};
}

}  // namespace UC::Trans
