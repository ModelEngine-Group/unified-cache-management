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
constexpr uint16_t kSdmaDirectMaxReadyLanes = 8;
}  // namespace

AscendSdmaDirectCopier::~AscendSdmaDirectCopier() { Cleanup(); }

Status AscendSdmaDirectCopier::Setup()
{
    Cleanup();

    auto s = AclStatus(aclrtCreateStream(&fftsStream_), "aclrtCreateStream(sdma-direct)");
    if (s.Failure()) {
        Cleanup();
        return s;
    }

    setup_ = true;
    return Status::OK();
}

Status AscendSdmaDirectCopier::WaitEvent(void* event)
{
    if (event == nullptr) { return Status::OK(); }
    if (!setup_) { return Status::OK(); }
    return AclStatus(aclrtStreamWaitEvent(fftsStream_, static_cast<aclrtEvent>(event)),
                     "aclrtStreamWaitEvent(sdma-direct)");
}

Status AscendSdmaDirectCopier::SubmitLoadObject(const void* hostDevicePtr, void** devices,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildHostToDeviceSpecs(hostDevicePtr, devices, sizes, specs);
    if (s.Failure()) { return s; }
    return LaunchSpecs(std::move(specs));
}

Status AscendSdmaDirectCopier::SubmitDumpObject(void** devices, void* hostDevicePtr,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildDeviceToHostSpecs(devices, hostDevicePtr, sizes, specs);
    if (s.Failure()) { return s; }
    return LaunchSpecs(std::move(specs));
}

Status AscendSdmaDirectCopier::Synchronize()
{
    if (!setup_) { return Status::OK(); }
    auto s = AclStatus(aclrtSynchronizeStream(fftsStream_), "aclrtSynchronizeStream(sdma-direct)");
    if (s.Failure()) { return s; }
    inFlight_.clear();
    return Status::OK();
}

void AscendSdmaDirectCopier::Cleanup() noexcept
{
    if (fftsStream_ != nullptr) {
        (void)aclrtDestroyStream(fftsStream_);
        fftsStream_ = nullptr;
    }
    inFlight_.clear();
    setup_ = false;
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
        if (sizes[i] != 0 && devices[i] != nullptr) {
            auto* src = static_cast<const std::byte*>(hostDevicePtr) + offset;
            specs.push_back({devices[i], src, sizes[i]});
        }
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
        if (sizes[i] != 0 && devices[i] != nullptr) {
            auto* dst = static_cast<std::byte*>(hostDevicePtr) + offset;
            specs.push_back({dst, devices[i], sizes[i]});
        }
        offset += sizes[i];
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::LaunchSpecs(std::vector<AscendFftsCopySpec>&& specs)
{
    if (specs.empty()) { return Status::OK(); }
    auto object = std::make_unique<InFlightObject>();
    object->specs = std::move(specs);
    uint16_t readyCount = 0;
    auto s = object->dispatcher.BuildCopies(object->specs, kSdmaDirectMaxReadyLanes, readyCount);
    if (s.Failure()) { return s; }
    s = object->dispatcher.Launch(fftsStream_, readyCount);
    if (s.Failure()) { return s; }
    inFlight_.push_back(std::move(object));
    return Status::OK();
}

Status AscendSdmaDirectCopier::AclStatus(aclError ret, const char* expr)
{
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
    return Status{static_cast<int32_t>(ret), expr};
}

}  // namespace UC::Trans
