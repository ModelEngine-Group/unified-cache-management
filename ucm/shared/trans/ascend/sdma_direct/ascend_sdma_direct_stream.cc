/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_sdma_direct_stream.h"
#include "ascend_sdma_direct_copier.h"

namespace UC::Trans {

namespace {

Status Unsupported(const char* op)
{
    return Status::InvalidParam("Cache SDMA Direct stream does not support {}", op);
}

}  // namespace

AscendSdmaDirectStream::AscendSdmaDirectStream() = default;

AscendSdmaDirectStream::~AscendSdmaDirectStream() = default;

Status AscendSdmaDirectStream::Setup()
{
    auto copier = std::make_unique<AscendSdmaDirectCopier>();
    auto s = copier->Setup();
    if (s.Failure()) [[unlikely]] { return s; }
    copier_ = std::move(copier);
    return Status::OK();
}

Status AscendSdmaDirectStream::DeviceToHost(void*, void*, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHost(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHost(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*, void*, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::HostToDevice(void*, void*, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDevice(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDevice(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*, void*, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void* host, void* device[],
                                                 const std::vector<size_t>& sizes)
{
    if (!copier_) [[unlikely]] { return Status::Error("Cache SDMA Direct stream is not setup"); }
    if (host == nullptr) [[unlikely]] {
        return Status::InvalidParam("Cache SDMA Direct requires host device pointer");
    }
    return copier_->SubmitLoadObject(host, device, sizes);
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void* device[], void* host,
                                                 const std::vector<size_t>& sizes)
{
    if (!copier_) [[unlikely]] { return Status::Error("Cache SDMA Direct stream is not setup"); }
    if (host == nullptr) [[unlikely]] {
        return Status::InvalidParam("Cache SDMA Direct requires host device pointer");
    }
    return copier_->SubmitDumpObject(device, host, sizes);
}

Status AscendSdmaDirectStream::DeviceToDevice(void*, void*, size_t)
{
    return Unsupported("DeviceToDevice");
}

Status AscendSdmaDirectStream::DeviceToDevice(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToDevice");
}

Status AscendSdmaDirectStream::DeviceToDevice(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToDevice");
}

Status AscendSdmaDirectStream::DeviceToDeviceAsync(void*, void*, size_t)
{
    return Unsupported("DeviceToDeviceAsync");
}

Status AscendSdmaDirectStream::DeviceToDeviceAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToDeviceAsync");
}

Status AscendSdmaDirectStream::DeviceToDeviceAsync(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToDeviceAsync");
}

Status AscendSdmaDirectStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Unsupported("AppendCallback");
}

Status AscendSdmaDirectStream::Synchronized()
{
    if (!copier_) { return Status::OK(); }
    return copier_->Synchronize();
}

Status AscendSdmaDirectStream::WaitEvent(const Event& event)
{
    if (!copier_) { return Status::OK(); }
    return copier_->WaitEvent(event);
}

}  // namespace UC::Trans
