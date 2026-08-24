/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_STREAM_H
#define UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_STREAM_H

#include <memory>
#include "trans/device.h"
#include "trans/stream.h"

namespace UC::Trans {

class AscendSdmaDirectCopier;

class AscendSdmaDirectStream : public Stream {
public:
    AscendSdmaDirectStream();
    ~AscendSdmaDirectStream() override;
    Status Setup() override;

    Status DeviceToHost(void* device, void* host, size_t size) override;
    Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHost(void* device[], void* host, size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device, void* host, size_t size) override;
    Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) override;

    Status HostToDevice(void* host, void* device, size_t size) override;
    Status HostToDevice(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDevice(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device, size_t size) override;
    Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], const std::vector<size_t>& sizes) override;
    Status DeviceToHostAsync(void* device[], void* host, const std::vector<size_t>& sizes) override;

    Status AppendCallback(std::function<void(bool)> cb) override;
    Status Synchronized() override;
    Status WaitEvent(void* event) override;

private:
    std::unique_ptr<AscendSdmaDirectCopier> copier_{nullptr};
};

}  // namespace UC::Trans

#endif
