/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_IO_AGGREGATION_STREAM_H
#define UNIFIEDCACHE_TRANS_ASCEND_IO_AGGREGATION_STREAM_H

#include <memory>
#include <vector>
#include "trans/stream.h"

namespace UC::Trans {

class AscendShardIOAggregator;

class AscendIoAggregationStream : public Stream {
public:
    AscendIoAggregationStream();
    ~AscendIoAggregationStream() override;
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

    Status DeviceToDevice(void* source, void* destination, size_t size) override;
    Status DeviceToDevice(void* source[], void* destination[], size_t size, size_t number) override;
    Status DeviceToDevice(void* source[], void* destination, size_t size, size_t number) override;
    Status DeviceToDeviceAsync(void* source, void* destination, size_t size) override;
    Status DeviceToDeviceAsync(void* source[], void* destination[], size_t size,
                               size_t number) override;
    Status DeviceToDeviceAsync(void* source[], void* destination, size_t size,
                               size_t number) override;

    Status HostToDeviceAsync(void* host, void* device[], const std::vector<size_t>& sizes) override;
    Status DeviceToHostAsync(void* device[], void* host, const std::vector<size_t>& sizes) override;

    Status AppendCallback(std::function<void(bool)> cb) override;
    Status Synchronized() override;
    Status WaitEvent(const Event& event) override;

private:
    Status EnsureAggregator(const std::vector<size_t>& sizes);

    std::unique_ptr<AscendShardIOAggregator> aggregator_{nullptr};
    std::vector<Event> pendingEvents_{};
};

}  // namespace UC::Trans

#endif
