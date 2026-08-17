/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_TRANS_STREAM_H
#define UNIFIEDCACHE_TRANS_STREAM_H

#include <cstdint>
#include <functional>
#include <vector>
#include "status/status.h"

namespace UC::Trans {

class Stream {
public:
    virtual ~Stream() = default;
    virtual Status Setup() = 0;

    virtual Status DeviceToHost(void* device, void* host, size_t size) = 0;
    virtual Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) = 0;
    virtual Status DeviceToHost(void* device[], void* host, size_t size, size_t number) = 0;
    virtual Status DeviceToHostAsync(void* device, void* host, size_t size) = 0;
    virtual Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) = 0;
    virtual Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) = 0;

    virtual Status HostToDevice(void* host, void* device, size_t size) = 0;
    virtual Status HostToDevice(void* host[], void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDevice(void* host, void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device, size_t size) = 0;
    virtual Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device[], const std::vector<size_t>& sizes)
    {
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            // skip zero-padded ghost slots (addr==nullptr or size==0) but
            // still advance offset so the host layout matches tensorSizes_.
            if (sizes[i] != 0 && device[i] != nullptr) {
                auto s = HostToDeviceAsync(pHost, device[i], sizes[i]);
                if (s.Failure()) [[unlikely]] { return s; }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }
    virtual Status DeviceToHostAsync(void* device[], void* host, const std::vector<size_t>& sizes)
    {
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            // skip zero-padded ghost slots (addr==nullptr or size==0) but
            // still advance offset so the host layout matches tensorSizes_.
            if (sizes[i] != 0 && device[i] != nullptr) {
                auto s = DeviceToHostAsync(device[i], pHost, sizes[i]);
                if (s.Failure()) [[unlikely]] { return s; }
            }
            offset += sizes[i];
        }
        return Status::OK();
    }
    virtual Status AppendCallback(std::function<void(bool)> cb) = 0;
    virtual Status Synchronized() = 0;
    virtual Status WaitEvent(void* event) = 0;
};

}  // namespace UC::Trans

#endif
