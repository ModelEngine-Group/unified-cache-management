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
#ifndef UNIFIEDCACHE_TRANS_CUDA_GDR_BUFFER_REGISTRY_H
#define UNIFIEDCACHE_TRANS_CUDA_GDR_BUFFER_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "status/status.h"

namespace UC::Trans {

struct HostBufferInfo {
    uint64_t addr;
    size_t size;
};

class HostBufferRegistry {
public:
    static void Register(void* host, size_t size);
    static bool Lookup(const void* host, HostBufferInfo* info);
    static void Unregister(void* host);
    static bool Resolve(const void* host, size_t size, uint64_t* base, size_t* totalSize);
    static std::vector<HostBufferInfo> Snapshot();
};

struct DeviceBufferInfo {
    uint64_t addr;
    size_t size;
};

class DeviceBufferRegistry {
public:
    static void Register(void* device, size_t size);
    static bool Lookup(const void* device, DeviceBufferInfo* info);
    static void Unregister(void* device);
    static bool Resolve(const void* device, size_t size, uint64_t* base, size_t* totalSize);
    static std::vector<DeviceBufferInfo> Snapshot();
};

class GdrMrBuffer {
public:
    static void GdrRegisterHostBuffer(void* host, size_t size);
    static void GdrUnregisterHostBuffer(void* host);
    static Status GdrRegisterDeviceBuffer(void* device, size_t size);
    static void GdrUnregisterDeviceBuffer(void* device);
};

class ScopedDeviceBufferRegistration {
public:
    ScopedDeviceBufferRegistration() = default;
    ScopedDeviceBufferRegistration(const ScopedDeviceBufferRegistration&) = delete;
    ScopedDeviceBufferRegistration& operator=(const ScopedDeviceBufferRegistration&) = delete;
    ScopedDeviceBufferRegistration(ScopedDeviceBufferRegistration&&) = default;
    ScopedDeviceBufferRegistration& operator=(ScopedDeviceBufferRegistration&&) = default;
    ~ScopedDeviceBufferRegistration();

    static Status Validate(const std::vector<uintptr_t>& addrs, const std::vector<size_t>& sizes);
    Status Register(const std::vector<uintptr_t>& addrs, const std::vector<size_t>& sizes);

private:
    std::vector<DeviceBufferInfo> buffers_;
};

}  // namespace UC::Trans

#endif
