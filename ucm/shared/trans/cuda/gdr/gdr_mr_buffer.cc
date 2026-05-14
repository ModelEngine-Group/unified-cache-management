/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "gdr_mr_buffer.h"
#include <map>
#include <mutex>

#if UCM_ENABLE_GDR_STREAM
#include "gdr_copy.h"
#endif

namespace {

std::mutex gHostRegistryMutex;
std::map<uint64_t, size_t> gHostBuffers;

std::mutex gDeviceRegistryMutex;
std::map<uint64_t, UC::Trans::DeviceBufferInfo> gDeviceBuffers;

bool InRange(uint64_t base, size_t totalSize, uint64_t addr, size_t size)
{
    if (addr < base) { return false; }
    const auto offset = addr - base;
    if (offset > totalSize) { return false; }
    return size <= totalSize - offset;
}

}  // namespace

namespace UC::Trans {

void HostBufferRegistry::Register(void* host, size_t size)
{
    if (!host || size == 0) { return; }
    std::lock_guard<std::mutex> lock{gHostRegistryMutex};
    gHostBuffers[reinterpret_cast<uint64_t>(host)] = size;
}

bool HostBufferRegistry::Lookup(const void* host, HostBufferInfo* info)
{
    if (!host || !info) { return false; }
    const auto addr = reinterpret_cast<uint64_t>(host);
    std::lock_guard<std::mutex> lock{gHostRegistryMutex};
    const auto it = gHostBuffers.find(addr);
    if (it == gHostBuffers.end()) { return false; }
    *info = {it->first, it->second};
    return true;
}

void HostBufferRegistry::Unregister(void* host)
{
    if (!host) { return; }
    std::lock_guard<std::mutex> lock{gHostRegistryMutex};
    gHostBuffers.erase(reinterpret_cast<uint64_t>(host));
}

bool HostBufferRegistry::Resolve(const void* host, size_t size, uint64_t* base, size_t* totalSize)
{
    if (!host || !base || !totalSize) { return false; }
    const auto addr = reinterpret_cast<uint64_t>(host);
    std::lock_guard<std::mutex> lock{gHostRegistryMutex};
    auto it = gHostBuffers.upper_bound(addr);
    if (it == gHostBuffers.begin()) { return false; }
    --it;
    if (!InRange(it->first, it->second, addr, size)) { return false; }
    *base = it->first;
    *totalSize = it->second;
    return true;
}

std::vector<HostBufferInfo> HostBufferRegistry::Snapshot()
{
    std::lock_guard<std::mutex> lock{gHostRegistryMutex};
    std::vector<HostBufferInfo> buffers;
    buffers.reserve(gHostBuffers.size());
    for (const auto& [addr, size] : gHostBuffers) { buffers.push_back({addr, size}); }
    return buffers;
}

void DeviceBufferRegistry::Register(void* device, size_t size)
{
    if (!device || size == 0) { return; }
    std::lock_guard<std::mutex> lock{gDeviceRegistryMutex};
    const auto addr = reinterpret_cast<uint64_t>(device);
    gDeviceBuffers[addr] = {addr, size};
}

bool DeviceBufferRegistry::Lookup(const void* device, DeviceBufferInfo* info)
{
    if (!device || !info) { return false; }
    const auto addr = reinterpret_cast<uint64_t>(device);
    std::lock_guard<std::mutex> lock{gDeviceRegistryMutex};
    const auto it = gDeviceBuffers.find(addr);
    if (it == gDeviceBuffers.end()) { return false; }
    *info = it->second;
    return true;
}

void DeviceBufferRegistry::Unregister(void* device)
{
    if (!device) { return; }
    std::lock_guard<std::mutex> lock{gDeviceRegistryMutex};
    gDeviceBuffers.erase(reinterpret_cast<uint64_t>(device));
}

bool DeviceBufferRegistry::Resolve(const void* device, size_t size, uint64_t* base,
                                   size_t* totalSize)
{
    if (!device || !base || !totalSize) { return false; }
    const auto addr = reinterpret_cast<uint64_t>(device);
    std::lock_guard<std::mutex> lock{gDeviceRegistryMutex};
    auto it = gDeviceBuffers.upper_bound(addr);
    if (it == gDeviceBuffers.begin()) { return false; }
    --it;
    const auto& info = it->second;
    if (!InRange(info.addr, info.size, addr, size)) { return false; }
    *base = info.addr;
    *totalSize = info.size;
    return true;
}

std::vector<DeviceBufferInfo> DeviceBufferRegistry::Snapshot()
{
    std::lock_guard<std::mutex> lock{gDeviceRegistryMutex};
    std::vector<DeviceBufferInfo> buffers;
    buffers.reserve(gDeviceBuffers.size());
    for (const auto& [addr, info] : gDeviceBuffers) {
        (void)addr;
        buffers.push_back(info);
    }
    return buffers;
}

void GdrMrBuffer::GdrRegisterHostBuffer(void* host, size_t size)
{
#if UCM_ENABLE_GDR_STREAM
    GdrCopyLib::RegisterHostBuffer(host, size);
#else
    (void)host;
    (void)size;
#endif
}

void GdrMrBuffer::GdrUnregisterHostBuffer(void* host)
{
#if UCM_ENABLE_GDR_STREAM
    GdrCopyLib::UnregisterHostBuffer(host);
#else
    (void)host;
#endif
}

Status GdrMrBuffer::GdrRegisterDeviceBuffer(void* device, size_t size)
{
#if UCM_ENABLE_GDR_STREAM
    return GdrCopyLib::RegisterDeviceBuffer(device, size);
#else
    (void)device;
    (void)size;
    return Status::OK();
#endif
}

void GdrMrBuffer::GdrUnregisterDeviceBuffer(void* device)
{
#if UCM_ENABLE_GDR_STREAM
    GdrCopyLib::UnregisterDeviceBuffer(device);
#else
    (void)device;
#endif
}

}  // namespace UC::Trans
