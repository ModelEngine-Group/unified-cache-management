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
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include "core/transport.h"

namespace transport {

class MemoryRegionManager final {
public:
    using NativeHandles = std::vector<MemoryHandle>;

    struct ProtocolRegistration {
        NativeHandles native_handles;
    };

    struct Region {
        MemoryRegion memory;
        MemoryHandle handle = kInvalidMemoryHandle;
        std::array<std::optional<ProtocolRegistration>,
                   static_cast<std::size_t>(TransportProtocol::Count)>
            registrations;
    };

    static MemoryHandle HashMemoryRegion(const MemoryRegion& memory);

    Status ValidateMemoryRegion(const MemoryRegion& memory) const;
    Status AddMemoryRegion(TransportProtocol protocol, const MemoryRegion& memory,
                           NativeHandles native_handles, MemoryHandle handle);
    Status RemoveMemoryRegion(TransportProtocol protocol, MemoryHandle handle);
    Status FindMemoryRegion(TransportProtocol protocol, MemoryHandle handle, Region& region) const;
    Status FindMemoryRegion(TransportProtocol protocol, int32_t device_id,
                            const void* start_address, Region& region) const;
    Status FindContainingMemoryRegion(TransportProtocol protocol, int32_t device_id,
                                      const void* address, std::uint64_t length,
                                      Region& region) const;
    Status GetMemoryRegions(TransportProtocol protocol, std::vector<Region>& regions) const;

private:
    using MemoryDomain = int32_t;
    using AddressIndex = std::map<std::uint64_t, Region*>;
    using MemoryDomainIndex = std::map<MemoryDomain, AddressIndex>;

    mutable std::mutex mutex_;
    std::unordered_map<MemoryHandle, std::unique_ptr<Region>> by_handle_;
    MemoryDomainIndex by_address_;
};

}  // namespace transport
