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

#include "core/memory_region_manager.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <mutex>
#include <utility>
#include "common/binary_codec.h"

namespace transport {

MemoryHandle MemoryRegionManager::HashMemoryRegion(const MemoryRegion& memory)
{
    constexpr MemoryHandle kOffset = 14695981039346656037ULL;
    constexpr MemoryHandle kPrime = 1099511628211ULL;
    MemoryHandle hash = kOffset;
    const auto mix = [&hash](std::uint64_t value) {
        for (std::size_t i = 0; i < sizeof(value); ++i) {
            hash ^= value & 0xffU;
            hash *= kPrime;
            value >>= 8U;
        }
    };
    mix(detail::PtrToU64(memory.addr));
    mix(memory.length);
    mix(static_cast<std::uint32_t>(memory.device_id));
    return hash == kInvalidMemoryHandle ? 1 : hash;
}

Status MemoryRegionManager::ValidateMemoryRegion(const MemoryRegion& memory) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto begin = detail::PtrToU64(memory.addr);
    if (memory.addr == nullptr || memory.length == 0) {
        return Status::InvalidParam("memory address is null or length is zero");
    }
    if (memory.length > std::numeric_limits<std::uint64_t>::max() - begin) {
        return Status::InvalidParam("memory address range overflows");
    }
    const auto end = begin + memory.length;
    const auto address_index = by_address_.find(memory.device_id);
    if (address_index == by_address_.end()) { return Status::OK(); }

    const auto& regions = address_index->second;
    const auto next = regions.lower_bound(begin);
    if (next != regions.end() && next->first == begin) {
        const auto& region = *next->second;

        // Accept the same start address only when the complete region matches.
        if (region.memory.length != memory.length) {
            return Status::InvalidParam("memory region overlaps an existing region");
        }
        return Status::OK();
    }

    if (next != regions.end() && next->first < end) {
        return Status::InvalidParam("memory region overlaps an existing region");
    }
    if (next != regions.begin()) {
        const auto previous = std::prev(next);
        const auto previous_begin = previous->first;
        const auto previous_length = previous->second->memory.length;
        if (previous_begin + previous_length > begin) {
            return Status::InvalidParam("memory region overlaps an existing region");
        }
    }
    return Status::OK();
}

Status MemoryRegionManager::AddMemoryRegion(TransportProtocol protocol, const MemoryRegion& memory,
                                            NativeHandles native_handles, MemoryHandle handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto begin = detail::PtrToU64(memory.addr);
    const auto protocol_index = static_cast<std::size_t>(protocol);
    auto& regions = by_address_[memory.device_id];
    auto next = regions.lower_bound(begin);
    if (next != regions.end() && next->first == begin) {
        auto& region = *next->second;
        region.registrations[protocol_index] = ProtocolRegistration{std::move(native_handles)};
        return Status::OK();
    }

    auto region = std::make_unique<Region>();
    region->memory = memory;
    region->handle = handle;
    region->registrations[protocol_index] = ProtocolRegistration{std::move(native_handles)};
    auto* indexed = region.get();
    by_handle_.emplace(handle, std::move(region));
    regions.emplace_hint(next, begin, indexed);
    return Status::OK();
}

Status MemoryRegionManager::RemoveMemoryRegion(TransportProtocol protocol, MemoryHandle handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto region = by_handle_.find(handle);
    if (region == by_handle_.end()) { return Status::InvalidParam(); }
    auto& registrations = region->second->registrations;
    const auto protocol_index = static_cast<std::size_t>(protocol);
    registrations[protocol_index].reset();
    if (std::any_of(registrations.begin(), registrations.end(),
                    [](const auto& registration) { return registration.has_value(); })) {
        return Status::OK();
    }
    const MemoryDomain domain = region->second->memory.device_id;
    const auto begin = detail::PtrToU64(region->second->memory.addr);
    const auto address_index = by_address_.find(domain);
    if (address_index != by_address_.end()) {
        address_index->second.erase(begin);
        if (address_index->second.empty()) { by_address_.erase(address_index); }
    }
    by_handle_.erase(region);
    return Status::OK();
}

Status MemoryRegionManager::FindMemoryRegion(TransportProtocol protocol, MemoryHandle handle,
                                             Region& result) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto region = by_handle_.find(handle);
    if (region == by_handle_.end() ||
        !region->second->registrations[static_cast<std::size_t>(protocol)].has_value()) {
        return Status::NotFound();
    }
    result = *region->second;
    return Status::OK();
}

Status MemoryRegionManager::FindMemoryRegion(TransportProtocol protocol, int32_t device_id,
                                             const void* start_address, Region& result) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto address_index = by_address_.find(device_id);
    if (address_index == by_address_.end()) { return Status::NotFound(); }
    const auto region = address_index->second.find(detail::PtrToU64(start_address));
    if (region == address_index->second.end() ||
        !region->second->registrations[static_cast<std::size_t>(protocol)].has_value()) {
        return Status::NotFound();
    }
    result = *region->second;
    return Status::OK();
}

Status MemoryRegionManager::FindContainingMemoryRegion(TransportProtocol protocol,
                                                       int32_t device_id, const void* address,
                                                       std::uint64_t length, Region& result) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto address_index = by_address_.find(device_id);
    if (address_index == by_address_.end()) { return Status::NotFound(); }
    const auto value = detail::PtrToU64(address);
    const auto& regions = address_index->second;
    auto next = regions.upper_bound(value);
    while (next != regions.begin()) {
        const auto* region = (--next)->second;
        if (!region->registrations[static_cast<std::size_t>(protocol)].has_value()) { continue; }
        const auto begin = detail::PtrToU64(region->memory.addr);
        const auto offset = value - begin;
        if (offset <= region->memory.length && length <= region->memory.length - offset) {
            result = *region;
            return Status::OK();
        }
    }
    return Status::NotFound();
}

Status MemoryRegionManager::GetMemoryRegions(TransportProtocol protocol,
                                             std::vector<Region>& result) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    result.clear();
    result.reserve(by_handle_.size());
    for (const auto& region : by_handle_) {
        if (region.second->registrations[static_cast<std::size_t>(protocol)].has_value()) {
            result.push_back(*region.second);
        }
    }
    return Status::OK();
}

}  // namespace transport
