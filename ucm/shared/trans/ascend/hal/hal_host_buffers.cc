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
#include "hal_host_buffers.h"
#include <ascend_hal.h>
#include <numa.h>
#include <stdexcept>
#include <string>
#include "logger/logger.h"
#include "trans/device.h"

namespace UC::Trans {

struct HalHostBuffers::Mapping {
    drv_mem_handle_t* handle{nullptr};
    bool mapped{false};
};

HalHostBuffers::HalHostBuffers() = default;
HalHostBuffers::~HalHostBuffers() { Reset(); }

void HalHostBuffers::Reset()
{
    for (size_t i = 0; i < mappings_.size(); ++i) {
        if (!mappings_[i].mapped) { continue; }
        std::byte* addr = static_cast<std::byte*>(base_) + i * rankStride_;
        drvError_t ret = halMemUnmap(addr);
        if (ret != DRV_ERROR_NONE) {
            UC_ERROR("halMemUnmap failed: owner={} device={} rank={} addr={} ret={}", owner_,
                     deviceId_, i, static_cast<void*>(addr), static_cast<int>(ret));
        }
    }
    auto release = [&](size_t rank) {
        if (!mappings_[rank].handle) { return; }
        drvError_t ret = halMemRelease(mappings_[rank].handle);
        if (ret != DRV_ERROR_NONE) {
            UC_ERROR("halMemRelease failed: owner={} device={} rank={} ret={}", owner_, deviceId_,
                     rank, static_cast<int>(ret));
        }
    };
    // Drop imported references before the locally created allocation.
    for (size_t i = 0; i < mappings_.size(); ++i) {
        if (i != owner_) { release(i); }
    }
    if (owner_ < mappings_.size()) { release(owner_); }
    if (base_ != nullptr) {
        drvError_t ret = halMemAddressFree(base_);
        if (ret != DRV_ERROR_NONE) {
            UC_ERROR("halMemAddressFree failed: owner={} device={} addr={} ret={}", owner_,
                     deviceId_, base_, static_cast<int>(ret));
        }
    }
    mappings_.clear();
    base_ = nullptr;
    rankStride_ = 0;
}

void HalHostBuffers::Setup(int32_t deviceId, size_t dataBytes, size_t nRanks, size_t rank)
{
    if (base_ != nullptr) { throw std::logic_error("HAL host buffers are already initialized"); }
    if (deviceId < 0 || dataBytes == 0 || nRanks == 0 || rank >= nRanks) {
        throw std::invalid_argument(
            fmt::format("invalid HAL host layout: owner={} device={} data_bytes={} ranks={}", rank,
                        deviceId, dataBytes, nRanks));
    }
    owner_ = rank;
    deviceId_ = deviceId;
    try {
        if (numa_available() < 0) { throw std::runtime_error("Host NUMA is not available"); }
        const int numaCount = numa_num_configured_nodes();
        if (numaCount <= 0) {
            throw std::runtime_error(fmt::format("invalid Host NUMA node count: {}", numaCount));
        }
        const int32_t numaNode = static_cast<int32_t>(owner_ % static_cast<size_t>(numaCount));
        UC_INFO("HAL NUMA assignment: owner={} numa_count={} numa_node={}", owner_, numaCount,
                numaNode);
        Device device;
        Status status = device.Setup(deviceId_);
        if (status.Failure()) {
            throw std::runtime_error(fmt::format(
                "Device::Setup failed: owner={} device={} status={}", owner_, deviceId_, status));
        }
        try {
            LocalSetup(dataBytes, nRanks, numaNode, MEM_HUGE_PAGE_TYPE);
        } catch (const std::runtime_error& error) {
            UC_WARN(
                "Huge-page allocation failed: owner={} device={} error={}; retrying with "
                "MEM_NORMAL_PAGE_TYPE",
                owner_, deviceId_, error.what());
            Reset();
            LocalSetup(dataBytes, nRanks, numaNode, MEM_NORMAL_PAGE_TYPE);
        }
    } catch (...) {
        Reset();
        throw;
    }
}

void HalHostBuffers::LocalSetup(size_t dataBytes, size_t nRanks, int32_t numaNode, uint32_t pgType)
{
    // halMemAddressReserve requires 1GB alignment if allocation > 512MB
    constexpr size_t vaAlignment = size_t(1) << 30;
    auto failed = [&](const char* api, drvError_t ret) {
        std::string message = fmt::format("{} failed: owner={} device={} ret={}", api, owner_,
                                          deviceId_, static_cast<int>(ret));
        UC_ERROR("{}", message);
        return std::runtime_error(message);
    };

    drv_mem_prop prop{};
    prop.side = MEM_HOST_NUMA_SIDE;
    prop.devid = static_cast<uint32_t>(numaNode);
    prop.pg_type = pgType;
    prop.mem_type = MEM_DDR_TYPE;
    size_t allocGranularity = 0;
    drvError_t ret =
        halMemGetAllocationGranularity(&prop, MEM_ALLOC_GRANULARITY_RECOMMENDED, &allocGranularity);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemGetAllocationGranularity", ret); }
    if (allocGranularity == 0 || vaAlignment % allocGranularity != 0) {
        throw std::runtime_error(fmt::format("invalid HAL granularity({}) for data size({})",
                                             allocGranularity, dataBytes));
    }
    rankStride_ = (dataBytes + allocGranularity - 1) / allocGranularity * allocGranularity;
    mappings_.resize(nRanks);
    const size_t reserveBytes =
        (rankStride_ * nRanks + vaAlignment - 1) / vaAlignment * vaAlignment;
    UC_INFO(
        "HAL host allocation: owner={} device={} numa={} ranks={} data_bytes={} "
        "rank_stride={} reserve_bytes={} page_type={} alloc_granularity={}",
        owner_, deviceId_, numaNode, nRanks, dataBytes, rankStride_, reserveBytes,
        static_cast<int>(prop.pg_type), allocGranularity);

    ret = halMemAddressReserve(&base_, reserveBytes, 0, nullptr, 0);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemAddressReserve", ret); }
    if (base_ == nullptr || reinterpret_cast<uintptr_t>(base_) % vaAlignment != 0) {
        throw std::runtime_error("HAL did not return a 1 GiB aligned VA reservation");
    }
    ret = halMemCreate(&mappings_[owner_].handle, rankStride_, &prop, 0);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemCreate", ret); }
    std::byte* local = static_cast<std::byte*>(base_) + owner_ * rankStride_;
    ret = halMemMap(local, rankStride_, 0, mappings_[owner_].handle, 0);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemMap", ret); }
    mappings_[owner_].mapped = true;
}

uint64_t HalHostBuffers::ExportHandle()
{
    if (base_ == nullptr) { throw std::logic_error("HAL host buffers are not initialized"); }
    auto failed = [&](const char* api, drvError_t ret) {
        std::string message = fmt::format("{} failed: owner={} device={} ret={}", api, owner_,
                                          deviceId_, static_cast<int>(ret));
        UC_ERROR("{}", message);
        return std::runtime_error(message);
    };
    uint64_t shareHandle = 0;
    drvError_t ret = halMemExportToShareableHandle(mappings_[owner_].handle, MEM_HANDLE_TYPE_NONE,
                                                   0, &shareHandle);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemExportToShareableHandle", ret); }
    ShareHandleAttr attr{};
    attr.enableFlag = SHR_HANDLE_NO_WLIST_ENABLE;
    ret = halMemShareHandleSetAttribute(shareHandle, SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER, attr);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemShareHandleSetAttribute", ret); }

    return shareHandle;
}

void HalHostBuffers::ImportHandle(size_t rank, uint64_t peerHandle)
{
    if (base_ == nullptr) { throw std::logic_error("HAL host buffers are not initialized"); }
    if (rank >= mappings_.size() || rank == owner_) {
        throw std::invalid_argument(fmt::format("invalid HAL peer rank: {}", rank));
    }
    if (mappings_[rank].handle != nullptr) {
        throw std::logic_error(fmt::format("HAL peer rank is already imported: {}", rank));
    }
    auto failed = [&](const char* api, drvError_t ret, size_t rank) {
        std::string message = fmt::format("{} failed: owner={} device={} rank={} ret={}", api,
                                          owner_, deviceId_, rank, static_cast<int>(ret));
        UC_ERROR("{}", message);
        return std::runtime_error(message);
    };
    drvError_t ret = halMemImportFromShareableHandle(peerHandle, static_cast<uint32_t>(deviceId_),
                                                     &mappings_[rank].handle);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemImportFromShareableHandle", ret, rank); }
    std::byte* addr = static_cast<std::byte*>(base_) + rank * rankStride_;
    ret = halMemMap(addr, rankStride_, 0, mappings_[rank].handle, 0);
    if (ret != DRV_ERROR_NONE) { throw failed("halMemMap", ret, rank); }
    mappings_[rank].mapped = true;
    UC_INFO("HAL peer mapping: owner={} device={} rank={} addr={} bytes={}", owner_, deviceId_,
            rank, static_cast<void*>(addr), rankStride_);
}

size_t HalHostBuffers::RankCount() const { return mappings_.size(); }

void* HalHostBuffers::HostData(size_t rank) const
{
    if (base_ == nullptr || rank >= mappings_.size() || rank != owner_ || !mappings_[rank].mapped) {
        return nullptr;
    }
    return static_cast<std::byte*>(base_) + rank * rankStride_;
}

void* HalHostBuffers::DeviceData(size_t rank) const
{
    if (base_ == nullptr || rank >= mappings_.size() || rank == owner_ || !mappings_[rank].mapped) {
        return nullptr;
    }
    return static_cast<std::byte*>(base_) + rank * rankStride_;
}

}  // namespace UC::Trans
