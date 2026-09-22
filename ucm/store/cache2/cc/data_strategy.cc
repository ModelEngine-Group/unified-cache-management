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
#include "data_strategy.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fmt/format.h>
#include <new>
#include <thread>

#if UCM_RUNTIME_ASCEND_HAL
#include "logger/logger.h"
#include "trans/device.h"
#endif

namespace UC::Cache2 {

#if UCM_RUNTIME_ASCEND_HAL
namespace Hal = Trans::Hal;

struct DataStrategy::Mapping {
    Hal::MemHandle handle{nullptr};
    bool mapped{false};
};
#endif

DataStrategy::DataStrategy() = default;
DataStrategy::~DataStrategy()
{
#if UCM_RUNTIME_ASCEND_HAL
    Reset();
#endif
}

Status DataStrategy::Setup(CtrlLayout& ctrl, int32_t deviceId, size_t slotSize,
                           size_t nSlotsPerRank, size_t rank, size_t timeoutMs)
{
    if (nSlotsPerRank_ != 0) {
        return Status::Error("cache2 data strategy is already initialized");
    }
#if UCM_RUNTIME_ASCEND_HAL
    const size_t totalSlots = ctrl.SlotCount();
    owner_ = rank;
    deviceId_ = deviceId;
    Status status = Status::OK();
    try {
        Trans::Device device;
        status = device.Setup(deviceId_);
        if (status.Success()) {
            status = LocalSetup(slotSize * nSlotsPerRank, totalSlots / nSlotsPerRank,
                                Hal::PageType::Huge);
            if (status.Failure()) {
                UC_WARN(
                    "Huge-page allocation failed: owner={} device={} status={}; retrying "
                    "with normal pages",
                    owner_, deviceId_, status);
                Reset();
                status = LocalSetup(slotSize * nSlotsPerRank, totalSlots / nSlotsPerRank,
                                    Hal::PageType::Normal);
            }
        }
        if (status.Success()) { status = CrossRankSetup(ctrl, timeoutMs); }
    } catch (const std::bad_alloc&) {
        status = Status::OutOfMemory();
    } catch (const std::exception& error) {
        status = Status::Error(error.what());
    }
    if (status.Failure()) {
        UC_ERROR("cache2 data setup failed: owner={} device={} status={}", owner_, deviceId_,
                 status);
        Reset();
        return status;
    }
    slotSize_ = slotSize;
    nSlotsPerRank_ = nSlotsPerRank;
    return Status::OK();
#else
    return {Status::Unsupported().Underlying(),
            "cache2 HAL data strategy requires ascend-a5 runtime"};
#endif
}

#if UCM_RUNTIME_ASCEND_HAL
void DataStrategy::Reset()
{
    // reset for map
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (!mappings_[rank].mapped) { continue; }
        std::byte* addr = static_cast<std::byte*>(base_) + rank * rankStride_;
        Status status = Hal::MemUnmap(addr);
        if (status.Failure()) {
            UC_ERROR("HAL unmap failed: owner={} device={} rank={} addr={} status={}", owner_,
                     deviceId_, rank, static_cast<void*>(addr), status);
        }
    }
    auto release = [&](size_t rank) {
        if (mappings_[rank].handle == nullptr) { return; }
        Status status = Hal::MemRelease(mappings_[rank].handle);
        if (status.Failure()) {
            UC_ERROR("HAL release failed: owner={} device={} rank={} status={}", owner_, deviceId_,
                     rank, status);
        }
    };
    // for other rank, MemRelease is used to cancel the handle import, it won't release the physical
    // memory
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (rank != owner_) { release(rank); }
    }
    // for owner, MemRelease is used to release physical memory
    if (owner_ < mappings_.size()) { release(owner_); }
    if (base_ != nullptr) {
        Status status = Hal::MemAddressFree(base_);
        if (status.Failure()) {
            UC_ERROR("HAL address free failed: owner={} device={} addr={} status={}", owner_,
                     deviceId_, base_, status);
        }
    }
    mappings_.clear();
    base_ = nullptr;
    rankStride_ = 0;
}

Status DataStrategy::LocalSetup(size_t dataBytes, size_t nRanks, Hal::PageType pageType)
{
    constexpr size_t vaAlignment = Hal::kAddressAlignment;
    size_t allocGranularity = 0;
    Status status = Hal::MemGetAllocationGranularity(pageType, &allocGranularity);
    if (status.Failure()) {
        UC_ERROR(
            "halMemGetAllocationGranularity failed: owner={} device={} page_type={} "
            "data_bytes={} status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), dataBytes, status);
        return status;
    }
    if (allocGranularity == 0) {
        UC_ERROR(
            "Invalid HAL allocation granularity: owner={} device={} page_type={} "
            "alloc_granularity={} data_bytes={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), allocGranularity, dataBytes);
        return Status::Error(fmt::format("invalid HAL granularity({}) for data size({})",
                                         allocGranularity, dataBytes));
    }
    rankStride_ = (dataBytes + allocGranularity - 1) / allocGranularity * allocGranularity;
    mappings_.resize(nRanks);
    const size_t reserveBytes =
        (rankStride_ * nRanks + vaAlignment - 1) / vaAlignment * vaAlignment;
    UC_INFO(
        "HAL host allocation: owner={} device={} ranks={} data_bytes={} "
        "rank_stride={} reserve_bytes={} page_type={} alloc_granularity={}",
        owner_, deviceId_, nRanks, dataBytes, rankStride_, reserveBytes,
        static_cast<uint32_t>(pageType), allocGranularity);

    status = Hal::MemAddressReserve(&base_, reserveBytes);
    if (status.Failure()) {
        UC_ERROR(
            "halMemAddressReserve failed: owner={} device={} page_type={} ranks={} "
            "rank_stride={} reserve_bytes={} status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), nRanks, rankStride_, reserveBytes,
            status);
        return status;
    }
    if (base_ == nullptr) {
        UC_ERROR(
            "Invalid HAL VA reservation: owner={} device={} page_type={} addr={} "
            "va_alignment={} reserve_bytes={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), base_, vaAlignment, reserveBytes);
        return Status::Error("HAL did not return a valid ptr");
    }
    status = Hal::MemCreate(&mappings_[owner_].handle, rankStride_, pageType);
    if (status.Failure()) {
        UC_ERROR(
            "halMemCreate failed: owner={} device={} page_type={} rank_stride={} "
            "alloc_granularity={} status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), rankStride_, allocGranularity,
            status);
        return status;
    }
    std::byte* local = static_cast<std::byte*>(base_) + owner_ * rankStride_;
    status = Hal::MemMap(local, rankStride_, mappings_[owner_].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemMap failed: owner={} device={} page_type={} addr={} rank_stride={} "
            "status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), static_cast<void*>(local),
            rankStride_, status);
        return status;
    }
    mappings_[owner_].mapped = true;
    return Status::OK();
}

Status DataStrategy::CrossRankSetup(CtrlLayout& ctrl, size_t timeoutMs)
{
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    uint64_t shareHandle = 0;
    Status status = Hal::MemExportToShareableHandle(mappings_[owner_].handle, &shareHandle);
    if (status.Failure()) {
        UC_ERROR("halMemExportToShareableHandle failed: owner={} device={} status={}", owner_,
                 deviceId_, status);
        return status;
    }
    status = Hal::MemShareHandleDisableWhitelist(shareHandle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemShareHandleSetAttribute failed: owner={} device={} share_handle={} "
            "status={}",
            owner_, deviceId_, shareHandle, status);
        return status;
    }

    CtrlLayout::RankDataDesc desc;
    desc.handle.store(shareHandle, std::memory_order_relaxed);
    status = ctrl.SetRankDesc(owner_, desc);
    if (status.Failure()) {
        UC_ERROR("SetRankDesc failed: owner={} device={} share_handle={} status={}", owner_,
                 deviceId_, shareHandle, status);
        return {status.Underlying(), fmt::format("SetRankDesc failed: owner={} device={} status={}",
                                                 owner_, deviceId_, status)};
    }
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (rank == owner_) { continue; }
        Expected<CtrlLayout::RankDataDesc> peerDesc = ctrl.GetRankDesc(rank);
        while (!peerDesc) {
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (now >= deadline) { break; }
            std::this_thread::sleep_until(std::min(deadline, now + std::chrono::milliseconds(10)));
            if (std::chrono::steady_clock::now() >= deadline) { break; }
            peerDesc = ctrl.GetRankDesc(rank);
        }
        if (!peerDesc) {
            UC_ERROR("GetRankDesc timed out: owner={} device={} rank={} timeout_ms={} status={}",
                     owner_, deviceId_, rank, timeoutMs, peerDesc.Error());
            return {Status::Timeout().Underlying(),
                    fmt::format("GetRankDesc timed out: owner={} device={} rank={} timeout_ms={} "
                                "status={}",
                                owner_, deviceId_, rank, timeoutMs, peerDesc.Error())};
        }
        const uint64_t peerHandle = peerDesc.Value().handle.load(std::memory_order_relaxed);
        status = Hal::MemImportFromShareableHandle(peerHandle, static_cast<uint32_t>(deviceId_),
                                                   &mappings_[rank].handle);
        if (status.Failure()) {
            UC_ERROR(
                "halMemImportFromShareableHandle failed: owner={} device={} rank={} "
                "peer_handle={} status={}",
                owner_, deviceId_, rank, peerHandle, status);
            return {status.Underlying(),
                    fmt::format("HAL peer import failed: rank={} status={}", rank, status)};
        }
        std::byte* addr = static_cast<std::byte*>(base_) + rank * rankStride_;
        status = Hal::MemMap(addr, rankStride_, mappings_[rank].handle);
        if (status.Failure()) {
            UC_ERROR(
                "halMemMap failed: owner={} device={} rank={} addr={} rank_stride={} "
                "peer_handle={} status={}",
                owner_, deviceId_, rank, static_cast<void*>(addr), rankStride_, peerHandle, status);
            return {status.Underlying(),
                    fmt::format("HAL peer map failed: rank={} status={}", rank, status)};
        }
        mappings_[rank].mapped = true;
        UC_INFO("HAL peer mapping: owner={} device={} rank={} addr={} bytes={}", owner_, deviceId_,
                rank, static_cast<void*>(addr), rankStride_);
    }
    return Status::OK();
}
#endif

bool DataStrategy::HostAccessibleOf(size_t slotIdx) const
{
#if UCM_RUNTIME_ASCEND_HAL
    return nSlotsPerRank_ != 0 && slotIdx / nSlotsPerRank_ == owner_;
#else
    return false;
#endif
}

void* DataStrategy::DataAt(size_t slotIdx) const
{
#if UCM_RUNTIME_ASCEND_HAL
    if (nSlotsPerRank_ == 0) { return nullptr; }
    const size_t rank = slotIdx / nSlotsPerRank_;
    if (rank != owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return static_cast<std::byte*>(base_) + rank * rankStride_ +
           (slotIdx % nSlotsPerRank_) * slotSize_;
#else
    return nullptr;
#endif
}

void* DataStrategy::DeviceDataAt(size_t slotIdx) const
{
#if UCM_RUNTIME_ASCEND_HAL
    if (nSlotsPerRank_ == 0) { return nullptr; }
    const size_t rank = slotIdx / nSlotsPerRank_;
    if (rank == owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return static_cast<std::byte*>(base_) + rank * rankStride_ +
           (slotIdx % nSlotsPerRank_) * slotSize_;
#else
    return nullptr;
#endif
}

}  // namespace UC::Cache2
