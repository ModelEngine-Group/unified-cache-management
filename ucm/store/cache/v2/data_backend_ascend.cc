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
#include "data_backend_ascend.h"

#if UCM_RUNTIME_ASCEND_HAL
#include <fmt/format.h>
#include "logger/logger.h"
#include "trans/device.h"

namespace UC::Cache2 {
namespace Hal = Trans::Hal;

AscendHalDataBackend::~AscendHalDataBackend() { Reset(); }

Status AscendHalDataBackend::Setup(int32_t deviceId, size_t nRanks, size_t rankBytes)
{
    if (nRanks == 0 || rankBytes == 0) {
        return Status::InvalidParam("invalid HAL geometry: ranks={} bytes={}", nRanks, rankBytes);
    }
    deviceId_ = deviceId;
    rankBytes_ = rankBytes;
    nRanks_ = nRanks;
    mappings_.assign(nRanks, Mapping{});
    Trans::Device device;
    auto status = device.Setup(deviceId_);
    if (status.Failure()) {
        UC_ERROR("device setup failed: owner={} device={} status={}", owner_, deviceId_, status);
        mappings_.clear();
        return status;
    }
    return Status::OK();
}

Status AscendHalDataBackend::BindLocal(size_t rank)
{
    if (rank >= mappings_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    owner_ = rank;
    auto status = Allocate(Hal::PageType::Huge);
    if (status.Failure()) {
        UC_WARN(
            "Huge-page allocation failed: owner={} device={} status={}; retrying "
            "with normal pages",
            owner_, deviceId_, status);
        Reset();
        mappings_.assign(nRanks_, Mapping{});
        status = Allocate(Hal::PageType::Normal);
    }
    return status;
}

Status AscendHalDataBackend::ExportLocal(uint64_t* handle)
{
    if (owner_ >= mappings_.size() || mappings_[owner_].handle == nullptr) {
        return Status::Error("HAL backend has no local segment bound");
    }
    uint64_t shareHandle = 0;
    auto status = Hal::MemExportToShareableHandle(mappings_[owner_].handle, &shareHandle);
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
    *handle = shareHandle;
    return Status::OK();
}

Status AscendHalDataBackend::ImportPeer(size_t rank, uint64_t handle)
{
    if (rank >= mappings_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    auto status = Hal::MemImportFromShareableHandle(handle, static_cast<uint32_t>(deviceId_),
                                                    &mappings_[rank].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemImportFromShareableHandle failed: owner={} device={} rank={} "
            "peer_handle={} status={}",
            owner_, deviceId_, rank, handle, status);
        return {status.Underlying(),
                fmt::format("HAL peer import failed: rank={} status={}", rank, status)};
    }
    status = Hal::MemMap(RankAddr(rank), rankStride_, mappings_[rank].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemMap failed: owner={} device={} rank={} addr={} rank_stride={} "
            "peer_handle={} status={}",
            owner_, deviceId_, rank, static_cast<void*>(RankAddr(rank)), rankStride_, handle,
            status);
        return {status.Underlying(),
                fmt::format("HAL peer map failed: rank={} status={}", rank, status)};
    }
    mappings_[rank].mapped = true;
    UC_INFO("HAL peer mapping: owner={} device={} rank={} addr={} bytes={}", owner_, deviceId_,
            rank, static_cast<void*>(RankAddr(rank)), rankStride_);
    return Status::OK();
}

void AscendHalDataBackend::FinalizeSetup() { /* Driver share handles have no name to release. */ }

void* AscendHalDataBackend::HostAddrOf(size_t rank) const
{
    if (rank != owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return RankAddr(rank);
}

void* AscendHalDataBackend::DeviceAddrOf(size_t rank) const
{
    if (rank == owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return RankAddr(rank);
}

void AscendHalDataBackend::Reset()
{
    // reset for map
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (!mappings_[rank].mapped) { continue; }
        std::byte* addr = RankAddr(rank);
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

Status AscendHalDataBackend::Allocate(Hal::PageType pageType)
{
    constexpr size_t vaAlignment = Hal::kAddressAlignment;
    size_t allocGranularity = 0;
    Status status = Hal::MemGetAllocationGranularity(pageType, &allocGranularity);
    if (status.Failure()) {
        UC_ERROR(
            "halMemGetAllocationGranularity failed: owner={} device={} page_type={} "
            "data_bytes={} status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), rankBytes_, status);
        return status;
    }
    if (allocGranularity == 0) {
        UC_ERROR(
            "Invalid HAL allocation granularity: owner={} device={} page_type={} "
            "alloc_granularity={} data_bytes={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), allocGranularity, rankBytes_);
        return Status::Error(fmt::format("invalid HAL granularity({}) for data size({})",
                                         allocGranularity, rankBytes_));
    }
    rankStride_ = (rankBytes_ + allocGranularity - 1) / allocGranularity * allocGranularity;
    const size_t nRanks = mappings_.size();
    const size_t reserveBytes =
        (rankStride_ * nRanks + vaAlignment - 1) / vaAlignment * vaAlignment;
    UC_INFO(
        "HAL host allocation: owner={} device={} ranks={} data_bytes={} "
        "rank_stride={} reserve_bytes={} page_type={} alloc_granularity={}",
        owner_, deviceId_, nRanks, rankBytes_, rankStride_, reserveBytes,
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
    status = Hal::MemMap(RankAddr(owner_), rankStride_, mappings_[owner_].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemMap failed: owner={} device={} page_type={} addr={} rank_stride={} "
            "status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType),
            static_cast<void*>(RankAddr(owner_)), rankStride_, status);
        return status;
    }
    mappings_[owner_].mapped = true;
    return Status::OK();
}

std::byte* AscendHalDataBackend::RankAddr(size_t rank) const
{
    return static_cast<std::byte*>(base_) + rank * rankStride_;
}

}  // namespace UC::Cache2
#endif  // UCM_RUNTIME_ASCEND_HAL
