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
#include <numeric>
#include "logger/logger.h"
#include "trans/device.h"

namespace UC::Cache2 {
namespace Hal = Trans::Hal;

AscendHalDataBackend::~AscendHalDataBackend() { Reset(); }

Status AscendHalDataBackend::Setup(const BackendOptions& options)
{
    const size_t nRanks = options.rankCount;
    const size_t rankBytes = options.rankBytes;
    if (nRanks == 0 || rankBytes == 0) {
        return Status::InvalidParam("invalid HAL geometry: ranks={} bytes={}", nRanks, rankBytes);
    }
    deviceId_ = options.deviceId;
    rankBytes_ = rankBytes;
    nRanks_ = nRanks;
    localPlacement_ = options.localPlacement;
    mappings_.assign(nRanks, Mapping{});
    Trans::Device device;
    auto status = device.Setup(deviceId_);
    if (status.Failure()) {
        UC_ERROR("device setup failed: owner={} device={} status={}", owner_, deviceId_, status);
        mappings_.clear();
        return status;
    }
    /* The HAL wrapper exposes no NUMA placement parameter yet -- hal_memory's
     * MemCreate takes only a size and a page type -- so a resolved plan cannot be
     * honoured at allocation time. Report it rather than pretending it landed;
     * closing this needs the target SDK's allocation parameter. */
    if (localPlacement_.node >= 0) {
        UC_WARN(
            "A5 HAL allocation cannot apply the NUMA plan yet: owner={} device={} node={} "
            "placement={}",
            owner_, deviceId_, localPlacement_.node,
            localPlacement_.placement == Numa::Placement::Affinity ? "affinity" : "rank-slice");
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
    status = ReserveRankAddress(rank);
    if (status.Failure()) { return status; }
    void* addr = mappings_[rank].addr;
    status = Hal::MemMap(addr, rankStride_, mappings_[rank].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemMap failed: owner={} device={} rank={} addr={} rank_stride={} "
            "peer_handle={} status={}",
            owner_, deviceId_, rank, addr, rankStride_, handle, status);
        return {status.Underlying(),
                fmt::format("HAL peer map failed: rank={} status={}", rank, status)};
    }
    mappings_[rank].mapped = true;
    UC_INFO("HAL peer mapping: owner={} device={} rank={} addr={} bytes={}", owner_, deviceId_,
            rank, addr, rankStride_);
    return Status::OK();
}

void AscendHalDataBackend::FinalizeSetup() { /* Driver share handles have no name to release. */ }

void* AscendHalDataBackend::HostAddrOf(size_t rank) const
{
    if (rank != owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return mappings_[rank].addr;
}

void* AscendHalDataBackend::DeviceAddrOf(size_t rank) const
{
    if (rank == owner_ || rank >= mappings_.size() || !mappings_[rank].mapped) { return nullptr; }
    return mappings_[rank].addr;
}

void AscendHalDataBackend::Reset()
{
    // reset for map
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (!mappings_[rank].mapped) { continue; }
        void* addr = mappings_[rank].addr;
        Status status = Hal::MemUnmap(addr);
        if (status.Failure()) {
            UC_ERROR("HAL unmap failed: owner={} device={} rank={} addr={} status={}", owner_,
                     deviceId_, rank, addr, status);
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
    for (size_t rank = 0; rank < mappings_.size(); ++rank) {
        if (mappings_[rank].addr == nullptr) { continue; }
        Status status = Hal::MemAddressFree(mappings_[rank].addr);
        if (status.Failure()) {
            UC_ERROR("HAL address free failed: owner={} device={} rank={} addr={} status={}",
                     owner_, deviceId_, rank, mappings_[rank].addr, status);
        }
    }
    mappings_.clear();
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
    // Reserve, create and map all use rankStride_, so it must fit both alignments.
    const size_t allocAlignment = std::lcm(vaAlignment, allocGranularity);
    rankStride_ = (rankBytes_ + allocAlignment - 1) / allocAlignment * allocAlignment;
    UC_INFO(
        "HAL host allocation: owner={} device={} ranks={} data_bytes={} "
        "rank_stride={} page_type={} alloc_granularity={}",
        owner_, deviceId_, mappings_.size(), rankBytes_, rankStride_,
        static_cast<uint32_t>(pageType), allocGranularity);

    status = ReserveRankAddress(owner_);
    if (status.Failure()) { return status; }
    status = Hal::MemCreate(&mappings_[owner_].handle, rankStride_, pageType);
    if (status.Failure()) {
        UC_ERROR(
            "halMemCreate failed: owner={} device={} page_type={} rank_stride={} "
            "alloc_granularity={} status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), rankStride_, allocGranularity,
            status);
        return status;
    }
    void* local = mappings_[owner_].addr;
    status = Hal::MemMap(local, rankStride_, mappings_[owner_].handle);
    if (status.Failure()) {
        UC_ERROR(
            "halMemMap failed: owner={} device={} page_type={} addr={} rank_stride={} "
            "status={}",
            owner_, deviceId_, static_cast<uint32_t>(pageType), local, rankStride_, status);
        return status;
    }
    mappings_[owner_].mapped = true;
    UC_INFO("HAL local mapping: owner={} device={} addr={} bytes={}", owner_, deviceId_, local,
            rankStride_);
    return Status::OK();
}

Status AscendHalDataBackend::ReserveRankAddress(size_t rank)
{
    Status status = Hal::MemAddressReserve(&mappings_[rank].addr, rankStride_);
    if (status.Failure()) {
        UC_ERROR(
            "halMemAddressReserve failed: owner={} device={} rank={} reserve_bytes={} status={}",
            owner_, deviceId_, rank, rankStride_, status);
        return status;
    }
    if (mappings_[rank].addr == nullptr) {
        UC_ERROR("Invalid HAL VA reservation: owner={} device={} rank={} reserve_bytes={}", owner_,
                 deviceId_, rank, rankStride_);
        return Status::Error("HAL did not return a valid ptr");
    }
    UC_INFO("HAL VA reservation: owner={} device={} rank={} addr={} reserve_bytes={}", owner_,
            deviceId_, rank, mappings_[rank].addr, rankStride_);
    return Status::OK();
}

}  // namespace UC::Cache2
#endif  // UCM_RUNTIME_ASCEND_HAL
