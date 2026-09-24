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
#include "data_backend_posix.h"
#include <fmt/format.h>
#include <limits>
#include <string>
#include <unistd.h>
#include <utility>
#include "logger/logger.h"
#include "posix_shm.h"

namespace UC::Cache2 {
namespace {

/* Deterministic across processes, unlike std::hash. */
uint64_t HashName(const std::string& name)
{
    uint64_t hash = 14695981039346656037ULL;
    for (char c : name) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ULL;
    }
    /* kInvalid is the unpublished marker in CtrlLayout::RankDataDesc. */
    if (hash == std::numeric_limits<uint64_t>::max()) { --hash; }
    return hash;
}

}  // namespace

PosixShmDataBackend::PosixShmDataBackend(const std::string& uniqueId)
    : baseName_("/ucm_cache2_" + uniqueId + "_data")
{
}

PosixShmDataBackend::~PosixShmDataBackend() { Reset(); }

Status PosixShmDataBackend::Setup(int32_t deviceId, size_t nRanks, size_t rankBytes)
{
    if (nRanks == 0 || rankBytes == 0) {
        return Status::InvalidParam("invalid posix-shm geometry: ranks={} bytes={}", nRanks,
                                    rankBytes);
    }
    const long pageSize = ::sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) { return Status::Error("sysconf(_SC_PAGESIZE) failed"); }
    rankStride_ = (rankBytes + static_cast<size_t>(pageSize) - 1) / pageSize * pageSize;
    segments_.assign(nRanks, Segment{});
    deviceId_ = deviceId;
    UC_INFO("posix-shm setup: device={} ranks={} rank_bytes={} rank_stride={}", deviceId_, nRanks,
            rankBytes, rankStride_);
    return Status::OK();
}

Status PosixShmDataBackend::BindLocal(size_t rank)
{
    if (rank >= segments_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    void* addr = nullptr;
    auto status = MapSegment(rank, true, addr);
    if (status.Failure()) { return status; }
    segments_[rank] = Segment{addr, true};
    ownerRank_ = rank;
    UC_INFO("posix-shm local segment: owner={} device={} name={} bytes={}", rank, deviceId_,
            SegmentName(rank), rankStride_);
    return Status::OK();
}

Status PosixShmDataBackend::ExportLocal(uint64_t* handle)
{
    if (ownerRank_ >= segments_.size() || !segments_[ownerRank_].owner) {
        return Status::Error("posix-shm has no local segment bound");
    }
    *handle = HashName(SegmentName(ownerRank_));
    return Status::OK();
}

Status PosixShmDataBackend::ImportPeer(size_t rank, uint64_t handle)
{
    if (rank >= segments_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    if (handle != HashName(SegmentName(rank))) {
        return Status::InvalidParam("peer handle({}) does not match segment name of rank({})",
                                    handle, rank);
    }
    void* addr = nullptr;
    auto status = MapSegment(rank, false, addr);
    if (status.Failure()) { return status; }
    segments_[rank] = Segment{addr, false};
    UC_INFO("posix-shm peer segment: owner={} device={} rank={} name={} bytes={}", ownerRank_,
            deviceId_, rank, SegmentName(rank), rankStride_);
    return Status::OK();
}

void* PosixShmDataBackend::HostAddrOf(size_t rank) const
{
    return rank < segments_.size() ? segments_[rank].addr : nullptr;
}

void* PosixShmDataBackend::DeviceAddrOf(size_t rank) const { return nullptr; }

void PosixShmDataBackend::FinalizeSetup()
{
    if (ownerRank_ >= segments_.size() || !segments_[ownerRank_].owner) { return; }
    /* Every participant has mapped the local segment, so the name can be
     * released; the object lives on through the mappings. */
    PosixShm{SegmentName(ownerRank_)}.ShmUnlink();
    nameUnlinked_ = true;
    UC_INFO("posix-shm segment name released: owner={} device={} name={}", ownerRank_, deviceId_,
            SegmentName(ownerRank_));
}

void PosixShmDataBackend::Reset()
{
    for (size_t rank = 0; rank < segments_.size(); ++rank) {
        auto& segment = segments_[rank];
        if (segment.addr != nullptr) { PosixShm::MUnmap(segment.addr, rankStride_); }
        if (segment.owner && !nameUnlinked_) { PosixShm{SegmentName(rank)}.ShmUnlink(); }
        segment = Segment{};
    }
    segments_.clear();
    rankStride_ = 0;
    nameUnlinked_ = false;
}

std::string PosixShmDataBackend::SegmentName(size_t rank) const
{
    return baseName_ + "_" + std::to_string(rank);
}

Status PosixShmDataBackend::MapSegment(size_t rank, bool create, void*& addr)
{
    const auto name = SegmentName(rank);
    const auto flags = PosixShm::OpenFlag::READ_WRITE |
                       (create ? PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL : 0);
    PosixShm shm{name};
    auto status = shm.ShmOpen(flags);
    if (status == Status::DuplicateKey()) {
        /* A leftover segment of a crashed run (same uniqueId): unlink and
         * retry; mappings held by other processes stay valid. */
        UC_WARN("posix-shm segment exists, recreating: name={}", name);
        shm.ShmUnlink();
        status = shm.ShmOpen(PosixShm::OpenFlag::READ_WRITE | PosixShm::OpenFlag::CREATE |
                             PosixShm::OpenFlag::EXCL);
    }
    if (status.Failure()) {
        UC_ERROR("posix-shm open failed: owner={} device={} name={} create={} status={}",
                 ownerRank_, deviceId_, name, create, status);
        return {status.Underlying(),
                fmt::format("shm open failed: name={} status={}", name, status)};
    }
    if (create) {
        status = shm.Truncate(rankStride_);
        if (status.Failure()) {
            UC_ERROR("posix-shm truncate failed: owner={} device={} name={} bytes={} status={}",
                     ownerRank_, deviceId_, name, rankStride_, status);
            return {status.Underlying(),
                    fmt::format("shm truncate failed: name={} bytes={} status={}", name,
                                rankStride_, status)};
        }
    }
    status = shm.MMap(addr, rankStride_, /*write=*/true, /*read=*/true, /*shared=*/true);
    if (status.Failure()) {
        UC_ERROR("posix-shm mmap failed: owner={} device={} name={} bytes={} status={}", ownerRank_,
                 deviceId_, name, rankStride_, status);
        return {status.Underlying(), fmt::format("shm mmap failed: name={} bytes={} status={}",
                                                 name, rankStride_, status)};
    }
    return Status::OK();
}

}  // namespace UC::Cache2
