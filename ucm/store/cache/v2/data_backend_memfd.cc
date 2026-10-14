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
#include "data_backend_memfd.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <linux/magic.h>
#include <linux/memfd.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
#include <utility>
#include "logger/logger.h"
#include "numa/numa_policy.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::Cache2 {

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kConnectBackoff = std::chrono::milliseconds(10);

/* Upper bound on a single receive wait, so a huge setup budget cannot overflow
 * the millisecond count handed to the socket. */
constexpr int64_t kMaxReceiveMs = 3600 * 1000;
constexpr size_t kHugeTlb2MBytes = 2ULL << 20;

#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_SHIFT
#define MFD_HUGE_SHIFT 26
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << MFD_HUGE_SHIFT)
#endif

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

MemfdDataBackend::MemfdDataBackend(const std::string& domainId) : domainId_(domainId) {}

MemfdDataBackend::~MemfdDataBackend() { Reset(); }

Status MemfdDataBackend::Setup(const BackendOptions& options)
{
    if (options.rankCount == 0 || options.rankBytes == 0) {
        return Status::InvalidParam("invalid memfd geometry: ranks={} bytes={}", options.rankCount,
                                    options.rankBytes);
    }
    if (options.memfdPageMode == MemfdPageMode::Invalid) {
        return Status::InvalidParam("invalid memfd page mode");
    }
    pageMode_ = options.memfdPageMode;
    if (pageMode_ == MemfdPageMode::HugeTlb2M) {
        backingPageSize_ = kHugeTlb2MBytes;
    } else {
        const long pageSize = ::sysconf(_SC_PAGESIZE);
        if (pageSize <= 0) { return Status::Error("sysconf(_SC_PAGESIZE) failed"); }
        backingPageSize_ = static_cast<size_t>(pageSize);
    }
    if (options.rankBytes > std::numeric_limits<size_t>::max() - (backingPageSize_ - 1)) {
        return Status::InvalidParam("memfd rank bytes overflow while aligning: bytes={} page={}",
                                    options.rankBytes, backingPageSize_);
    }
    rankStride_ = (options.rankBytes + backingPageSize_ - 1) / backingPageSize_ * backingPageSize_;
    segments_ = std::vector<Segment>(options.rankCount);
    deviceId_ = options.deviceId;
    requireHostDeviceAddress_ = options.requireHostDeviceAddress;
    localPlacement_ = options.localPlacement;
    deadline_ = options.deadline;
    UC_INFO(
        "memfd setup: device={} ranks={} rank_bytes={} rank_stride={} host_device={} "
        "page_mode={} page_bytes={} pages_per_rank={}",
        deviceId_, options.rankCount, options.rankBytes, rankStride_, requireHostDeviceAddress_,
        MemfdPageModeName(pageMode_), backingPageSize_, rankStride_ / backingPageSize_);
    return Status::OK();
}

Status MemfdDataBackend::BindLocal(size_t rank)
{
    if (rank >= segments_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    auto status = CreateLocalSegment(rank);
    if (status.Failure()) { return status; }
    status = PlaceAndRegister(rank);
    if (status.Failure()) { return status; }
    ownerRank_ = rank;
    status = StartFdServer();
    if (status.Failure()) { return status; }
    UC_INFO("memfd local segment: owner={} device={} name={} bytes={} host_device={}", rank,
            deviceId_, SegmentName(rank), rankStride_, requireHostDeviceAddress_);
    return Status::OK();
}

Status MemfdDataBackend::ExportLocal(uint64_t* handle)
{
    if (ownerRank_ >= segments_.size() || !segments_[ownerRank_].owner) {
        return Status::Error("memfd backend has no local segment bound");
    }
    /* A process-local fd cannot be published; peers fetch the real descriptor
     * over the socket instead, and this token only identifies the segment. */
    *handle = HashName(SocketName(ownerRank_));
    return Status::OK();
}

Status MemfdDataBackend::ImportPeer(size_t rank, uint64_t handle)
{
    if (rank >= segments_.size()) { return Status::InvalidParam("rank({}) out of range", rank); }
    const auto name = SocketName(rank);
    if (handle != HashName(name)) {
        return Status::InvalidParam("peer handle({}) does not match socket name of rank({})",
                                    handle, rank);
    }
    FdSocket peer;
    for (;;) {
        auto status = peer.Connect(name);
        if (status.Success()) { break; }
        if (Clock::now() >= deadline_) {
            UC_ERROR("memfd fd connect timed out: owner={} device={} rank={} name={}", ownerRank_,
                     deviceId_, rank, name);
            return Status::Timeout();
        }
        std::this_thread::sleep_for(kConnectBackoff);
    }
    int32_t fd = -1;
    /* Wait for readability under whatever is left of the setup budget, then take
     * the descriptor: RecvFd itself blocks, and a peer that connected but never
     * sent would otherwise hang Setup with no way out. A spent budget still gets
     * one immediate check, so a fast peer is not lost to scheduling. */
    const auto remainingMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - Clock::now()).count();
    const auto bounded = std::clamp<int64_t>(remainingMs, 0, kMaxReceiveMs);
    pollfd ready{peer.NativeHandle(), POLLIN, 0};
    if (::poll(&ready, 1, static_cast<int>(bounded)) <= 0) {
        UC_ERROR("memfd fd receive timed out: owner={} device={} rank={} name={}", ownerRank_,
                 deviceId_, rank, name);
        return Status::Timeout();
    }
    auto status = peer.RecvFd(fd);
    if (status.Failure()) {
        UC_ERROR("memfd fd receive failed: owner={} device={} rank={} name={} status={}",
                 ownerRank_, deviceId_, rank, name, status);
        return status;
    }
    struct stat info{};
    const bool statFailed = ::fstat(fd, &info) != 0;
    if (statFailed || static_cast<size_t>(info.st_size) < rankStride_) {
        ::close(fd);
        return Status::InvalidParam("peer fd of rank({}) is shorter than rank bytes({})", rank,
                                    rankStride_);
    }
    status = ValidatePageBacking(fd);
    if (status.Failure()) {
        ::close(fd);
        return status;
    }
    auto mem = std::make_unique<MemFd>();
    status = mem->Adopt(fd, rankStride_);
    if (status.Failure()) { return status; }
    auto& segment = segments_[rank];
    segment.mem = std::move(mem);
    /* The owner already placed and touched this segment; here it only gets
     * registered in this process's device context. */
    status = RegisterSegment(segment);
    if (status.Failure()) { return status; }
    UC_INFO("memfd peer segment: owner={} device={} rank={} bytes={}", ownerRank_, deviceId_, rank,
            rankStride_);
    return Status::OK();
}

void MemfdDataBackend::FinalizeSetup()
{
    /* Every peer has fetched the local fd, so the socket can go: closing the
     * listener releases the abstract name. Mappings and registrations stay. */
    StopFdServer();
}

void* MemfdDataBackend::HostAddrOf(size_t rank) const
{
    if (rank >= segments_.size() || segments_[rank].mem == nullptr) { return nullptr; }
    return segments_[rank].mem->Addr();
}

void* MemfdDataBackend::DeviceAddrOf(size_t rank) const
{
    (void)rank;
    /* Every memfd segment is host-accessible, so there is no device-only address. */
    return nullptr;
}

void* MemfdDataBackend::HostMappedDeviceAddrOf(size_t rank) const
{
    return rank < segments_.size() ? segments_[rank].deviceAddr : nullptr;
}

void MemfdDataBackend::Reset()
{
    StopFdServer();
    for (auto& segment : segments_) {
        if (segment.registered && segment.mem != nullptr) {
            Trans::Buffer::UnregisterHostBuffer(segment.mem->Addr());
        }
        segment = Segment{};
    }
    segments_.clear();
    rankStride_ = 0;
    backingPageSize_ = 0;
    pageMode_ = MemfdPageMode::Base;
}

Status MemfdDataBackend::CreateLocalSegment(size_t rank)
{
    const auto name = SegmentName(rank);
    auto mem = std::make_unique<MemFd>();
    uint32_t additionalFlags = 0;
    if (pageMode_ == MemfdPageMode::HugeTlb2M) { additionalFlags = MFD_HUGETLB | MFD_HUGE_2MB; }
    auto status = mem->Create(name, rankStride_, additionalFlags);
    if (status.Failure()) {
        UC_ERROR("memfd create failed: owner={} device={} name={} bytes={} page_mode={} status={}",
                 rank, deviceId_, name, rankStride_, MemfdPageModeName(pageMode_), status);
        return {status.Underlying(),
                fmt::format("memfd create failed: name={} bytes={} page_mode={} status={}", name,
                            rankStride_, MemfdPageModeName(pageMode_), status)};
    }
    status = ValidatePageBacking(mem->Fd());
    if (status.Failure()) {
        UC_ERROR("memfd page backing invalid: owner={} device={} name={} page_mode={} status={}",
                 rank, deviceId_, name, MemfdPageModeName(pageMode_), status);
        return status;
    }
    if (pageMode_ == MemfdPageMode::HugeTlb2M) {
        UC_INFO(
            "memfd HugeTLB backing verified: owner={} device={} page_bytes={} pages={} bytes={}",
            rank, deviceId_, backingPageSize_, rankStride_ / backingPageSize_, rankStride_);
    }
    segments_[rank].mem = std::move(mem);
    segments_[rank].owner = true;
    return Status::OK();
}

Status MemfdDataBackend::PlaceAndRegister(size_t rank)
{
    auto& segment = segments_[rank];
    /* Placement has to be settled before the pages are touched, and touching has
     * to happen before registration (which pins them). The bind itself is
     * skipped only while no plan has been resolved yet. */
    if (localPlacement_.node < 0) {
        UC_WARN(
            "memfd segment has no NUMA plan, leaving placement to the kernel: device={} "
            "rank={}",
            deviceId_, rank);
    } else {
        auto bindStatus = Numa::BindBeforeTouch(segment.mem->Addr(), rankStride_, localPlacement_);
        if (bindStatus.Failure()) {
            UC_ERROR("memfd NUMA bind failed: device={} rank={} node={} status={}", deviceId_, rank,
                     localPlacement_.node, bindStatus);
            return bindStatus;
        }
    }
    if (backingPageSize_ != 0) {
        auto* base = static_cast<std::byte*>(segment.mem->Addr());
        for (size_t offset = 0; offset < rankStride_; offset += backingPageSize_) {
            base[offset] = std::byte{0};
        }
    }
    return RegisterSegment(segment);
}

Status MemfdDataBackend::ValidatePageBacking(int32_t fd) const
{
    struct statfs fileSystem{};
    if (::fstatfs(fd, &fileSystem) != 0) {
        const auto error = errno;
        return Status::OsApiError(
            fmt::format("fstatfs memfd failed: errno={} ({})", error, std::strerror(error)));
    }
    const bool hugeTlbFs = static_cast<unsigned long>(fileSystem.f_type) ==
                           static_cast<unsigned long>(HUGETLBFS_MAGIC);
    if (pageMode_ == MemfdPageMode::Base) {
        if (hugeTlbFs) {
            return Status::Error("memfd expected base-page backing, but received HugeTLB backing");
        }
        return Status::OK();
    }
    const bool twoMiB = static_cast<size_t>(fileSystem.f_bsize) == kHugeTlb2MBytes;
    if (!hugeTlbFs || !twoMiB) {
        return Status::Error(
            fmt::format("memfd expected 2MiB HugeTLB backing: fs_magic=0x{:x} block_bytes={}",
                        static_cast<uint64_t>(fileSystem.f_type), fileSystem.f_bsize));
    }
    return Status::OK();
}

Status MemfdDataBackend::RegisterSegment(Segment& segment)
{
    /* Registration acts on the process's current device context, which nothing
     * else here establishes; without it a standalone store could register the
     * segment against the wrong device, or get an unusable SDMA alias. */
    Trans::Device device;
    auto status = device.Setup(deviceId_);
    if (status.Failure()) {
        UC_ERROR("memfd device setup failed: device={} status={}", deviceId_, status);
        return status;
    }
    /* Registration itself is unconditional: the device has to be able to reach
     * this host memory whichever copy path runs. Only the device-visible alias
     * is SDMA-specific, so it is fetched on demand and stays nullptr otherwise. */
    void* deviceAddr = nullptr;
    void** aliasOut = requireHostDeviceAddress_ ? &deviceAddr : nullptr;
    status = Trans::Buffer::RegisterHostBuffer(segment.mem->Addr(), rankStride_, aliasOut);
    if (status.Failure()) {
        UC_ERROR("memfd host register failed: device={} bytes={} status={}", deviceId_, rankStride_,
                 status);
        return status;
    }
    segment.deviceAddr = deviceAddr;
    segment.registered = true;
    return Status::OK();
}

Status MemfdDataBackend::StartFdServer()
{
    const auto name = SocketName(ownerRank_);
    auto status = fdServer_.Listen(name);
    if (status.Failure()) {
        UC_ERROR("memfd fd socket listen failed: owner={} device={} name={} status={}", ownerRank_,
                 deviceId_, name, status);
        return status;
    }
    const size_t peers = segments_.empty() ? 0 : segments_.size() - 1;
    fdThread_ = std::thread([this, peers] { AcceptLoop(peers); });
    return Status::OK();
}

void MemfdDataBackend::AcceptLoop(size_t peerCount)
{
    const auto& local = segments_[ownerRank_];
    const int32_t fd = local.mem == nullptr ? -1 : local.mem->Fd();
    for (size_t sent = 0; sent < peerCount; ++sent) {
        if (fdServer_.AcceptAndSend(fd).Failure()) { break; }
    }
    fdServer_.Close();
}

void MemfdDataBackend::StopFdServer()
{
    fdServer_.Close();
    if (fdThread_.joinable()) { fdThread_.join(); }
}

std::string MemfdDataBackend::SegmentName(size_t rank) const
{
    return "ucm_cache2_" + domainId_ + "_data_" + std::to_string(rank);
}

std::string MemfdDataBackend::SocketName(size_t rank) const
{
    return "ucm_cache2_" + domainId_ + "_fd_" + std::to_string(rank);
}

}  // namespace UC::Cache2
