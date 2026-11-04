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
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fmt/format.h>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include "ctrl_layout.h"
#include "data_backend.h"
#include "data_backend_ascend.h"
#include "data_backend_memfd.h"
#include "logger/logger.h"
#include "numa/numa_policy.h"
#include "status/status.h"

namespace UC::Cache2 {

/* Everything needed to set up the data plane for one worker. Assembled from
 * configuration plus connector-derived hints; not a user-facing knob set. */
struct DataOptions {
    std::string domainId{}; /* connector uniqueId; do not widen scope here */
    int32_t deviceId{-1};   /* device ordinal */
    size_t myRank{0};       /* control-plane partition index */
    size_t slotSize{0};
    size_t slotsPerRank{0};
    size_t setupTimeoutMs{600000};                 /* whole-Setup budget; 0 means no wait */
    bool requireHostDeviceAddress{false};          /* derived from the copy mode, not a switch */
    std::optional<int32_t> deviceNumaNode{};       /* device-layer probe; empty when unknown */
    std::optional<size_t> fallbackNumaRank{};      /* local worker rank, NOT myRank */
    std::optional<size_t> fallbackNumaRankCount{}; /* all workers on this machine */
};

class DataStrategy {
    size_t slotSize_{};
    size_t nSlotsPerRank_{};
    std::unique_ptr<DataBackend> backend_;

    /* Every host-memory platform uses memfd. The HAL branch remains available
     * for an explicitly enabled experimental build. */
    static std::unique_ptr<DataBackend> MakeBackend(const std::string& uniqueId)
    {
#if UCM_RUNTIME_ASCEND_HAL
        (void)uniqueId;
        return std::make_unique<AscendHalDataBackend>();
#else
        return std::make_unique<MemfdDataBackend>(uniqueId);
#endif
    }

    Status PublishLocal(CtrlLayout& ctrl, size_t myRank, DataBackend& backend)
    {
        uint64_t shareHandle = 0;
        Status status = backend.ExportLocal(&shareHandle);
        if (status.Failure()) {
            UC_ERROR("export local segment failed: backend={} owner={} status={}", backend.Name(),
                     myRank, status);
            return status;
        }
        CtrlLayout::RankDataDesc desc;
        desc.handle.store(shareHandle, std::memory_order_relaxed);
        status = ctrl.SetRankDesc(myRank, desc);
        if (status.Failure()) {
            UC_ERROR("SetRankDesc failed: owner={} status={}", myRank, status);
            return {status.Underlying(),
                    fmt::format("SetRankDesc failed: owner={} status={}", myRank, status)};
        }
        return Status::OK();
    }

    using Clock = std::chrono::steady_clock;

    Status ImportPeers(CtrlLayout& ctrl, size_t nRanks, size_t myRank, Clock::time_point deadline,
                       size_t timeoutMs, DataBackend& backend)
    {
        for (size_t rank = 0; rank < nRanks; ++rank) {
            if (rank == myRank) { continue; }
            Expected<CtrlLayout::RankDataDesc> peerDesc = ctrl.GetRankDesc(rank);
            while (!peerDesc) {
                const Clock::time_point now = Clock::now();
                if (now >= deadline) { break; }
                std::this_thread::sleep_until(
                    std::min(deadline, now + std::chrono::milliseconds(10)));
                if (Clock::now() >= deadline) { break; }
                peerDesc = ctrl.GetRankDesc(rank);
            }
            if (!peerDesc) {
                UC_ERROR("GetRankDesc timed out: owner={} rank={} timeout_ms={} status={}", myRank,
                         rank, timeoutMs, peerDesc.Error());
                return {
                    Status::Timeout().Underlying(),
                    fmt::format("GetRankDesc timed out: owner={} rank={} timeout_ms={} status={}",
                                myRank, rank, timeoutMs, peerDesc.Error())};
            }
            const uint64_t peerHandle = peerDesc.Value().handle.load(std::memory_order_relaxed);
            Status status = backend.ImportPeer(rank, peerHandle);
            if (status.Failure()) {
                UC_ERROR(
                    "import peer segment failed: backend={} owner={} rank={} peer_handle={} "
                    "status={}",
                    backend.Name(), myRank, rank, peerHandle, status);
                return {status.Underlying(),
                        fmt::format("import peer failed: rank={} status={}", rank, status)};
            }
        }
        return Status::OK();
    }

    /* Waits until every peer marked itself ready (i.e. mapped all segments),
     * so that name-based segment resources can be released safely. */
    Status WaitPeersReady(CtrlLayout& ctrl, size_t nRanks, size_t myRank,
                          Clock::time_point deadline, size_t timeoutMs)
    {
        for (size_t rank = 0; rank < nRanks; ++rank) {
            if (rank == myRank) { continue; }
            while (!ctrl.IsRankDataReady(rank)) {
                const Clock::time_point now = Clock::now();
                if (now >= deadline) { break; }
                std::this_thread::sleep_until(
                    std::min(deadline, now + std::chrono::milliseconds(10)));
                if (Clock::now() >= deadline) { break; }
            }
            if (!ctrl.IsRankDataReady(rank)) {
                UC_ERROR("peer data ready timed out: owner={} rank={} timeout_ms={}", myRank, rank,
                         timeoutMs);
                return {Status::Timeout().Underlying(),
                        fmt::format("peer data ready timed out: owner={} rank={} timeout_ms={}",
                                    myRank, rank, timeoutMs)};
            }
        }
        return Status::OK();
    }

public:
    DataStrategy() = default;
    ~DataStrategy() = default;
    DataStrategy(const DataStrategy&) = delete;
    DataStrategy& operator=(const DataStrategy&) = delete;

    // Returns a failure status after releasing resources acquired by this call.
    // Ranks must set up concurrently: peer imports and the all-ranks-ready
    // barrier share timeoutMs; zero allows one read attempt per peer.
    Status Setup(CtrlLayout& ctrl, const DataOptions& options)
    {
        const size_t myRank = options.myRank;
        const size_t slotSize = options.slotSize;
        const size_t nSlotsPerRank = options.slotsPerRank;
        const size_t timeoutMs = options.setupTimeoutMs;
        if (nSlotsPerRank_ != 0) {
            return Status::Error("cache2 data strategy is already initialized");
        }
        if (options.domainId.empty()) { return Status::InvalidParam("cache2 uniqueId is empty"); }
        const size_t totalSlots = ctrl.SlotCount();
        const size_t nRanks = nSlotsPerRank != 0 ? totalSlots / nSlotsPerRank : 0;
        if (nRanks == 0 || myRank >= nRanks) {
            return Status::InvalidParam(
                "invalid cache2 data strategy geometry: rank={} ranks={} slots={} "
                "slots_per_rank={}",
                myRank, nRanks, totalSlots, nSlotsPerRank);
        }
        auto backend = MakeBackend(options.domainId);
        const Clock::time_point deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        BackendOptions backendOptions;
        backendOptions.deviceId = options.deviceId;
        backendOptions.rankCount = nRanks;
        backendOptions.rankBytes = slotSize * nSlotsPerRank;
        backendOptions.requireHostDeviceAddress = options.requireHostDeviceAddress;
        backendOptions.deadline = deadline;
        const size_t numaRankCount = options.fallbackNumaRankCount.value_or(nRanks);
        auto placement =
            Numa::Resolve(options.deviceNumaNode, options.fallbackNumaRank, numaRankCount);
        if (!placement) {
            UC_ERROR("cache2 numa placement failed: owner={} device={} status={}", myRank,
                     options.deviceId, placement.Error());
            return {placement.Error().Underlying(),
                    fmt::format("numa placement failed: {}", placement.Error())};
        }
        backendOptions.localPlacement = placement.Value();
        Status status = Status::OK();
        try {
            status = backend->Setup(backendOptions);
            if (status.Success()) { status = backend->BindLocal(myRank); }
            if (status.Success()) { status = PublishLocal(ctrl, myRank, *backend); }
            if (status.Success()) {
                status = ImportPeers(ctrl, nRanks, myRank, deadline, timeoutMs, *backend);
            }
            if (status.Success()) { status = ctrl.MarkRankDataReady(myRank); }
            if (status.Success()) {
                status = WaitPeersReady(ctrl, nRanks, myRank, deadline, timeoutMs);
            }
            if (status.Success()) { backend->FinalizeSetup(); }
        } catch (const std::bad_alloc&) {
            status = Status::OutOfMemory();
        } catch (const std::exception& error) {
            status = Status::Error(error.what());
        }
        if (status.Failure()) {
            UC_ERROR("cache2 data setup failed: backend={} owner={} device={} status={}",
                     backend->Name(), myRank, options.deviceId, status);
            return status;
        }
        slotSize_ = slotSize;
        nSlotsPerRank_ = nSlotsPerRank;
        backend_ = std::move(backend);
        return Status::OK();
    }

    // True when the slot has a CPU/IO-accessible address; callers must then
    // use DataAt exclusively and never query DeviceDataAt for such slots.
    bool HostAccessibleOf(size_t slotIdx) const { return DataAt(slotIdx) != nullptr; }

    // Returns the CPU/IO-accessible address, or nullptr for host-inaccessible slots.
    void* DataAt(size_t slotIdx) const
    {
        if (nSlotsPerRank_ == 0 || backend_ == nullptr) { return nullptr; }
        const size_t rank = slotIdx / nSlotsPerRank_;
        void* base = backend_->HostAddrOf(rank);
        if (base == nullptr) { return nullptr; }
        return static_cast<std::byte*>(base) + (slotIdx % nSlotsPerRank_) * slotSize_;
    }

    // Returns the device-accessible address, or nullptr; only queried for
    // slots whose HostAccessibleOf is false.
    void* DeviceDataAt(size_t slotIdx) const
    {
        if (nSlotsPerRank_ == 0 || backend_ == nullptr) { return nullptr; }
        const size_t rank = slotIdx / nSlotsPerRank_;
        void* base = backend_->DeviceAddrOf(rank);
        if (base == nullptr) { return nullptr; }
        return static_cast<std::byte*>(base) + (slotIdx % nSlotsPerRank_) * slotSize_;
    }

    // Returns the device-visible alias of a host-accessible slot, or nullptr
    // when the backend provides none. Only for transfer modes that need a device
    // address (Memfd SDMA Direct); never use it for CPU access.
    void* HostMappedDeviceDataAt(size_t slotIdx) const
    {
        if (nSlotsPerRank_ == 0 || backend_ == nullptr) { return nullptr; }
        const size_t rank = slotIdx / nSlotsPerRank_;
        void* base = backend_->HostMappedDeviceAddrOf(rank);
        if (base == nullptr) { return nullptr; }
        return static_cast<std::byte*>(base) + (slotIdx % nSlotsPerRank_) * slotSize_;
    }
};

}  // namespace UC::Cache2
