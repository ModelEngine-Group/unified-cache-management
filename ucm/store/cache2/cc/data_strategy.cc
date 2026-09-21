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
#include <fmt/format.h>
#include <stdexcept>
#include <thread>

#if UCM_RUNTIME_ASCEND_HAL
#include "trans/ascend/hal/hal_host_buffers.h"
#endif

namespace UC::Cache2 {

DataStrategy::DataStrategy() = default;
DataStrategy::~DataStrategy() = default;

void DataStrategy::Setup(CtrlLayout& ctrl, int32_t deviceId, size_t slotSize, size_t nSlotsPerRank,
                         size_t rank, size_t timeoutMs)
{
    if (nSlotsPerRank_ != 0) {
        throw std::logic_error("cache2 data strategy is already initialized");
    }
    const size_t totalSlots = ctrl.SlotCount();
    if (deviceId < 0 || slotSize == 0 || nSlotsPerRank == 0 || totalSlots == 0 ||
        totalSlots % nSlotsPerRank != 0 || rank >= totalSlots / nSlotsPerRank) {
        throw std::invalid_argument(
            fmt::format("invalid cache2 data layout: rank={} device={} slot_size={} "
                        "slots_per_rank={} total_slots={}",
                        rank, deviceId, slotSize, nSlotsPerRank, totalSlots));
    }
    try {
        LocalSetup(deviceId, slotSize * nSlotsPerRank, totalSlots / nSlotsPerRank, rank);
        CrossRankSetup(ctrl, timeoutMs);
        slotSize_ = slotSize;
        nSlotsPerRank_ = nSlotsPerRank;
    } catch (...) {
#if UCM_RUNTIME_ASCEND_HAL
        hostBuffers_.reset();
#endif
        throw;
    }
}

void DataStrategy::LocalSetup(int32_t deviceId, size_t dataBytes, size_t nRanks, size_t rank)
{
#if UCM_RUNTIME_ASCEND_HAL
    hostBuffers_ = std::make_unique<Trans::HalHostBuffers>();
    hostBuffers_->Setup(deviceId, dataBytes, nRanks, rank);
#else
    throw std::runtime_error("cache2 HAL data strategy requires ascend-a5 runtime");
#endif
}

void DataStrategy::CrossRankSetup(CtrlLayout& ctrl, size_t timeoutMs)
{
#if UCM_RUNTIME_ASCEND_HAL
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    CtrlLayout::RankDataDesc desc;
    desc.handle.store(hostBuffers_->ExportHandle(), std::memory_order_relaxed);
    Status status = ctrl.SetRankDesc(hostBuffers_->Owner(), desc);
    if (status.Failure()) {
        throw std::runtime_error(fmt::format("SetRankDesc failed: owner={} device={} status={}",
                                             hostBuffers_->Owner(), hostBuffers_->DeviceId(),
                                             status));
    }
    for (size_t rank = 0; rank < hostBuffers_->RankCount(); ++rank) {
        if (rank == hostBuffers_->Owner()) { continue; }
        Expected<CtrlLayout::RankDataDesc> peerDesc = ctrl.GetRankDesc(rank);
        while (!peerDesc) {
            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (now >= deadline) { break; }
            std::this_thread::sleep_until(std::min(deadline, now + std::chrono::milliseconds(1)));
            if (std::chrono::steady_clock::now() >= deadline) { break; }
            peerDesc = ctrl.GetRankDesc(rank);
        }
        if (!peerDesc) {
            throw std::runtime_error(fmt::format(
                "GetRankDesc timed out: owner={} device={} rank={} timeout_ms={} status={}",
                hostBuffers_->Owner(), hostBuffers_->DeviceId(), rank, timeoutMs,
                peerDesc.Error()));
        }
        const uint64_t peerHandle = peerDesc.Value().handle.load(std::memory_order_relaxed);
        hostBuffers_->ImportHandle(rank, peerHandle);
    }
#else
    throw std::runtime_error("cache2 HAL data strategy requires ascend-a5 runtime");
#endif
}

void* DataStrategy::DataAt(size_t slotIdx)
{
#if UCM_RUNTIME_ASCEND_HAL
    if (nSlotsPerRank_ == 0) { return nullptr; }
    std::byte* data = static_cast<std::byte*>(hostBuffers_->HostData(slotIdx / nSlotsPerRank_));
    return data ? data + (slotIdx % nSlotsPerRank_) * slotSize_ : nullptr;
#else
    return nullptr;
#endif
}

void* DataStrategy::DeviceDataAt(size_t slotIdx)
{
#if UCM_RUNTIME_ASCEND_HAL
    if (nSlotsPerRank_ == 0) { return nullptr; }
    std::byte* data = static_cast<std::byte*>(hostBuffers_->DeviceData(slotIdx / nSlotsPerRank_));
    return data ? data + (slotIdx % nSlotsPerRank_) * slotSize_ : nullptr;
#else
    return nullptr;
#endif
}

}  // namespace UC::Cache2
