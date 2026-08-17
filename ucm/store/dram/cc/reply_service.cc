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
#include "reply_service.h"
#include <atomic>
#include <limits>
#include <system_error>
#include "logger/logger.h"

namespace UC::Dram {

ReplyService::ReplyService(Options options) : options_(std::move(options)) {}

Expected<std::unique_ptr<ReplyService>> ReplyService::Create(Options options)
{
    auto service = std::unique_ptr<ReplyService>(new ReplyService(std::move(options)));
    auto status = service->Init();
    if (status.Failure()) { return status; }
    return service;
}

Status ReplyService::Init()
{
    if (options_.slotSize == 0 || options_.slotCount == 0 || options_.pollInterval.count() <= 0 ||
        !options_.publishEvent) {
        return Status::InvalidParam("invalid ReplyService options");
    }
    auto status = buffers_.Init("dram_reply_slots", BufferPool::MemoryType::Host, options_.slotSize,
                                options_.slotCount, true);
    if (status.Failure()) { return status; }
    slotContexts_ = std::make_unique<SlotContext[]>(options_.slotCount);
    activeLeaseIndices_.reserve(options_.slotCount);
    activeLeasePositions_.assign(options_.slotCount, kNoIndex);
    return Status::OK();
}

UC::DramPool::KvOpcode ReplyService::ToOpcode(OpType op) noexcept
{
    switch (op) {
        case OpType::LOOKUP: return DramPool::KvOpcode::Lookup;
        case OpType::DUMP: return DramPool::KvOpcode::Dump;
        case OpType::LOAD: return DramPool::KvOpcode::Load;
    }
    return DramPool::KvOpcode::None;
}

std::size_t ReplyService::ReplyPayloadSize(OpType op, std::size_t entryCount) const noexcept
{
    if (entryCount == 0 || entryCount > kMaxProtocolBatchEntries) { return 0; }
    const auto opcode = ToOpcode(op);
    return opcode == DramPool::KvOpcode::None ? 0
                                              : protocol_.GetPackedResponseSize(opcode, entryCount);
}

bool ReplyService::CompletionReady(const Lease& lease) const noexcept
{
    if (lease.slot.local_addr == nullptr || lease.payloadSize == 0 ||
        lease.payloadSize > lease.slot.length) {
        return false;
    }
    bool ready = false;
    const auto status = protocol_.IsResponseReady(lease.slot.local_addr, ready);
    if (status.Failure() || !ready) { return false; }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
}

Status ReplyService::DecodeReply(const Lease& lease, std::vector<EntryResult>* entryResults)
{
    if (entryResults == nullptr || lease.entryCount > kMaxProtocolBatchEntries) {
        return Status::InvalidParam("invalid DramPool reply metadata");
    }
    DramPool::KvResponse response;
    auto status = protocol_.UnpackResponse(lease.slot.local_addr, ToOpcode(lease.op),
                                           static_cast<std::uint16_t>(lease.entryCount), response);
    if (status.Failure()) { return status; }
    if (response.results.size() != lease.entryCount) { return Status::DeserializeFailed(); }

    entryResults->clear();
    entryResults->reserve(lease.entryCount);
    for (std::size_t index = 0; index < lease.entryCount; ++index) {
        const auto code = static_cast<std::int32_t>(response.results[index]);
        const auto found = lease.op == OpType::LOOKUP ? code != 0 : code == 0;
        entryResults->push_back(EntryResult{index, found, code});
    }
    return Status::OK();
}

ReplyService::~ReplyService() { Shutdown(); }

ReplyMemoryRegion ReplyService::MemoryRegion() const noexcept
{
    return ReplyMemoryRegion{buffers_.GetLocalAddr(), buffers_.GetTotalSize()};
}

std::size_t ReplyService::Available() const
{
    std::lock_guard lock(activeLeasesMutex_);
    return options_.slotCount - activeLeaseIndices_.size();
}

Status ReplyService::Start()
{
    if (acceptingLeases_.exchange(true, std::memory_order_acq_rel)) {
        return Status::DuplicateKey();
    }
    try {
        worker_ = std::thread([this] { Run(); });
        return Status::OK();
    } catch (const std::system_error& error) {
        acceptingLeases_.store(false, std::memory_order_release);
        return Status::Error(fmt::format("failed to start ReplyService: {}", error.what()));
    }
}

Expected<ReplySlot> ReplyService::Acquire(const RequestToken& token, OpType op,
                                          std::size_t entryCount)
{
    const auto payloadSize = ReplyPayloadSize(op, entryCount);
    if (token.nodeId == std::numeric_limits<NodeId>::max() ||
        token.epoch == kInvalidConnectionEpoch || token.requestId == kInvalidRequestId ||
        entryCount == 0 || payloadSize == 0 || payloadSize > options_.slotSize) {
        return Status::InvalidParam("invalid reply lease request");
    }

    ReplySlot slot;
    if (!acceptingLeases_.load(std::memory_order_acquire)) {
        return Status::Error("ReplyService is stopping");
    }

    const auto allocated = buffers_.Allocate(slot);
    if (allocated == Status::NoSpace()) {
        return Status::Error("ReplyService slot capacity invariant violated");
    }
    if (allocated.Failure()) { return allocated; }
    if (slot.slot_index >= options_.slotCount || slot.local_addr == nullptr || slot.length == 0 ||
        !buffers_.IsValidPointer(slot.local_addr)) {
        (void)buffers_.Free(slot.slot_index);
        return Status::Error("reply buffer pool returned an invalid slot");
    }

    {
        auto& context = slotContexts_[slot.slot_index];
        std::lock_guard slotLock(context.mutex);
        {
            std::unique_lock registryLock(activeLeasesMutex_);
            if (!acceptingLeases_.load(std::memory_order_acquire)) {
                registryLock.unlock();
                const auto released = buffers_.Free(slot.slot_index);
                if (released.Failure()) { return released; }
                return Status::Error("ReplyService is stopping");
            }
            if (context.lease.has_value() || activeLeasePositions_[slot.slot_index] != kNoIndex) {
                registryLock.unlock();
                (void)buffers_.Free(slot.slot_index);
                return Status::Error("reply buffer pool returned an invalid slot");
            }

            context.lease = Lease{token, slot, op, entryCount, payloadSize, false};
            activeLeasePositions_[slot.slot_index] = activeLeaseIndices_.size();
            activeLeaseIndices_.push_back(slot.slot_index);
            ++activeLeaseVersion_;
        }
    }
    wake_.notify_one();
    return slot;
}

Status ReplyService::Release(const RequestToken& token, const ReplySlot& slot) noexcept
{
    try {
        const auto index = static_cast<std::size_t>(slot.slot_index);
        if (index >= options_.slotCount) {
            return Status::InvalidParam("reply slot is not leased");
        }

        auto& context = slotContexts_[index];
        std::unique_lock slotLock(context.mutex);
        if (!context.lease.has_value()) { return Status::InvalidParam("reply slot is not leased"); }
        const auto& lease = *context.lease;
        if (lease.token != token || lease.slot.local_addr != slot.local_addr ||
            lease.slot.length != slot.length) {
            return Status::InvalidParam("reply slot ownership mismatch");
        }

        {
            std::lock_guard registryLock(activeLeasesMutex_);
            const auto activePosition = activeLeasePositions_[index];
            if (activePosition >= activeLeaseIndices_.size() ||
                activeLeaseIndices_[activePosition] != index) {
                return Status::Error("reply active-lease index invariant violated");
            }
        }

        // Free may clear the complete slot. Keep only this slot locked while it
        // does so; a concurrent Acquire of the same index will wait for slotLock.
        auto result = buffers_.Free(slot.slot_index);
        if (result.Failure()) { return result; }

        {
            std::lock_guard registryLock(activeLeasesMutex_);
            const auto activePosition = activeLeasePositions_[index];
            if (activePosition >= activeLeaseIndices_.size() ||
                activeLeaseIndices_[activePosition] != index) {
                AbortDramStore(
                    Status::Error("reply active-lease index changed unexpectedly during release"));
            }
            const auto lastIndex = activeLeaseIndices_.back();
            activeLeaseIndices_[activePosition] = lastIndex;
            activeLeasePositions_[lastIndex] = activePosition;
            activeLeaseIndices_.pop_back();
            activeLeasePositions_[index] = kNoIndex;
            context.lease.reset();
            ++activeLeaseVersion_;
        }
        return result;
    } catch (...) {
        AbortDramStore(Status::Error("ReplyService failed to release a reply lease"));
    }
}

void ReplyService::Run() noexcept
{
    try {
        std::vector<std::size_t> activeLeaseSnapshot;
        activeLeaseSnapshot.reserve(options_.slotCount);
        while (acceptingLeases_.load(std::memory_order_acquire)) {
            std::uint64_t observedActiveLeaseVersion = 0;
            {
                std::lock_guard lock(activeLeasesMutex_);
                activeLeaseSnapshot.assign(activeLeaseIndices_.begin(), activeLeaseIndices_.end());
                observedActiveLeaseVersion = activeLeaseVersion_;
            }
            bool progress = false;
            for (const auto index : activeLeaseSnapshot) {
                std::optional<ReplyObserved> observed;
                RequestToken token;
                {
                    auto& context = slotContexts_[index];
                    std::lock_guard slotLock(context.mutex);
                    auto& current = context.lease;
                    if (!current.has_value() || current->delivered) { continue; }
                    if (!CompletionReady(*current)) { continue; }
                    ReplyObserved event;
                    event.token = current->token;
                    event.status = DecodeReply(*current, &event.entryResults);
                    if (event.status == Status::Retry()) { continue; }
                    token = current->token;
                    observed.emplace(std::move(event));
                    current->delivered = true;
                }

                options_.publishEvent(token.nodeId, NodeEvent{std::move(*observed)});
                progress = true;
            }

            if (!progress) {
                std::unique_lock lock(activeLeasesMutex_);
                wake_.wait_for(lock, options_.pollInterval, [this, observedActiveLeaseVersion] {
                    return !acceptingLeases_.load(std::memory_order_acquire) ||
                           activeLeaseVersion_ != observedActiveLeaseVersion;
                });
            }
        }
    } catch (...) {
        AbortDramStore(Status::Error("ReplyService observer stopped unexpectedly"));
    }
}

void ReplyService::Shutdown()
{
    acceptingLeases_.store(false, std::memory_order_release);
    wake_.notify_all();
    if (worker_.joinable()) { worker_.join(); }

    std::lock_guard lock(activeLeasesMutex_);
    if (!activeLeaseIndices_.empty()) {
        UC_ERROR("ReplyService stopped with {} abandoned reply leases", activeLeaseIndices_.size());
    }
}

}  // namespace UC::Dram
