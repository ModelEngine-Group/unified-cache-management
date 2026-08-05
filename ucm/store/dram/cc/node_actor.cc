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
#include "node_actor.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>
#include "logger/logger.h"

namespace UC::Dram {
namespace {

template <typename RequestEntry>
void FillTransferEntries(const std::vector<IoEntry>& entries,
                         std::vector<RequestEntry>& requestEntries)
{
    requestEntries.resize(entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& source = entries[index];
        auto& target = requestEntries[index];
        std::memcpy(target.key.data(), source.blockId.data(), source.blockId.size());
        target.addr = source.buffer.address;
        target.len = static_cast<std::uint32_t>(source.buffer.length);
        target.idx = source.shardId;
    }
}

}  // namespace

NodeActor::NodeActor(Config config, NodeDependencies dependencies)
    : config_(std::move(config)), dependencies_(std::move(dependencies))
{
}

Status NodeActor::EncodeRequest(const ReplySlot& replySlot, OpType op,
                                const std::vector<IoEntry>& entries,
                                std::vector<std::uint8_t>& payload)
{
    const auto batchSize = static_cast<std::uint16_t>(entries.size());
    const auto responseAddress = reinterpret_cast<std::uint64_t>(replySlot.local_addr);
    const auto pack = [this, &payload](const DramPool::KvRequest& request) {
        const auto size = protocol_.GetPackedRequestSize(request.opcode, request);
        payload.resize(size);
        auto status = protocol_.PackRequest(payload.data(), request.opcode, request);
        if (status.Failure()) { payload.clear(); }
        return status;
    };

    switch (op) {
        case OpType::LOOKUP: {
            DramPool::KvLookupRequest request;
            request.opcode = DramPool::KvOpcode::Lookup;
            request.resp_addr = responseAddress;
            request.batch_size = batchSize;
            request.entries.resize(entries.size());
            for (std::size_t index = 0; index < entries.size(); ++index) {
                std::memcpy(request.entries[index].key.data(), entries[index].blockId.data(),
                            entries[index].blockId.size());
            }
            return pack(request);
        }
        case OpType::DUMP: {
            DramPool::KvDumpRequest request;
            request.opcode = DramPool::KvOpcode::Dump;
            request.resp_addr = responseAddress;
            request.ttl = 0;
            request.batch_size = batchSize;
            FillTransferEntries(entries, request.entries);
            return pack(request);
        }
        case OpType::LOAD: {
            DramPool::KvLoadRequest request;
            request.opcode = DramPool::KvOpcode::Load;
            request.resp_addr = responseAddress;
            request.batch_size = batchSize;
            FillTransferEntries(entries, request.entries);
            return pack(request);
        }
    }
    return Status::InvalidParam("unsupported request operation");
}

void NodeActor::QueueCompletion(Request request, Status status,
                                std::vector<EntryResult> entryResults)
{
    for (std::size_t index = 0; index < entryResults.size(); ++index) {
        entryResults[index].originalIndex = request.entries[index].originalIndex;
    }
    completionBatch_.push_back(RequestCompleted{request.taskId, request.requestId,
                                                config_.endpoint.nodeId, std::move(status),
                                                std::move(entryResults)});
}

void NodeActor::ReleaseReplySlot(RequestRecord& request)
{
    if (request.replySlot.local_addr == nullptr) { return; }
    const auto release = dependencies_.releaseReplySlot(request.token, request.replySlot);
    if (release.Failure()) {
        UC_ERROR("DramStore node {} failed to release request {} reply slot: {}",
                 config_.endpoint.nodeId, request.request.requestId, release);
    }
}

void NodeActor::RetireRequest(RequestId requestId)
{
    const auto found = activeRequests_.find(requestId);
    assert(found != activeRequests_.end());
    assert(found->second.state == RequestState::COMPLETED);
    auto request = std::move(found->second);
    activeRequests_.erase(found);
    ReleaseReplySlot(request);
    QueueCompletion(std::move(request.request), std::move(request.failure),
                    std::move(request.entryResults));
}

void NodeActor::FinalizeRequests(TimePoint now)
{
    const bool active = state_ == NodeState::ACTIVE;
    if (active) { nextActionAt_ = TimePoint::max(); }

    bool needsFence = false;
    for (auto it = activeRequests_.begin(); it != activeRequests_.end();) {
        if (it->second.state != RequestState::COMPLETED) {
            if (active && it->second.IsExposed()) {
                nextActionAt_ = std::min(nextActionAt_, it->second.request.deadline);
                if (it->second.request.deadline <= now) { needsFence = true; }
            }
            ++it;
            continue;
        }
        const auto requestId = it->first;
        ++it;
        RetireRequest(requestId);
    }

    if (needsFence) {
        state_ = NodeState::FENCING;
        for (auto& entry : activeRequests_) {
            if (!entry.second.IsExposed()) { continue; }
            entry.second.state = RequestState::WAITING_FENCE;
            entry.second.failure = Status::Timeout();
        }
        nextActionAt_ = now;
    }
}

void NodeActor::ExpirePendingRequests(TimePoint now)
{
    if (pendingCheckAt_ > now) { return; }

    pendingCheckAt_ = TimePoint::max();
    for (auto it = pendingRequests_.begin(); it != pendingRequests_.end();) {
        if (it->deadline > now) {
            pendingCheckAt_ = std::min(pendingCheckAt_, it->deadline);
            ++it;
            continue;
        }
        auto request = std::move(*it);
        it = pendingRequests_.erase(it);
        QueueCompletion(std::move(request), Status::Timeout());
    }
}

void NodeActor::DispatchPendingRequests()
{
    if (state_ != NodeState::ACTIVE) { return; }
    while (!pendingRequests_.empty() &&
           activeRequests_.size() < config_.limits.maxInflightRequests) {
        auto request = std::move(pendingRequests_.front());
        pendingRequests_.pop_front();
        StartRequest(std::move(request));
    }
    if (pendingRequests_.empty()) { pendingCheckAt_ = TimePoint::max(); }
}

void NodeActor::FlushCompletions()
{
    if (completionBatch_.empty()) { return; }
    dependencies_.publishCompletion(completionBatch_);
    completionBatch_.clear();
}

void NodeActor::StartRequest(Request request)
{
    const auto requestId = request.requestId;
    RequestRecord record{std::move(request)};
    record.token = RequestToken{config_.endpoint.nodeId, kDefaultLaneId, epoch_, requestId};
    auto inserted = activeRequests_.emplace(requestId, std::move(record));
    assert(inserted.second);
    auto& active = inserted.first->second;

    auto acquired = dependencies_.acquireReplySlot(active.token, active.request.op,
                                                   active.request.entries.size());
    if (!acquired) {
        active.Complete(acquired.Error());
        RetireRequest(requestId);
        return;
    }
    active.replySlot = std::move(acquired).Value();

    std::vector<std::uint8_t> payload;
    auto status =
        EncodeRequest(active.replySlot, active.request.op, active.request.entries, payload);
    if (status.Failure()) {
        active.Complete(std::move(status));
        RetireRequest(requestId);
        return;
    }

    TransportCommand command{
        Transmit{active.token, std::move(payload)}
    };
    status = dependencies_.submitTransport(command);
    if (status.Failure() && active.state != RequestState::COMPLETED) {
        active.Complete(std::move(status));
    }
    if (active.state == RequestState::COMPLETED) { RetireRequest(requestId); }
}

void NodeActor::Handle(Request request, TimePoint now)
{
    if (request.deadline <= now) {
        QueueCompletion(std::move(request), Status::Timeout());
        return;
    }
    pendingCheckAt_ = std::min(pendingCheckAt_, request.deadline);
    pendingRequests_.push_back(std::move(request));
}

void NodeActor::TryFence(TimePoint now)
{
    TransportCommand command{
        FenceEpoch{config_.endpoint.nodeId, kDefaultLaneId, epoch_}
    };
    const auto status = dependencies_.submitTransport(command);
    if (status.Success()) {
        nextActionAt_ = TimePoint::max();
        return;
    }
    // Submission failure leaves the runtime recovery fence pending.
    nextActionAt_ = now + config_.reconnectInterval;
}

void NodeActor::Handle(FenceCompleted event, TimePoint now)
{
    if (state_ != NodeState::FENCING || event.epoch != epoch_) { return; }
    if (event.status.Failure()) {
        AbortDramStore(Status::Error(fmt::format("DramStore node {} recovery fence failed: {}",
                                                 config_.endpoint.nodeId, event.status)));
    }

    for (auto& entry : activeRequests_) {
        if (entry.second.state == RequestState::WAITING_FENCE) {
            entry.second.state = RequestState::COMPLETED;
        }
    }

    ++epoch_;
    if (epoch_ == kInvalidConnectionEpoch) { ++epoch_; }
    state_ = NodeState::DISCONNECTED;
    nextActionAt_ = now;
}

void NodeActor::TryConnect(TimePoint now)
{
    TransportCommand command{
        Connect{config_.endpoint.nodeId, kDefaultLaneId, epoch_,
                config_.endpoint.transportManagerId}
    };
    const auto status = dependencies_.submitTransport(command);
    if (status.Success()) {
        state_ = NodeState::CONNECTING;
        nextActionAt_ = TimePoint::max();
        return;
    }
    // Connect submission failures are operational failures; retry while disconnected.
    nextActionAt_ = now + config_.reconnectInterval;
}

void NodeActor::Handle(ReplyObserved event, TimePoint now)
{
    const auto found = activeRequests_.find(event.token.requestId);
    if (found == activeRequests_.end() || found->second.token != event.token ||
        found->second.state == RequestState::COMPLETED) {
        return;
    }
    if (found->second.failure == Status::Timeout() || found->second.request.deadline <= now) {
        found->second.Complete(Status::Timeout());
        return;
    }
    auto status = event.status;
    std::vector<EntryResult> entryResults;
    if (status.Success() && found->second.request.op != OpType::LOOKUP) {
        for (const auto& result : event.entryResults) {
            if (result.code != 0) {
                status = Status::Error("DramPool returned an item failure");
                break;
            }
        }
    } else if (status.Success()) {
        entryResults = std::move(event.entryResults);
    }
    found->second.Complete(std::move(status), std::move(entryResults));
}

void NodeActor::Handle(TransmitCompleted event, TimePoint)
{
    const auto found = activeRequests_.find(event.token.requestId);
    if (found == activeRequests_.end() || found->second.state != RequestState::TRANSMITTING ||
        found->second.token != event.token) {
        return;
    }
    if (event.status.Success()) {
        found->second.state = RequestState::INFLIGHT;
        return;
    }
    found->second.Complete(std::move(event.status));
}

void NodeActor::Handle(ConnectCompleted event, TimePoint now)
{
    if (event.epoch != epoch_ || state_ != NodeState::CONNECTING) { return; }
    if (event.status.Success()) {
        state_ = NodeState::ACTIVE;
        assert(activeRequests_.empty());
        nextActionAt_ = TimePoint::max();
    } else {
        state_ = NodeState::DISCONNECTED;
        nextActionAt_ = now + config_.reconnectInterval;
    }
}

void NodeActor::Handle(NodeEvent event, TimePoint now)
{
    std::visit([this, now](auto&& message) { Handle(std::move(message), now); }, std::move(event));
}

void NodeActor::Advance(TimePoint now)
{
    FinalizeRequests(now);
    if (nextActionAt_ <= now) {
        if (state_ == NodeState::DISCONNECTED) {
            TryConnect(now);
        } else if (state_ == NodeState::FENCING) {
            TryFence(now);
        }
    }
    ExpirePendingRequests(now);
    DispatchPendingRequests();
    FlushCompletions();
}

NodeActor::TimePoint NodeActor::NextWakeup() const noexcept
{
    return std::min(nextActionAt_, pendingCheckAt_);
}

}  // namespace UC::Dram
