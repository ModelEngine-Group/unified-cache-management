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
#include <cstdio>
#include <cstring>
#include <utility>
#include "logger/logger.h"

namespace UC::Dram {
namespace {

double ReadCpuLoadAvg1m()
{
    FILE* fp = std::fopen("/proc/loadavg", "r");
    if (fp == nullptr) { return 0.0; }
    double load = 0.0;
    std::fscanf(fp, "%lf", &load);
    std::fclose(fp);
    return load;
}

// Fold shardId into the wire key for non-zero shards so multiple shards of one
// block coexist on the same drampool node (which keys entries by BlockId alone).
// shardId == 0 keeps the original key so lookup (always shard 0) finds the entry.
Detail::BlockId StorageKey(const Detail::BlockId& blockId, std::uint32_t shardId)
{
    if (shardId == 0) { return blockId; }
    Detail::BlockId result = blockId;
    std::uint32_t seed = 0;
    std::memcpy(&seed, result.data(), sizeof(seed));
    seed ^= shardId + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    std::memcpy(result.data(), &seed, sizeof(seed));
    return result;
}

template <typename RequestEntry>
void FillTransferEntries(const std::vector<IoEntry>& entries,
                         std::vector<RequestEntry>& requestEntries)
{
    requestEntries.resize(entries.size());
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& source = entries[index];
        auto& target = requestEntries[index];
        const auto key = StorageKey(source.blockId, source.shardId);
        std::memcpy(target.key.data(), key.data(), key.size());
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

const char* NodeActor::NodeStateName(NodeState state) noexcept
{
    constexpr const char* names[] = {"DISCONNECTED", "CONNECTING", "ACTIVE", "FENCING"};
    return names[static_cast<std::uint8_t>(state)];
}

Status NodeActor::EncodeRequest(const ReplySlot& replySlot, RequestId requestId, OpType op,
                                const std::vector<IoEntry>& entries,
                                std::vector<std::uint8_t>& payload)
{
    const auto batchSize = static_cast<std::uint16_t>(entries.size());
    const auto responseAddress = reinterpret_cast<std::uint64_t>(replySlot.localAddr);
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
            request.opcode = OpType::LOOKUP;
            request.request_id = requestId;
            request.resp_addr = responseAddress;
            request.batch_size = batchSize;
            request.entries.resize(entries.size());
            for (std::size_t index = 0; index < entries.size(); ++index) {
                const auto key = StorageKey(entries[index].blockId, entries[index].shardId);
                std::memcpy(request.entries[index].key.data(), key.data(), key.size());
            }
            return pack(request);
        }
        case OpType::DUMP: {
            DramPool::KvDumpRequest request;
            request.opcode = OpType::DUMP;
            request.request_id = requestId;
            request.resp_addr = responseAddress;
            request.ttl = 0;
            request.batch_size = batchSize;
            FillTransferEntries(entries, request.entries);
            return pack(request);
        }
        case OpType::LOAD: {
            DramPool::KvLoadRequest request;
            request.opcode = OpType::LOAD;
            request.request_id = requestId;
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
    request.timing.completedUs = SteadyNowUs();
    request.timing.completedTsUs = UnixNowUs();
    const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
        return begin != 0 && end >= begin ? end - begin : 0;
    };
    const auto remoteWaitStarted = request.timing.controlTransportCompletedUs != 0
                                       ? request.timing.controlTransportCompletedUs
                                       : request.timing.controlTransportSubmittedUs;
    UC_INFO(
        "[PERF] component=dramstore event=request_done task_id={} request_id={} opcode={} "
        "node_id={} entries={} status={} status_code={} node_queued_ts_us={} "
        "transport_submitted_ts_us={} "
        "transmit_completed_ts_us={} reply_observed_ts_us={} completed_ts_us={} "
        "node_queue_us={} reply_slot_wait_us={} encode_us={} control_submit_us={} "
        "control_transfer_us={} remote_wait_us={} reply_process_us={} total_us={} "
        "cpu_loadavg_1m={:.2f}",
        request.taskId, request.requestId, static_cast<unsigned>(request.op), request.nodeId,
        request.entries.size(), status.Success() ? "SUCCESS" : "FAILED", status.Underlying(),
        request.timing.nodeQueuedTsUs, request.timing.controlTransportSubmittedTsUs,
        request.timing.controlTransportCompletedTsUs, request.timing.replyObservedTsUs,
        request.timing.completedTsUs,
        elapsed(request.timing.nodeQueuedUs, request.timing.nodeActorStartedUs),
        elapsed(request.timing.nodeActorStartedUs, request.timing.replySlotAcquiredUs),
        elapsed(request.timing.replySlotAcquiredUs, request.timing.requestEncodedUs),
        elapsed(request.timing.controlTransportSubmitStartedUs, request.timing.controlTransportSubmittedUs),
        elapsed(request.timing.controlTransportSubmittedUs, request.timing.controlTransportCompletedUs),
        elapsed(remoteWaitStarted, request.timing.replyObservedUs),
        elapsed(request.timing.replyObservedUs, request.timing.replyProcessedUs),
        elapsed(request.timing.nodeQueuedUs, request.timing.completedUs), ReadCpuLoadAvg1m());
    for (std::size_t index = 0; index < entryResults.size(); ++index) {
        entryResults[index].originalIndex = request.entries[index].originalIndex;
    }
    completionBatch_.push_back(RequestCompleted{request.taskId, request.requestId,
                                                config_.endpoint.nodeId, std::move(status),
                                                std::move(entryResults)});
}

void NodeActor::ReleaseReplySlot(RequestRecord& request)
{
    if (request.replySlot.localAddr == nullptr) { return; }
    const auto release = dependencies_.releaseReplySlot(request.token, request.replySlot);
    if (release.Failure()) {
        UC_ERROR(
            "DramStore reply slot release failed, task_id={} request_id={} op={} node_id={} "
            "epoch={} slot_index={} status={}",
            request.request.taskId, request.request.requestId,
            static_cast<unsigned>(request.request.op), config_.endpoint.nodeId, request.token.epoch,
            request.replySlot.slotIndex, release);
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
    std::size_t timedOutCount = 0;
    TaskId firstTaskId = 0;
    RequestId firstRequestId = 0;
    RequestState firstRequestState = RequestState::COMPLETED;
    for (auto it = activeRequests_.begin(); it != activeRequests_.end();) {
        if (it->second.state != RequestState::COMPLETED) {
            if (active && it->second.IsExposed()) {
                nextActionAt_ = std::min(nextActionAt_, it->second.request.deadline);
                if (it->second.request.deadline <= now) {
                    if (!needsFence) {
                        firstTaskId = it->second.request.taskId;
                        firstRequestId = it->second.request.requestId;
                        firstRequestState = it->second.state;
                    }
                    needsFence = true;
                    ++timedOutCount;
                }
            }
            ++it;
            continue;
        }
        const auto requestId = it->first;
        ++it;
        RetireRequest(requestId);
    }

    if (needsFence) {
        std::size_t affectedCount = 0;
        state_ = NodeState::FENCING;
        for (auto& entry : activeRequests_) {
            if (!entry.second.IsExposed()) { continue; }
            ++affectedCount;
            entry.second.state = RequestState::WAITING_FENCE;
            entry.second.failure = Status::Timeout();
        }
        UC_WARN(
            "DramStore request timeout triggered node recovery, node_id={} epoch={} "
            "timed_out_requests={} affected_requests={} pending_requests={} "
            "first_task_id={} first_request_id={} first_request_state={}",
            config_.endpoint.nodeId, epoch_, timedOutCount, affectedCount, pendingRequests_.size(),
            firstTaskId, firstRequestId, RequestStateToString(firstRequestState));
        nextActionAt_ = now;
    }
}

void NodeActor::ExpirePendingRequests(TimePoint now)
{
    if (pendingCheckAt_ > now) { return; }

    pendingCheckAt_ = TimePoint::max();
    std::size_t expiredCount = 0;
    TaskId firstTaskId = 0;
    RequestId firstRequestId = 0;
    for (auto it = pendingRequests_.begin(); it != pendingRequests_.end();) {
        if (it->deadline > now) {
            pendingCheckAt_ = std::min(pendingCheckAt_, it->deadline);
            ++it;
            continue;
        }
        auto request = std::move(*it);
        it = pendingRequests_.erase(it);
        if (expiredCount == 0) {
            firstTaskId = request.taskId;
            firstRequestId = request.requestId;
        }
        ++expiredCount;
        QueueCompletion(std::move(request), Status::Timeout());
    }
    if (expiredCount != 0) {
        UC_WARN(
            "DramStore requests expired while pending, node_id={} node_state={} "
            "expired_requests={} remaining_pending={} first_task_id={} first_request_id={}",
            config_.endpoint.nodeId, NodeStateName(state_), expiredCount, pendingRequests_.size(),
            firstTaskId, firstRequestId);
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
    active.request.timing.nodeActorStartedUs = SteadyNowUs();
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=dramstore event=stage task_id={} request_id={} opcode={} node_id={} "
    //     "stage=REQUEST_DISPATCHED ts_us={}",
    //     active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
    //     config_.endpoint.nodeId, UnixNowUs());
    UC_DEBUG(
        "DramStore request dispatch, task_id={} request_id={} op={} node_id={} epoch={} "
        "entries={}",
        active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
        config_.endpoint.nodeId, epoch_, active.request.entries.size());

    auto acquired = dependencies_.acquireReplySlot(active.token, active.request.op,
                                                   active.request.entries.size());
    if (!acquired) {
        UC_WARN(
            "DramStore reply slot acquisition failed, task_id={} request_id={} op={} "
            "node_id={} epoch={} entries={} status={}",
            active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
            config_.endpoint.nodeId, epoch_, active.request.entries.size(), acquired.Error());
        active.Complete(acquired.Error());
        RetireRequest(requestId);
        return;
    }
    active.replySlot = std::move(acquired).Value();
    active.request.timing.replySlotAcquiredUs = SteadyNowUs();

    std::vector<std::uint8_t> payload;
    auto status = EncodeRequest(active.replySlot, active.request.requestId, active.request.op,
                                active.request.entries, payload);
    if (status.Failure()) {
        UC_ERROR(
            "DramStore request encoding failed, task_id={} request_id={} op={} "
            "node_id={} epoch={} entries={} status={}",
            active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
            config_.endpoint.nodeId, epoch_, active.request.entries.size(), status);
        active.Complete(std::move(status));
        RetireRequest(requestId);
        return;
    }
    active.request.timing.requestEncodedUs = SteadyNowUs();

    const auto payloadSize = payload.size();
    TransportCommand command{
        Transmit{active.token, std::move(payload)}
    };
    active.request.timing.controlTransportSubmitStartedUs = SteadyNowUs();
    status = dependencies_.submitTransport(command);
    active.request.timing.controlTransportSubmittedUs = SteadyNowUs();
    active.request.timing.controlTransportSubmittedTsUs = UnixNowUs();
    if (status.Success()) {
        // Stage PERF log disabled: request_done already contains the corresponding timing.
        // UC_INFO(
        //     "[PERF] component=dramstore event=stage task_id={} request_id={} opcode={} "
        //     "node_id={} stage=CONTROL_TRANSFER_SUBMITTED ts_us={}",
        //     active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
        //     config_.endpoint.nodeId, active.request.timing.controlTransportSubmittedTsUs);
        UC_DEBUG(
            "DramStore request submitted, task_id={} request_id={} op={} node_id={} "
            "epoch={} entries={} payload_bytes={}",
            active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
            config_.endpoint.nodeId, epoch_, active.request.entries.size(), payloadSize);
    } else {
        UC_WARN(
            "DramStore request transport submission failed, task_id={} request_id={} op={} "
            "node_id={} epoch={} entries={} payload_bytes={} status={}",
            active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
            config_.endpoint.nodeId, epoch_, active.request.entries.size(), payloadSize, status);
    }
    if (status.Failure() && active.state != RequestState::COMPLETED) {
        active.Complete(std::move(status));
    }
    if (active.state == RequestState::COMPLETED) { RetireRequest(requestId); }
}

void NodeActor::Handle(Request request, TimePoint now)
{
    if (request.timing.nodeQueuedUs == 0) {
        request.timing.nodeQueuedUs = SteadyNowUs();
        request.timing.nodeQueuedTsUs = UnixNowUs();
    }
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=dramstore event=stage task_id={} request_id={} opcode={} node_id={} "
    //     "stage=NODE_QUEUED ts_us={}",
    //     request.taskId, request.requestId, static_cast<unsigned>(request.op), request.nodeId,
    //     request.timing.nodeQueuedTsUs);
    if (request.deadline <= now) {
        UC_WARN(
            "DramStore request expired before node admission, task_id={} request_id={} op={} "
            "node_id={} node_state={}",
            request.taskId, request.requestId, static_cast<unsigned>(request.op),
            config_.endpoint.nodeId, NodeStateName(state_));
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
    UC_WARN(
        "DramStore node recovery fence submission failed, node_id={} epoch={} "
        "active_requests={} pending_requests={} status={} retry_after_ms={}",
        config_.endpoint.nodeId, epoch_, activeRequests_.size(), pendingRequests_.size(), status,
        config_.reconnectInterval.count());
    nextActionAt_ = now + config_.reconnectInterval;
}

void NodeActor::Handle(FenceCompleted event, TimePoint now)
{
    if (state_ != NodeState::FENCING || event.epoch != epoch_) {
        UC_DEBUG(
            "DramStore ignored stale fence completion, node_id={} event_epoch={} "
            "current_epoch={} node_state={}",
            config_.endpoint.nodeId, event.epoch, epoch_, NodeStateName(state_));
        return;
    }
    if (event.status.Failure()) {
        // A failed fence means the remote peer was unreachable (e.g. the
        // DramPool was killed). An unreachable peer cannot access local
        // registered memory, so the safety property a successful Disconnect
        // would establish already holds. Fall through to DISCONNECTED and let
        // the retry loop re-establish the connection when the remote returns.
        UC_WARN(
            "DramStore node recovery fence failed, node_id={} epoch={} "
            "active_requests={} status={}; continuing with reconnect",
            config_.endpoint.nodeId, epoch_, activeRequests_.size(), event.status);
    } else {
        UC_INFO("DramStore node recovery fence completed, node_id={} epoch={} active_requests={}",
                config_.endpoint.nodeId, epoch_, activeRequests_.size());
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
    UC_WARN(
        "DramStore node connect submission failed, node_id={} epoch={} status={} "
        "retry_after_ms={}",
        config_.endpoint.nodeId, epoch_, status, config_.reconnectInterval.count());
    nextActionAt_ = now + config_.reconnectInterval;
}

void NodeActor::Handle(ReplyObserved event, TimePoint now)
{
    const auto found = activeRequests_.find(event.token.requestId);
    if (found == activeRequests_.end() || found->second.token != event.token ||
        found->second.state == RequestState::COMPLETED) {
        UC_DEBUG(
            "DramStore ignored stale reply, node_id={} request_id={} event_epoch={} "
            "current_epoch={} node_state={}",
            config_.endpoint.nodeId, event.token.requestId, event.token.epoch, epoch_,
            NodeStateName(state_));
        return;
    }
    if (found->second.failure == Status::Timeout() || found->second.request.deadline <= now) {
        UC_WARN(
            "DramStore reply arrived after request timeout, task_id={} request_id={} op={} "
            "node_id={} epoch={} request_state={}",
            found->second.request.taskId, found->second.request.requestId,
            static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId, epoch_,
            RequestStateToString(found->second.state));
        found->second.Complete(Status::Timeout());
        return;
    }
    found->second.request.timing.replyObservedUs = SteadyNowUs();
    found->second.request.timing.replyObservedTsUs = UnixNowUs();
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=dramstore event=stage task_id={} request_id={} opcode={} node_id={} "
    //     "stage=REPLY_OBSERVED ts_us={}",
    //     found->second.request.taskId, found->second.request.requestId,
    //     static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId,
    //     found->second.request.timing.replyObservedTsUs);
    auto status = event.status;
    std::vector<EntryResult> entryResults;
    if (status.Success() && found->second.request.op != OpType::LOOKUP) {
        std::size_t failedEntries = 0;
        std::int32_t firstErrorCode = 0;
        for (const auto& result : event.entryResults) {
            if (result.code != 0) {
                if (failedEntries == 0) { firstErrorCode = result.code; }
                ++failedEntries;
            }
        }
        if (failedEntries != 0) {
            UC_WARN(
                "DramStore request contains failed items, task_id={} request_id={} op={} "
                "node_id={} epoch={} failed_entries={} total_entries={} first_error_code={}",
                found->second.request.taskId, found->second.request.requestId,
                static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId, epoch_,
                failedEntries, event.entryResults.size(), firstErrorCode);
            status = Status::Error("DramPool returned an item failure");
        }
    } else if (status.Success()) {
        entryResults = std::move(event.entryResults);
    }
    if (event.status.Failure()) {
        UC_WARN(
            "DramStore reply processing failed, task_id={} request_id={} op={} node_id={} "
            "epoch={} request_state={} status={}",
            found->second.request.taskId, found->second.request.requestId,
            static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId, epoch_,
            RequestStateToString(found->second.state), event.status);
    }
    found->second.request.timing.replyProcessedUs = SteadyNowUs();
    found->second.Complete(std::move(status), std::move(entryResults));
}

void NodeActor::Handle(TransmitCompleted event, TimePoint)
{
    const auto found = activeRequests_.find(event.token.requestId);
    if (found == activeRequests_.end() || found->second.state != RequestState::TRANSMITTING ||
        found->second.token != event.token) {
        UC_DEBUG(
            "DramStore ignored stale transmit completion, node_id={} request_id={} "
            "event_epoch={} current_epoch={} node_state={}",
            config_.endpoint.nodeId, event.token.requestId, event.token.epoch, epoch_,
            NodeStateName(state_));
        return;
    }
    found->second.request.timing.controlTransportCompletedUs = SteadyNowUs();
    found->second.request.timing.controlTransportCompletedTsUs = UnixNowUs();
    // Stage PERF log disabled: request_done already contains the corresponding timing.
    // UC_INFO(
    //     "[PERF] component=dramstore event=stage task_id={} request_id={} opcode={} node_id={} "
    //     "status={} stage=CONTROL_TRANSFER_COMPLETED ts_us={}",
    //     found->second.request.taskId, found->second.request.requestId,
    //     static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId,
    //     event.status.Success() ? "SUCCESS" : "FAILED",
    //     found->second.request.timing.controlTransportCompletedTsUs);
    if (event.status.Success()) {
        found->second.state = RequestState::INFLIGHT;
        return;
    }
    UC_WARN(
        "DramStore request transmission failed, task_id={} request_id={} op={} node_id={} "
        "epoch={} entries={} status={}",
        found->second.request.taskId, found->second.request.requestId,
        static_cast<unsigned>(found->second.request.op), config_.endpoint.nodeId, epoch_,
        found->second.request.entries.size(), event.status);
    found->second.Complete(std::move(event.status));
}

void NodeActor::Handle(ConnectCompleted event, TimePoint now)
{
    if (event.epoch != epoch_ || state_ != NodeState::CONNECTING) {
        UC_DEBUG(
            "DramStore ignored stale connect completion, node_id={} event_epoch={} "
            "current_epoch={} node_state={}",
            config_.endpoint.nodeId, event.epoch, epoch_, NodeStateName(state_));
        return;
    }
    if (event.status.Success()) {
        state_ = NodeState::ACTIVE;
        assert(activeRequests_.empty());
        nextActionAt_ = TimePoint::max();
        UC_INFO(
            "DramStore node connected, node_id={} epoch={} endpoint={}:{} "
            "pending_requests={}",
            config_.endpoint.nodeId, epoch_, config_.endpoint.controlHost,
            config_.endpoint.controlPort, pendingRequests_.size());
    } else {
        state_ = NodeState::DISCONNECTED;
        nextActionAt_ = now + config_.reconnectInterval;
        UC_WARN(
            "DramStore node connect failed, node_id={} epoch={} endpoint={}:{} status={} "
            "retry_after_ms={}",
            config_.endpoint.nodeId, epoch_, config_.endpoint.controlHost,
            config_.endpoint.controlPort, event.status, config_.reconnectInterval.count());
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
