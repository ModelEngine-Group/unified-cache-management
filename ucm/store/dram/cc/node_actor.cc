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
#include "time/now_time.h"

namespace UC::Dram {
namespace {

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

void RecordRequestCompletionMetrics(OpType op, const Status& status, double started)
{
    UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(op, "requests_completed_total"), 1.0);
    if (status.Failure()) {
        UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(op, "requests_failed_total"), 1.0);
    }
    if (status == Status::Timeout()) {
        UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(op, "request_timeouts_total"), 1.0);
    }
    UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(op, "request_duration_ms"),
                             (NowTime::Now() - started) * 1e3);
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
    RecordRequestCompletionMetrics(request.op, status, request.metricsStarted);
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
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_fence_timeout_triggers_total"), 1.0);
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
        if (it == pendingRequests_.begin()) { nextPrerequisitePollAt_ = TimePoint::min(); }
        auto request = std::move(*it);
        it = pendingRequests_.erase(it);
        UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(request.op, "request_pending_duration_ms"),
                                 (NowTime::Now() - request.metricsPendingStarted) * 1e3);
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

void NodeActor::DispatchPendingRequests(TimePoint now)
{
    if (state_ != NodeState::ACTIVE) { return; }
    while (!pendingRequests_.empty() &&
           activeRequests_.size() < config_.limits.maxInflightRequests) {
        auto& request = pendingRequests_.front();
        if (request.prerequisiteHandle != 0 && now < nextPrerequisitePollAt_) { break; }
        if (!TryStartRequest(request, now)) {
            nextPrerequisitePollAt_ = now + config_.pollInterval;
            break;
        }
        pendingRequests_.pop_front();
        nextPrerequisitePollAt_ = TimePoint::min();
        if (state_ != NodeState::ACTIVE) { break; }
    }
    if (pendingRequests_.empty()) { pendingCheckAt_ = TimePoint::max(); }
}

void NodeActor::FlushCompletions()
{
    if (completionBatch_.empty()) { return; }
    dependencies_.publishCompletion(completionBatch_);
    completionBatch_.clear();
}

bool NodeActor::TryStartRequest(Request& request, TimePoint now)
{
    if (request.deadline <= now) {
        QueueCompletion(std::move(request), Status::Timeout());
        return true;
    }
    if (request.prerequisiteHandle != 0) {
        auto ready = [&]() -> Expected<bool> {
            try {
                return dependencies_.queryPrerequisite(request.prerequisiteHandle);
            } catch (...) {
                return Status::Error("prerequisite event query threw");
            }
        }();
        if (!ready) {
            QueueCompletion(std::move(request), ready.Error());
            return true;
        }
        if (!ready.Value()) { return false; }
        request.prerequisiteHandle = 0;
    }
    StartRequest(std::move(request));
    return true;
}

void NodeActor::StartRequest(Request request)
{
    const auto prepareStarted = NowTime::Now();
    UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(request.op, "request_pending_duration_ms"),
                             (prepareStarted - request.metricsPendingStarted) * 1e3);
    const auto requestId = request.requestId;
    RequestRecord record{std::move(request)};
    record.token = RequestToken{config_.endpoint.nodeId, kDefaultLaneId, epoch_, requestId};
    auto inserted = activeRequests_.emplace(requestId, std::move(record));
    assert(inserted.second);
    auto& active = inserted.first->second;
    UC_DEBUG(
        "DramStore request dispatch, task_id={} request_id={} op={} node_id={} epoch={} "
        "entries={}",
        active.request.taskId, requestId, static_cast<unsigned>(active.request.op),
        config_.endpoint.nodeId, epoch_, active.request.entries.size());

    auto acquired = dependencies_.acquireReplySlot(active.token, active.request.op,
                                                   active.request.entries.size());
    if (!acquired) {
        if (acquired.Error() == Status::NoSpace()) {
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_reply_slot_nospace_total"), 1.0);
        }
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

    const auto payloadSize = payload.size();
    const auto transportQueuedAt = NowTime::Now();
    UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(active.request.op, "request_setup_duration_ms"),
                             (transportQueuedAt - prepareStarted) * 1e3);
    TransportCommand command{
        Transmit{active.token, active.request.op, std::move(payload), transportQueuedAt}
    };
    status = dependencies_.submitTransport(command);
    if (status.Success()) {
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
    const auto metricsNow = NowTime::Now();
    UC::Metrics::UpdateStats(DRAMSTORE_OP_METRIC(request.op, "request_queue_duration_ms"),
                             (metricsNow - request.metricsStarted) * 1e3);
    request.metricsPendingStarted = metricsNow;
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
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_fence_attempts_total"), 1.0);
    TransportCommand command{
        FenceEpoch{config_.endpoint.nodeId, kDefaultLaneId, epoch_}
    };
    const auto status = dependencies_.submitTransport(command);
    if (status.Success()) {
        nextActionAt_ = TimePoint::max();
        return;
    }
    // Submission failure leaves the runtime recovery fence pending.
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_fence_failures_total"), 1.0);
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
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_fence_failures_total"), 1.0);
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
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_connect_attempts_total"), 1.0);
    TransportCommand command{
        Connect{config_.endpoint.nodeId, kDefaultLaneId, epoch_, config_.endpoint.peerAddr}
    };
    const auto status = dependencies_.submitTransport(command);
    if (status.Success()) {
        state_ = NodeState::CONNECTING;
        nextActionAt_ = TimePoint::max();
        return;
    }
    // Connect submission failures are operational failures; retry while disconnected.
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_connect_failures_total"), 1.0);
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
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_stale_replies_total"), 1.0);
        return;
    }
    // ReplyObserved and TransmitCompleted are published by different threads. A fast reply may
    // therefore be handled first, in which case the remote phase has no valid start time yet.
    if (found->second.remoteStarted != 0.0) {
        UC::Metrics::UpdateStats(
            DRAMSTORE_OP_METRIC(found->second.request.op, "request_remote_duration_ms"),
            (NowTime::Now() - found->second.remoteStarted) * 1e3);
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
    if (event.status.Success()) {
        found->second.state = RequestState::INFLIGHT;
        found->second.remoteStarted = event.metricsCompletedAt;
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
        UC_INFO("DramStore node connected, node_id={} epoch={} endpoint={} pending_requests={}",
                config_.endpoint.nodeId, epoch_, config_.endpoint.peerAddr,
                pendingRequests_.size());
    } else {
        state_ = NodeState::DISCONNECTED;
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("dramstore_connect_failures_total"), 1.0);
        nextActionAt_ = now + config_.reconnectInterval;
        UC_WARN(
            "DramStore node connect failed, node_id={} epoch={} endpoint={} status={} "
            "retry_after_ms={}",
            config_.endpoint.nodeId, epoch_, config_.endpoint.peerAddr, event.status,
            config_.reconnectInterval.count());
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
    DispatchPendingRequests(now);
    FlushCompletions();
}

NodeActor::TimePoint NodeActor::NextWakeup() const noexcept
{
    auto next = std::min(nextActionAt_, pendingCheckAt_);
    if (state_ == NodeState::ACTIVE && !pendingRequests_.empty() &&
        activeRequests_.size() < config_.limits.maxInflightRequests &&
        pendingRequests_.front().prerequisiteHandle != 0) {
        next = std::min(next, nextPrerequisitePollAt_);
    }
    return next;
}

}  // namespace UC::Dram
