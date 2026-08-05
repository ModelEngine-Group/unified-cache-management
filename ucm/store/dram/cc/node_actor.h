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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_NODE_ACTOR_H
#define UNIFIEDCACHE_DRAM_STORE_CC_NODE_ACTOR_H

#include <chrono>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kv_protocol.h"
#include "messages.h"
#include "status/status.h"

namespace UC::Dram {

class NodeActor final {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    struct Config {
        NodeEndpoint endpoint;
        NodeLimits limits;
        std::chrono::milliseconds reconnectInterval{0};
    };

    NodeActor(Config config, NodeDependencies dependencies);

    NodeActor(const NodeActor&) = delete;
    NodeActor& operator=(const NodeActor&) = delete;

    void Handle(Request request, TimePoint now);
    void Handle(NodeEvent event, TimePoint now);
    void Advance(TimePoint now);
    TimePoint NextWakeup() const noexcept;

private:
    enum class NodeState : std::uint8_t {
        DISCONNECTED = 0,
        CONNECTING,
        ACTIVE,
        FENCING,
    };

    struct RequestRecord {
        Request request;
        RequestState state{RequestState::TRANSMITTING};
        RequestToken token;
        ReplySlot replySlot;
        Status failure{Status::OK()};
        std::vector<EntryResult> entryResults;

        void Complete(Status status, std::vector<EntryResult> results = {})
        {
            failure = std::move(status);
            entryResults = std::move(results);
            state = RequestState::COMPLETED;
        }

        bool IsExposed() const noexcept
        {
            return state == RequestState::TRANSMITTING || state == RequestState::INFLIGHT;
        }
    };

    void QueueCompletion(Request request, Status status,
                         std::vector<EntryResult> entryResults = {});
    void ReleaseReplySlot(RequestRecord& request);
    void RetireRequest(RequestId requestId);
    void FinalizeRequests(TimePoint now);
    void ExpirePendingRequests(TimePoint now);
    void DispatchPendingRequests();
    void FlushCompletions();
    void StartRequest(Request request);

    void Handle(TransmitCompleted event, TimePoint);
    void Handle(ConnectCompleted event, TimePoint now);
    void Handle(FenceCompleted event, TimePoint now);
    void Handle(ReplyObserved event, TimePoint now);

    void TryFence(TimePoint now);
    void TryConnect(TimePoint now);
    Status EncodeRequest(const ReplySlot& replySlot, OpType op, const std::vector<IoEntry>& entries,
                         std::vector<std::uint8_t>& payload);

    Config config_;
    NodeDependencies dependencies_;
    DramPool::ProtocolManager protocol_;
    NodeState state_{NodeState::DISCONNECTED};
    ConnectionEpoch epoch_{1};
    std::unordered_map<RequestId, RequestRecord> activeRequests_;
    std::vector<RequestCompleted> completionBatch_;
    TimePoint nextActionAt_{TimePoint::min()};

    std::deque<Request> pendingRequests_;
    TimePoint pendingCheckAt_{TimePoint::max()};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_NODE_ACTOR_H
