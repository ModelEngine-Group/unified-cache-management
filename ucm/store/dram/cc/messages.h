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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_MESSAGES_H
#define UNIFIEDCACHE_DRAM_STORE_CC_MESSAGES_H

#include <functional>
#include <string>
#include <variant>
#include <vector>
#include "status/status.h"
#include "types.h"

namespace UC::Dram {

struct RequestCompleted {
    TaskId taskId{};
    RequestId requestId{0};
    NodeId nodeId{0};
    Status status{Status::OK()};
    std::vector<EntryResult> entryResults;
};

struct Transmit {
    RequestToken token;
    std::vector<std::uint8_t> payload;
};

struct Connect {
    NodeId nodeId{0};
    LaneId laneId{kDefaultLaneId};
    ConnectionEpoch epoch{0};
    std::string transportManagerId;
};

struct FenceEpoch {
    NodeId nodeId{0};
    LaneId laneId{kDefaultLaneId};
    ConnectionEpoch epoch{0};
};

using TransportCommand = std::variant<Transmit, Connect, FenceEpoch>;

struct TransmitCompleted {
    RequestToken token;
    Status status{Status::OK()};
};

struct ConnectCompleted {
    NodeId nodeId{0};
    LaneId laneId{kDefaultLaneId};
    ConnectionEpoch epoch{0};
    Status status{Status::OK()};
};

struct FenceCompleted {
    NodeId nodeId{0};
    LaneId laneId{kDefaultLaneId};
    ConnectionEpoch epoch{0};
    Status status{Status::OK()};
};

struct ReplyObserved {
    RequestToken token;
    Status status{Status::OK()};
    std::vector<EntryResult> entryResults;
};

using NodeEvent = std::variant<TransmitCompleted, ConnectCompleted, FenceCompleted, ReplyObserved>;

// Shared fail-stop path for violations that make reliable event delivery or
// remote-memory safety impossible to prove.
[[noreturn]] void AbortDramStore(const Status& status) noexcept;

// Requests and commands are consumed only when the submitter returns Status::OK().
using RequestSubmitter = std::function<Status(Request&)>;
using TransportCommandSubmitter = std::function<Status(TransportCommand&)>;

// Completions and events are reliable runtime facts. terminal shutdown may discard them.
using TaskCompletionPublisher = std::function<void(std::vector<RequestCompleted>&)>;
using NodeEventPublisher = std::function<void(NodeId, NodeEvent)>;

using ReplySlotAcquirer =
    std::function<Expected<ReplySlot>(const RequestToken&, OpType, std::size_t)>;
using ReplySlotReleaser = std::function<Status(const RequestToken&, const ReplySlot&)>;

struct NodeDependencies {
    TaskCompletionPublisher publishCompletion;
    TransportCommandSubmitter submitTransport;
    ReplySlotAcquirer acquireReplySlot;
    ReplySlotReleaser releaseReplySlot;
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_MESSAGES_H
