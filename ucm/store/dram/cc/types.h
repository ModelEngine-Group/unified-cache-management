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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_TYPES_H
#define UNIFIEDCACHE_DRAM_STORE_CC_TYPES_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "pool/buffer_pool.h"
#include "type/types.h"

namespace UC::Dram {

using TaskId = Detail::TaskHandle;
using RequestId = std::uint64_t;
using NodeId = std::uint64_t;
using LaneId = std::uint32_t;
using ConnectionEpoch = std::uint64_t;
using ReplySlot = BufferPool::Slot;

inline constexpr LaneId kDefaultLaneId = 0;
inline constexpr RequestId kInvalidRequestId = 0;
inline constexpr ConnectionEpoch kInvalidConnectionEpoch = 0;
inline constexpr std::size_t kMaxProtocolBatchEntries = std::numeric_limits<std::uint16_t>::max();

enum class OpType : std::uint8_t {
    LOOKUP = 0,
    DUMP,
    LOAD,
};

enum class RequestState : std::uint8_t {
    PENDING = 0,
    TRANSMITTING,
    INFLIGHT,
    WAITING_FENCE,
};

enum class NodeState : std::uint8_t {
    DISCONNECTED = 0,
    CONNECTING,
    ACTIVE,
    FENCING,
    STOPPING,
    STOPPED,
};

struct BufferRef {
    std::uintptr_t address{0};
    std::uint64_t length{0};
};

struct IoEntry {
    Detail::BlockId blockId{};
    std::uint32_t shardId{0};
    BufferRef buffer;
    std::size_t originalIndex{0};
};

struct EntryResult {
    std::size_t originalIndex{0};
    bool found{false};
    std::int32_t code{0};
};

struct Request {
    TaskId taskId{};
    RequestId requestId{0};
    NodeId nodeId{0};
    OpType op{OpType::LOOKUP};
    std::vector<IoEntry> entries;
    std::chrono::steady_clock::time_point deadline;
};

struct RequestToken {
    NodeId nodeId{0};
    LaneId laneId{kDefaultLaneId};
    ConnectionEpoch epoch{0};
    RequestId requestId{0};

    friend bool operator==(const RequestToken& lhs, const RequestToken& rhs) noexcept
    {
        return lhs.nodeId == rhs.nodeId && lhs.laneId == rhs.laneId && lhs.epoch == rhs.epoch &&
               lhs.requestId == rhs.requestId;
    }
    friend bool operator!=(const RequestToken& lhs, const RequestToken& rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

struct NodeEndpoint {
    NodeId nodeId{0};
    std::string controlHost;
    std::uint16_t controlPort{0};
    std::string transportManagerId;
};

struct NodeLimits {
    std::size_t maxPendingRequests{0};
    std::size_t maxInflightRequests{0};
    std::size_t maxPendingEntries{0};
    std::size_t maxBatchEntries{0};
};

struct TimeoutConfig {
    std::chrono::milliseconds lookup{0};
    std::chrono::milliseconds dump{0};
    std::chrono::milliseconds load{0};
};

struct TaskLimits {
    std::size_t maxLiveTasks{1024};
    std::size_t maxLiveEntries{65536};
};

struct NodeSchedulerConfig {
    std::vector<NodeEndpoint> nodes;
    NodeLimits limits;
    std::chrono::milliseconds reconnectInterval{0};
    std::size_t runnerCount{1};
};

struct TransportRuntimeConfig {
    std::size_t workerCount{1};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_TYPES_H
