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

enum class OpType : std::uint8_t {
    LOOKUP = 0,
    DUMP,
    LOAD,
};

enum class TaskState : std::uint8_t {
    PREPARING = 0,
    RUNNING,
    DRAINING_FAILURE,
    SUCCEEDED,
    FAILED,
};

enum class RequestState : std::uint8_t {
    PENDING = 0,
    SENDING,
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
    std::string key;
    std::uint32_t layer{0};
    BufferRef buffer;
    std::size_t originalIndex{0};
};

struct DramTask {
    OpType op{OpType::LOOKUP};
    std::vector<IoEntry> entries;
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

// Memory key
struct LayerKey {
    std::uint32_t layerId{0};
    std::uint32_t rkey{0};
};

struct MemoryKey {
    std::vector<LayerKey> layerKeys;
    std::uint32_t replyKey{0};
};

// Node
struct NodeEndpoint {
    NodeId nodeId{0};
    // Endpoint
};

struct NodeLimits {
    std::size_t maxPendingRequests{0};
    std::size_t maxInflightRequests{0};
    std::size_t maxBatchEntries{0};
};

// Store Config
struct TimeoutConfig {
    std::chrono::milliseconds lookup{100};
    std::chrono::milliseconds dump{100};
    std::chrono::milliseconds load{100};
};

struct RuntimeConfig {
    std::vector<NodeEndpoint> nodes;
    NodeLimits nodeLimits;
    TimeoutConfig timeouts;
    std::chrono::milliseconds reconnectBackoff{0};
    std::size_t actorThreadCount{1};
    std::size_t transportThreadCount{1};
    std::size_t eventQueueCapacity{1};
};

}  // namespace UC::Dram

#endif
