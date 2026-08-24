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

inline std::uint64_t SteadyNowUs()
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

inline std::uint64_t UnixNowUs()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

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
    TRANSMITTING = 0,
    INFLIGHT,
    WAITING_FENCE,
    COMPLETED,
};

inline const char* RequestStateToString(RequestState state) noexcept
{
    constexpr const char* names[] = {"TRANSMITTING", "INFLIGHT", "WAITING_FENCE", "COMPLETED"};
    return names[static_cast<std::uint8_t>(state)];
}

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

struct RequestTiming {
    std::uint64_t nodeQueuedUs{0};
    std::uint64_t nodeActorStartedUs{0};
    std::uint64_t replySlotAcquiredUs{0};
    std::uint64_t requestEncodedUs{0};
    std::uint64_t controlTransportSubmitStartedUs{0};
    std::uint64_t controlTransportSubmittedUs{0};
    std::uint64_t controlTransportCompletedUs{0};
    std::uint64_t replyObservedUs{0};
    std::uint64_t replyProcessedUs{0};
    std::uint64_t completedUs{0};
    std::uint64_t nodeQueuedTsUs{0};
    std::uint64_t controlTransportSubmittedTsUs{0};
    std::uint64_t controlTransportCompletedTsUs{0};
    std::uint64_t replyObservedTsUs{0};
    std::uint64_t completedTsUs{0};
};

struct Request {
    TaskId taskId{};
    RequestId requestId{0};
    NodeId nodeId{0};
    OpType op{OpType::LOOKUP};
    std::vector<IoEntry> entries;
    std::chrono::steady_clock::time_point deadline;
    RequestTiming timing;
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
    std::size_t maxInflightRequests{0};
    std::size_t maxBatchEntries{0};
};

struct TimeoutConfig {
    std::chrono::milliseconds lookup{0};
    std::chrono::milliseconds dump{0};
    std::chrono::milliseconds load{0};
};

struct NodeSchedulerConfig {
    std::vector<NodeEndpoint> nodes;
    NodeLimits limits;
    std::chrono::milliseconds reconnectInterval{0};
    std::size_t runnerCount{1};
    std::int32_t deviceId{0};
};

struct TransportRuntimeConfig {
    std::size_t workerCount{1};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_TYPES_H
