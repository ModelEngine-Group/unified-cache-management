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
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
#include "connection_manager.h"
#include "transport_task_manager.h"

namespace UC::ASU {

enum class ChannelState : std::uint8_t {
    ACTIVE,
    DRAINING,
    FAILED,
};

class ConnectionChannel {
public:
    ConnectionChannel(std::uint32_t id, ConnectionGroup* grp, void* qp);

    std::uint32_t GetChannelId() const { return channelId; }
    ChannelState GetState() const { return state.load(std::memory_order_acquire); }
    std::uint32_t GetInflightCount() const { return inflightCount.load(std::memory_order_acquire); }
    ConnectionGroup* GetGroup() const { return group; }
    void* GetNativeQp() const { return nativeQp; }

    void SetDrainStartTime(std::uint64_t t);
    std::uint64_t GetDrainStartTime() const;
    std::uint32_t FetchAddErrorCount(std::uint32_t val);

    // Test helper to set state directly
    void SetState(ChannelState s) { state.store(s, std::memory_order_release); }
    void SetInflightCount(std::uint32_t c) { inflightCount.store(c, std::memory_order_release); }

    Status StubSend(TransportTaskContext* ctx);
    void IncrementInflight();
    void ReleaseInflight();
    bool BeginDrain();
    void FinishDrain();

private:
    // Hot path - cache line aligned
    alignas(64) std::atomic<std::uint32_t> inflightCount{0};
    alignas(64) std::atomic<ChannelState> state{ChannelState::ACTIVE};

    std::uint32_t channelId{0};
    ConnectionGroup* group{nullptr};
    void* nativeQp{nullptr};

    // Cold path
    std::atomic<std::uint64_t> inflightBytes{0};
    std::atomic<std::uint32_t> errorCount{0};
    std::atomic<std::uint64_t> lastErrorTime{0};
    std::atomic<std::uint64_t> drainStartTime{0};
};

class ConnectionGroup {
public:
    ConnectionGroup(std::uint32_t id, const AsuEndpoint& ep);

    std::uint32_t GetGroupId() const { return groupId; }
    const AsuEndpoint& GetEndpoint() const { return endpoint; }
    const std::vector<std::unique_ptr<ConnectionChannel>>& GetChannels() const { return channels; }

    ConnectionChannel* AddChannel(ConnectionHandle handle);
    void RemoveChannel(ConnectionChannel* channel);
    bool HasActiveChannel() const;

private:
    std::uint32_t groupId{0};
    AsuEndpoint endpoint;
    std::vector<std::unique_ptr<ConnectionChannel>> channels;
    std::atomic<std::uint32_t> nextChannelId_{0};
};

}  // namespace UC::ASU