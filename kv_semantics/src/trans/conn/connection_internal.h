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
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "connection_manager.h"
#include "kv_transport.h"
#include "kv_types.h"
#include "trans_provider.h"

namespace kv {

enum class ChannelState : std::uint8_t {
    ACTIVE,
    DRAINING,
    FAILED,
};

class ConnectionChannel {
public:
    ConnectionChannel(std::uint32_t id, ConnectionGroup* grp,
                      TransProvider::ConnectionHandle handle, TransProvider* provider);
    ~ConnectionChannel();

    std::uint32_t GetChannelId() const { return channelId; }
    std::uint32_t GetGroupId() const { return groupId; }
    ChannelState GetState() const { return state.load(std::memory_order_acquire); }
    std::uint32_t GetInflightCount() const { return inflightCount.load(std::memory_order_acquire); }
    ConnectionGroup* GetGroup() const { return group; }
    TransProvider::ConnectionHandle GetConnection() const { return handle_; }

    std::uint32_t FetchAddErrorCount(std::uint32_t val);
    std::uint32_t GetErrorCount() const;
    void ResetErrorCount();

    // Test helper to set state directly
    void SetState(ChannelState s) { state.store(s, std::memory_order_release); }
    void SetInflightCount(std::uint32_t c) { inflightCount.store(c, std::memory_order_release); }

    void IncrementInflight();
    // Decrements inflight; a DRAINING channel whose inflight reaches zero
    // transitions to FAILED (X=1 drain semantics).
    void ReleaseInflight();
    // ACTIVE -> DRAINING (reached X consecutive link failures).
    bool MarkForDrain();
    // DRAINING -> FAILED once inflight drains (also used by forced teardown).
    bool MarkFailed();
    // Returns and clears the provider handle so whole-group teardown can call
    // DeleteConnections exactly once (destructor skips detached handles).
    TransProvider::ConnectionHandle DetachHandle();

private:
    // inflightCount: Maintained by upper-level users, it records the number of requests currently
    // being processed on the channel
    alignas(64) std::atomic<std::uint32_t> inflightCount{0};
    alignas(64) std::atomic<ChannelState> state{ChannelState::ACTIVE};

    std::uint32_t channelId{0};
    std::uint32_t groupId{0};
    ConnectionGroup* group{nullptr};
    TransProvider::ConnectionHandle handle_{nullptr};
    TransProvider* provider_{nullptr};

    // Cold path
    std::atomic<std::uint32_t> errorCount{0};
    // (probe/flow-control fields are excluded on the core branch)
};

class ConnectionGroup {
public:
    ConnectionGroup(std::uint32_t id, const NodeEndpoint& ep,
                    const ServerKvCapabilities& capabilities = {}, std::uint32_t qpNum = 1);

    std::uint32_t GetGroupId() const { return groupId; }
    const NodeEndpoint& GetEndpoint() const { return endpoint; }
    const ServerKvCapabilities& GetServerCapabilities() const { return serverCapabilities; }
    const std::vector<std::shared_ptr<ConnectionChannel>>& GetChannels() const { return channels; }
    std::uint32_t GetQpNum() const { return qpNum_; }

    std::shared_ptr<ConnectionChannel> AddChannel(ConnectionHandle handle, TransProvider* provider);
    void RemoveChannel(ConnectionChannel* channel);
    // Drops every channel reference (whole-group replacement); callers must
    // already have detached the handles they own.
    void ClearChannels();
    bool HasActiveChannel() const;
    // Group inflight = sum of channel inflight counts.
    std::uint32_t GetInflightCount() const;

private:
    std::uint32_t groupId{0};
    NodeEndpoint endpoint;
    ServerKvCapabilities serverCapabilities;
    std::uint32_t qpNum_{1};
    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    std::atomic<std::uint32_t> nextChannelId_{0};
};

}  // namespace kv
