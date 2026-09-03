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
#include <functional>
#include <memory>
#include <vector>
#include "asu_transport.h"
#include "trans_provider.h"
#include "types.h"
#include "connection_manager.h"

namespace UC::ASU {

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
    void ReleaseInflight();
    bool MarkForDrain();

private:
    // inflightCount: Maintained by upper-level users, it records the number of requests currently
    // being processed on the channel
    alignas(64) std::atomic<std::uint32_t> inflightCount{0};
    alignas(64) std::atomic<ChannelState> state{ChannelState::ACTIVE};

    std::uint32_t channelId{0};
    ConnectionGroup* group{nullptr};
    TransProvider::ConnectionHandle handle_{nullptr};
    TransProvider* provider_{nullptr};

    // Cold path
    std::atomic<std::uint32_t> errorCount{0};
};

class ConnectionGroup {
public:
    ConnectionGroup(std::uint32_t id, const AsuEndpoint& ep,
                    const ServerKvCapabilities& capabilities = {});

    std::uint32_t GetGroupId() const { return groupId; }
    const AsuEndpoint& GetEndpoint() const { return endpoint; }
    const ServerKvCapabilities& GetServerCapabilities() const { return serverCapabilities; }
    const std::vector<std::shared_ptr<ConnectionChannel>>& GetChannels() const { return channels; }

    std::shared_ptr<ConnectionChannel> AddChannel(ConnectionHandle handle, TransProvider* provider);
    void RemoveChannel(ConnectionChannel* channel);
    bool HasActiveChannel() const;

private:
    std::uint32_t groupId{0};
    AsuEndpoint endpoint;
    ServerKvCapabilities serverCapabilities;
    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    std::atomic<std::uint32_t> nextChannelId_{0};
};

}  // namespace UC::ASU
