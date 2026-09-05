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
#include "connection_internal.h"
#include <algorithm>
#include "logger.h"

namespace kv {

// ─── ConnectionChannel ───

ConnectionChannel::ConnectionChannel(std::uint32_t id, ConnectionGroup* grp,
                                     TransProvider::ConnectionHandle handle,
                                     TransProvider* provider)
    : channelId(id), group(grp), handle_(handle), provider_(provider)
{
    state.store(ChannelState::ACTIVE, std::memory_order_release);
    inflightCount.store(0, std::memory_order_release);
    errorCount.store(0, std::memory_order_release);
}

ConnectionChannel::~ConnectionChannel()
{
    if (handle_ && provider_) {
        provider_->DeleteConnections({handle_});
        handle_ = nullptr;
    }
}

std::uint32_t ConnectionChannel::FetchAddErrorCount(std::uint32_t val)
{
    return errorCount.fetch_add(val, std::memory_order_relaxed);
}

std::uint32_t ConnectionChannel::GetErrorCount() const
{
    return errorCount.load(std::memory_order_relaxed);
}

void ConnectionChannel::ResetErrorCount() { errorCount.store(0, std::memory_order_relaxed); }

void ConnectionChannel::IncrementInflight()
{
    inflightCount.fetch_add(1, std::memory_order_acq_rel);
}

void ConnectionChannel::ReleaseInflight()
{
    auto current = inflightCount.load(std::memory_order_acquire);
    while (current > 0) {
        if (inflightCount.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {
            return;
        }
    }
}

bool ConnectionChannel::MarkForDrain()
{
    ChannelState expected = ChannelState::ACTIVE;
    if (!state.compare_exchange_strong(expected, ChannelState::DRAINING,
                                       std::memory_order_acq_rel)) {
        KV_DEBUG("ConnectionChannel::MarkForDrain CAS FAILED: current_state={} (expected ACTIVE=0)",
                 static_cast<int>(expected));
        return false;
    }
    KV_DEBUG("ConnectionChannel::MarkForDrain CAS OK: ch_id={} ACTIVE->DRAINING", channelId);
    return true;
}

// ─── ConnectionGroup ───

ConnectionGroup::ConnectionGroup(std::uint32_t id, const AsuEndpoint& ep,
                                 const ServerKvCapabilities& capabilities)
    : groupId(id), endpoint(ep), serverCapabilities(capabilities)
{
}

std::shared_ptr<ConnectionChannel> ConnectionGroup::AddChannel(ConnectionHandle handle,
                                                               TransProvider* provider)
{
    auto id = nextChannelId_.fetch_add(1, std::memory_order_relaxed);
    auto channel = std::make_shared<ConnectionChannel>(id, this, handle, provider);
    channels.push_back(channel);
    KV_DEBUG("ConnectionGroup::AddChannel groupId={} ch_id={} totalChannels={}", groupId,
             channel->GetChannelId(), channels.size());
    return channel;
}

void ConnectionGroup::RemoveChannel(ConnectionChannel* channel)
{
    auto it = std::find_if(
        channels.begin(), channels.end(),
        [channel](const std::shared_ptr<ConnectionChannel>& p) { return p.get() == channel; });
    if (it != channels.end()) {
        KV_DEBUG("ConnectionGroup::RemoveChannel groupId={} ch_id={} removed, totalChannels={}",
                 groupId, channel->GetChannelId(), channels.size() - 1);
        channels.erase(it);
    }
}

bool ConnectionGroup::HasActiveChannel() const
{
    for (auto& channel : channels) {
        if (channel->GetState() == ChannelState::ACTIVE) { return true; }
    }
    return false;
}

}  // namespace kv
