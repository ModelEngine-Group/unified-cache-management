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
                                     TransProvider* provider,
                                     const std::shared_ptr<ConnectionGroup>& lease)
    : channelId(id), group(grp), handle_(handle), provider_(provider), groupLease(lease)
{
    groupId = grp != nullptr ? grp->GetGroupId() : 0;
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
            // X=1 drain semantics: a draining channel whose inflight reached
            // zero completes the DRAINING -> FAILED transition.
            if (current - 1 == 0 && GetState() == ChannelState::DRAINING) {
                if (MarkFailed()) {
                    KV_DEBUG("ConnectionChannel drained to FAILED ch_id={} group_id={}",
                             channelId, groupId);
                }
            }
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

bool ConnectionChannel::MarkFailed()
{
    ChannelState expected = ChannelState::DRAINING;
    if (!state.compare_exchange_strong(expected, ChannelState::FAILED, std::memory_order_acq_rel)) {
        return false;
    }
    KV_DEBUG("ConnectionChannel::MarkFailed ch_id={} group_id={} DRAINING->FAILED", channelId,
             groupId);
    return true;
}

bool ConnectionChannel::Revive()
{
    ChannelState expected = ChannelState::FAILED;
    if (!state.compare_exchange_strong(expected, ChannelState::ACTIVE,
                                       std::memory_order_acq_rel)) {
        return false;
    }
    // Counters restart from zero for the revived channel; no reuse of stale
    // failure accounting.
    errorCount.store(0, std::memory_order_relaxed);
    inflightCount.store(0, std::memory_order_release);
    probeInFlight.store(0, std::memory_order_release);
    KV_INFO("ConnectionChannel::Revive ch_id={} group_id={} FAILED->ACTIVE (probe success)",
            channelId, groupId);
    return true;
}

bool ConnectionChannel::TryStartProbe()
{
    std::uint32_t expected = 0;
    return probeInFlight.compare_exchange_strong(expected, 1, std::memory_order_acq_rel);
}

void ConnectionChannel::FinishProbe() { probeInFlight.store(0, std::memory_order_release); }

void ConnectionChannel::MarkBusinessSubmit()
{
    lastBusinessSubmitNs.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count(),
        std::memory_order_release);
}

TransProvider::ConnectionHandle ConnectionChannel::DetachHandle()
{
    auto handle = handle_;
    handle_ = nullptr;
    return handle;
}

// ─── ConnectionGroup ───

ConnectionGroup::ConnectionGroup(std::uint32_t id, const NodeEndpoint& ep,
                                 const ServerKvCapabilities& capabilities, std::uint32_t qpNum)
    : groupId(id), endpoint(ep), serverCapabilities(capabilities), qpNum_(qpNum)
{
}

std::shared_ptr<ConnectionChannel> ConnectionGroup::AddChannel(ConnectionHandle handle,
                                                               TransProvider* provider)
{
    auto id = nextChannelId_.fetch_add(1, std::memory_order_relaxed);
    auto lease = self_.lock();
    auto channel = std::make_shared<ConnectionChannel>(id, this, handle, provider, lease);
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

void ConnectionGroup::ClearChannels()
{
    KV_DEBUG("ConnectionGroup::ClearChannels groupId={} droppedChannels={}", groupId,
             channels.size());
    channels.clear();
}

bool ConnectionGroup::HasActiveChannel() const
{
    for (auto& channel : channels) {
        if (channel->GetState() == ChannelState::ACTIVE) { return true; }
    }
    return false;
}

std::uint32_t ConnectionGroup::GetInflightCount() const
{
    std::uint32_t total = 0;
    for (auto& channel : channels) { total += channel->GetInflightCount(); }
    return total;
}

}  // namespace kv
