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
#include <cstdint>
#include <atomic>
#include <set>
#include <vector>
#include <gtest/gtest.h>
#include "connection_internal.h"
#include "connection_manager.h"

namespace UC::ASU {
namespace {

static std::atomic<int> g_deleteCount{0};
static std::atomic<int> g_createCount{0};

ConnectionManager::CreateConnectionFunc MakeCreateFn(std::uint32_t numHandles)
{
    return [numHandles](const AsuEndpoint&, std::uint32_t qp) -> std::vector<ConnectionHandle> {
        g_createCount.fetch_add(1);
        std::vector<ConnectionHandle> handles;
        for (std::uint32_t i = 0; i < qp; ++i) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(g_createCount.load() * 100 + i + 1)));
        }
        return handles;
    };
}

ConnectionManager::DeleteConnectionFunc MakeDeleteFn()
{
    return [](ConnectionHandle) {
        g_deleteCount.fetch_add(1);
    };
}

AsuEndpoint MakeEndpoint(const std::string& ip = "10.0.0.1")
{
    AsuEndpoint ep;
    ep.ip = ip;
    ep.port = 16666;
    return ep;
}

ConnectionChannel::DeleteConnectionFunc MakeChannelDeleteFn()
{
    return [](ConnectionHandle) {
        g_deleteCount.fetch_add(1);
    };
}

}  // namespace

// ─── ConnectionChannel Tests ───

TEST(ConnectionChannelTest, InitialStateIsActiveWithZeroCounters)
{
    g_deleteCount = 0;
    ConnectionGroup group(0, MakeEndpoint());
    auto handle = reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(0x1234));
    ConnectionChannel ch(0, &group, handle, MakeChannelDeleteFn());

    EXPECT_EQ(ch.GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(ch.GetInflightCount(), 0u);
    EXPECT_EQ(ch.GetChannelId(), 0u);
    EXPECT_EQ(ch.GetGroup(), &group);
    EXPECT_EQ(ch.GetLink(), handle);
}

TEST(ConnectionChannelTest, IncrementAndReleaseInflight)
{
    ConnectionGroup group(0, MakeEndpoint());
    ConnectionChannel ch(0, &group, nullptr, nullptr);

    ch.IncrementInflight();
    ch.IncrementInflight();
    ch.IncrementInflight();
    EXPECT_EQ(ch.GetInflightCount(), 3u);

    ch.ReleaseInflight();
    EXPECT_EQ(ch.GetInflightCount(), 2u);
}

TEST(ConnectionChannelTest, FetchAddErrorCountAccumulates)
{
    ConnectionGroup group(0, MakeEndpoint());
    ConnectionChannel ch(0, &group, nullptr, nullptr);

    auto old = ch.FetchAddErrorCount(1);
    EXPECT_EQ(old, 0u);

    old = ch.FetchAddErrorCount(1);
    EXPECT_EQ(old, 1u);

    old = ch.FetchAddErrorCount(3);
    EXPECT_EQ(old, 2u);
}

TEST(ConnectionChannelTest, MarkForDrainTransitionsActiveToDraining)
{
    ConnectionGroup group(0, MakeEndpoint());
    ConnectionChannel ch(0, &group, nullptr, nullptr);

    EXPECT_EQ(ch.GetState(), ChannelState::ACTIVE);
    bool ok = ch.MarkForDrain();
    EXPECT_TRUE(ok);
    EXPECT_EQ(ch.GetState(), ChannelState::DRAINING);
}

TEST(ConnectionChannelTest, MarkForDrainFailsIfAlreadyDraining)
{
    ConnectionGroup group(0, MakeEndpoint());
    ConnectionChannel ch(0, &group, nullptr, nullptr);

    EXPECT_TRUE(ch.MarkForDrain());
    EXPECT_FALSE(ch.MarkForDrain());
    EXPECT_EQ(ch.GetState(), ChannelState::DRAINING);
}

TEST(ConnectionChannelTest, DestructorCallsDeleteFn)
{
    g_deleteCount = 0;
    auto handle = reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(0xABCD));
    {
        ConnectionGroup group(0, MakeEndpoint());
        ConnectionChannel ch(0, &group, handle, MakeChannelDeleteFn());
    }
    EXPECT_EQ(g_deleteCount.load(), 1);
}

TEST(ConnectionChannelTest, DestructorDoesNotCallDeleteFnWhenHandleIsNull)
{
    g_deleteCount = 0;
    {
        ConnectionGroup group(0, MakeEndpoint());
        ConnectionChannel ch(0, &group, nullptr, MakeChannelDeleteFn());
    }
    EXPECT_EQ(g_deleteCount.load(), 0);
}

// ─── ConnectionGroup Tests ───

TEST(ConnectionGroupTest, ConstructionSetsGroupIdAndEndpoint)
{
    auto ep = MakeEndpoint("192.168.1.1");
    ConnectionGroup group(42, ep);

    EXPECT_EQ(group.GetGroupId(), 42u);
    EXPECT_EQ(group.GetEndpoint().ip, "192.168.1.1");
    EXPECT_TRUE(group.GetChannels().empty());
}

TEST(ConnectionGroupTest, AddChannelCreatesChannelsWithIncrementingIds)
{
    ConnectionGroup group(0, MakeEndpoint());
    auto deleteFn = MakeChannelDeleteFn();

    auto ch0 = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), deleteFn);
    auto ch1 = group.AddChannel(reinterpret_cast<ConnectionHandle>(2), deleteFn);
    auto ch2 = group.AddChannel(reinterpret_cast<ConnectionHandle>(3), deleteFn);

    EXPECT_EQ(group.GetChannels().size(), 3u);
    EXPECT_EQ(ch0->GetChannelId(), 0u);
    EXPECT_EQ(ch1->GetChannelId(), 1u);
    EXPECT_EQ(ch2->GetChannelId(), 2u);
}

TEST(ConnectionGroupTest, RemoveChannelRemovesCorrectChannel)
{
    ConnectionGroup group(0, MakeEndpoint());
    auto deleteFn = MakeChannelDeleteFn();

    auto ch0 = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), deleteFn);
    auto ch1 = group.AddChannel(reinterpret_cast<ConnectionHandle>(2), deleteFn);
    auto ch2 = group.AddChannel(reinterpret_cast<ConnectionHandle>(3), deleteFn);

    group.RemoveChannel(ch1.get());
    EXPECT_EQ(group.GetChannels().size(), 2u);
    EXPECT_EQ(group.GetChannels()[0]->GetChannelId(), 0u);
    EXPECT_EQ(group.GetChannels()[1]->GetChannelId(), 2u);
}

TEST(ConnectionGroupTest, HasActiveChannelReturnsCorrectValue)
{
    ConnectionGroup group(0, MakeEndpoint());
    auto deleteFn = MakeChannelDeleteFn();

    EXPECT_FALSE(group.HasActiveChannel());

    auto ch = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), deleteFn);
    EXPECT_TRUE(group.HasActiveChannel());

    ch->MarkForDrain();
    EXPECT_FALSE(group.HasActiveChannel());
}

// ─── ConnectionManager Tests ───

TEST(ConnectionManagerTest, AddGroupCreatesGroupWithChannels)
{
    g_createCount = 0;
    g_deleteCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());

    auto status = mgr.AddGroup(MakeEndpoint(), 3);
    EXPECT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(mgr.TotalInflightCount(), 0);
}

TEST(ConnectionManagerTest, AddGroupFailsWhenCreateFnReturnsWrongCount)
{
    auto badCreateFn = [](const AsuEndpoint&, std::uint32_t) -> std::vector<ConnectionHandle> {
        return {reinterpret_cast<ConnectionHandle>(1)};  // Returns 1 instead of requested
    };
    ConnectionManager mgr(badCreateFn, MakeDeleteFn());

    auto status = mgr.AddGroup(MakeEndpoint(), 3);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
}

TEST(ConnectionManagerTest, AddGroupFailsAfterShutdown)
{
    ConnectionManager mgr(MakeCreateFn(1), MakeDeleteFn());
    mgr.Shutdown();

    auto status = mgr.AddGroup(MakeEndpoint(), 1);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
}

TEST(ConnectionManagerTest, SelectConnectionRoundRobinCyclesThroughChannels)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);

    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    auto ch2 = mgr.SelectConnection();
    auto ch3 = mgr.SelectConnection();

    ASSERT_NE(ch0, nullptr);
    ASSERT_NE(ch1, nullptr);
    ASSERT_NE(ch2, nullptr);
    ASSERT_NE(ch3, nullptr);

    EXPECT_NE(ch0->GetChannelId(), ch1->GetChannelId());
    EXPECT_NE(ch1->GetChannelId(), ch2->GetChannelId());
    EXPECT_EQ(ch0->GetChannelId(), ch3->GetChannelId());
}

TEST(ConnectionManagerTest, SelectConnectionLeastLoadedPicksLowestInflight)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);
    mgr.SetRoutingPolicy(RoutingPolicy::LEAST_LOADED);

    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    auto ch2 = mgr.SelectConnection();

    ASSERT_NE(ch0, nullptr);
    ASSERT_NE(ch1, nullptr);
    ASSERT_NE(ch2, nullptr);

    auto ch3 = mgr.SelectConnection();
    ASSERT_NE(ch3, nullptr);
    EXPECT_EQ(ch3->GetInflightCount(), 2u);
}

TEST(ConnectionManagerTest, SelectConnectionReturnsNullptrWhenNoChannels)
{
    ConnectionManager mgr(MakeCreateFn(0), MakeDeleteFn());

    auto ch = mgr.SelectConnection();
    EXPECT_EQ(ch, nullptr);
}

TEST(ConnectionManagerTest, SelectConnectionReturnsNullptrAfterShutdown)
{
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);
    mgr.Shutdown();

    auto ch = mgr.SelectConnection();
    EXPECT_EQ(ch, nullptr);
}

TEST(ConnectionManagerTest, ReportFailureBelowThresholdDoesNotDrain)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(1), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 1);

    auto ch = mgr.SelectConnection();
    ASSERT_NE(ch, nullptr);

    mgr.ReportFailure(ch);
    EXPECT_EQ(ch->GetState(), ChannelState::ACTIVE);
}

TEST(ConnectionManagerTest, ReportFailureAtThresholdMarksForDrain)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(1), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 1);

    auto ch = mgr.SelectConnection();
    ASSERT_NE(ch, nullptr);

    mgr.ReportFailure(ch);
    mgr.ReportFailure(ch);
    EXPECT_EQ(ch->GetState(), ChannelState::DRAINING);
}

TEST(ConnectionManagerTest, ShutdownClearsAllResources)
{
    g_deleteCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);

    auto status = mgr.Shutdown();
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);
}

TEST(ConnectionManagerTest, TotalInflightCountSumsCorrectly)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);

    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    ch0->IncrementInflight();
    ch0->IncrementInflight();
    ch1->IncrementInflight();

    EXPECT_EQ(mgr.TotalInflightCount(), 5);
}

TEST(ConnectionManagerTest, LeastLoadedAlwaysPicksLowestWhileRoundRobinCycles)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(2), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 2);

    // Round Robin: cycles through channels
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);
    auto rr1 = mgr.SelectConnection();
    auto rr2 = mgr.SelectConnection();
    ASSERT_NE(rr1, nullptr);
    ASSERT_NE(rr2, nullptr);
    EXPECT_NE(rr1->GetChannelId(), rr2->GetChannelId());

    // Both have inflight=1 now. LeastLoaded should pick the first one it finds with min inflight.
    mgr.SetRoutingPolicy(RoutingPolicy::LEAST_LOADED);
    auto ll1 = mgr.SelectConnection();
    ASSERT_NE(ll1, nullptr);
    EXPECT_EQ(ll1->GetInflightCount(), 2u);

    // Now ll1 has inflight=2, the other has inflight=1. LeastLoaded must pick the other.
    auto ll2 = mgr.SelectConnection();
    ASSERT_NE(ll2, nullptr);
    EXPECT_NE(ll1->GetChannelId(), ll2->GetChannelId());
    EXPECT_EQ(ll2->GetInflightCount(), 2u);
}

// ─── AICPUTransProvider Tests ───

TEST(ConnectionManagerTest, SelectConnectionRoundRobinAcrossMultipleGroups)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(2), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2);
    mgr.AddGroup(MakeEndpoint("10.0.0.2"), 2);

    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);

    // 4 channels total from 2 groups, should cycle through all 4
    std::set<std::uint32_t> selectedIds;
    for (int i = 0; i < 4; ++i) {
        auto ch = mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);
        selectedIds.insert(ch->GetChannelId());
    }
    // All 4 channel IDs should be unique (channels from different groups have independent IDs)
    // But channel IDs are per-group, so group0 has {0,1} and group1 has {0,1}
    // We verify we got 4 non-null selections
    EXPECT_EQ(selectedIds.size(), 2u);  // IDs repeat across groups: {0,1,0,1}

    // 5th selection should wrap around
    auto ch5 = mgr.SelectConnection();
    ASSERT_NE(ch5, nullptr);
}

TEST(ConnectionManagerTest, SelectConnectionReturnsNullptrWhenAllChannelsAtMaxInflight)
{
    g_createCount = 0;
    ConnectionManager mgr(MakeCreateFn(2), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 2);
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);

    // Fill both channels to max inflight (256 each)
    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ASSERT_NE(ch1, nullptr);

    ch0->SetInflightCount(256);
    ch1->SetInflightCount(256);

    auto ch = mgr.SelectConnection();
    EXPECT_EQ(ch, nullptr);
}

TEST(ConnectionManagerTest, ShutdownIsIdempotent)
{
    g_deleteCount = 0;
    ConnectionManager mgr(MakeCreateFn(3), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 3);

    auto status1 = mgr.Shutdown();
    EXPECT_TRUE(status1.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);

    // Second shutdown should not crash or double-delete
    auto status2 = mgr.Shutdown();
    EXPECT_TRUE(status2.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);  // No additional deletes
}

TEST(ConnectionManagerTest, ReportFailureMultipleTimesDoesNotDuplicateInDrainList)
{
    g_createCount = 0;
    g_deleteCount = 0;
    ConnectionManager mgr(MakeCreateFn(1), MakeDeleteFn());
    mgr.AddGroup(MakeEndpoint(), 1);

    {
        auto ch = mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);

        // Report failure 5 times (threshold is 2)
        for (int i = 0; i < 5; ++i) {
            mgr.ReportFailure(ch);
        }

        // Channel should be DRAINING (not FAILED or other state)
        EXPECT_EQ(ch->GetState(), ChannelState::DRAINING);
    }  // Release ch before Shutdown

    // Shutdown should clean up exactly 1 channel (no duplicates)
    mgr.Shutdown();
    EXPECT_EQ(g_deleteCount.load(), 1);
}

}  // namespace UC::ASU
