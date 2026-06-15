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
#include "connection_manager.h"
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <vector>
#include "connection_internal.h"
#include "trans_provider.h"

namespace UC::ASU {
namespace {

static std::atomic<int> g_deleteCount{0};
static std::atomic<int> g_createCount{0};

class StubTransProvider : public TransProvider {
public:
    std::uint32_t forceReturnCount = 0;

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t qpNum,
                            uint32_t, std::vector<ConnectionHandle>& handles) override
    {
        g_createCount.fetch_add(1);
        handles.clear();
        std::uint32_t count = (forceReturnCount > 0) ? forceReturnCount : qpNum;
        for (std::uint32_t i = 0; i < count; ++i) {
            handles.push_back(reinterpret_cast<ConnectionHandle>(
                static_cast<uintptr_t>(g_createCount.load() * 100 + i + 1)));
        }
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        for (std::size_t i = 0; i < handles.size(); ++i) { g_deleteCount.fetch_add(1); }
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<TransProvider::SendIoBatch>&, uint32_t,
                             uint32_t) override
    {
        return {};
    }
    Status RegisterMemory(ConnectionHandle, const std::vector<RegisterMemoryDesc>& descs,
                          std::vector<MemHandle>& handles) override
    {
        handles.clear();
        handles.reserve(descs.size());
        for (std::size_t index = 0; index < descs.size(); ++index) {
            handles.push_back(reinterpret_cast<MemHandle>(static_cast<std::uintptr_t>(index) +
                                                          static_cast<std::uintptr_t>(1)));
        }
        return Status::OK();
    }
    std::vector<Status> UnregisterMemory(const std::vector<UnregisterMemoryDesc>&) override
    {
        return {};
    }
    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }
    std::vector<Status> FreeThread(const std::vector<ThreadHandle>&) override { return {}; }
    Status GetMemTokenId(MemHandle, uint32_t& tokenId) override
    {
        tokenId = 1;
        return Status::OK();
    }
};

AsuEndpoint MakeEndpoint(const std::string& ip = "10.0.0.1")
{
    AsuEndpoint ep;
    ep.ip = ip;
    ep.port = 16666;
    return ep;
}

}  // namespace

// ─── ConnectionChannel Tests ───

TEST(ConnectionChannelTest, InitialStateIsActiveWithZeroCounters)
{
    g_deleteCount = 0;
    StubTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint());
    auto handle = reinterpret_cast<ConnectionHandle>(static_cast<uintptr_t>(0x1234));
    ConnectionChannel ch(0, &group, handle, &provider);

    EXPECT_EQ(ch.GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(ch.GetInflightCount(), 0u);
    EXPECT_EQ(ch.GetChannelId(), 0u);
    EXPECT_EQ(ch.GetGroup(), &group);
    EXPECT_EQ(ch.GetConnection(), handle);
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
        StubTransProvider provider;
        ConnectionGroup group(0, MakeEndpoint());
        ConnectionChannel ch(0, &group, handle, &provider);
    }
    EXPECT_EQ(g_deleteCount.load(), 1);
}

TEST(ConnectionChannelTest, DestructorDoesNotCallDeleteFnWhenHandleIsNull)
{
    g_deleteCount = 0;
    {
        StubTransProvider provider;
        ConnectionGroup group(0, MakeEndpoint());
        ConnectionChannel ch(0, &group, nullptr, &provider);
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
    StubTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint());

    auto ch0 = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), &provider);
    auto ch1 = group.AddChannel(reinterpret_cast<ConnectionHandle>(2), &provider);
    auto ch2 = group.AddChannel(reinterpret_cast<ConnectionHandle>(3), &provider);

    EXPECT_EQ(group.GetChannels().size(), 3u);
    EXPECT_EQ(ch0->GetChannelId(), 0u);
    EXPECT_EQ(ch1->GetChannelId(), 1u);
    EXPECT_EQ(ch2->GetChannelId(), 2u);
}

TEST(ConnectionGroupTest, RemoveChannelRemovesCorrectChannel)
{
    StubTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint());

    auto ch0 = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), &provider);
    auto ch1 = group.AddChannel(reinterpret_cast<ConnectionHandle>(2), &provider);
    auto ch2 = group.AddChannel(reinterpret_cast<ConnectionHandle>(3), &provider);

    group.RemoveChannel(ch1.get());
    EXPECT_EQ(group.GetChannels().size(), 2u);
    EXPECT_EQ(group.GetChannels()[0]->GetChannelId(), 0u);
    EXPECT_EQ(group.GetChannels()[1]->GetChannelId(), 2u);
}

TEST(ConnectionGroupTest, HasActiveChannelReturnsCorrectValue)
{
    StubTransProvider provider;
    ConnectionGroup group(0, MakeEndpoint());

    EXPECT_FALSE(group.HasActiveChannel());

    auto ch = group.AddChannel(reinterpret_cast<ConnectionHandle>(1), &provider);
    EXPECT_TRUE(group.HasActiveChannel());

    ch->MarkForDrain();
    EXPECT_FALSE(group.HasActiveChannel());
}

// ─── ConnectionManager Tests ───

TEST(ConnectionManagerTest, AddGroupCreatesGroupWithChannels)
{
    g_createCount = 0;
    g_deleteCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);

    auto status = mgr.AddGroup(MakeEndpoint(), 3);
    EXPECT_TRUE(status.ok()) << status.message;

    EXPECT_EQ(mgr.TotalInflightCount(), 0);
}

TEST(ConnectionManagerTest, AddGroupFailsWhenCreateFnReturnsWrongCount)
{
    StubTransProvider provider;
    provider.forceReturnCount = 1;
    ConnectionManager mgr(provider, "", 5000);

    auto status = mgr.AddGroup(MakeEndpoint(), 3);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
}

TEST(ConnectionManagerTest, AddGroupFailsAfterShutdown)
{
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.Shutdown();

    auto status = mgr.AddGroup(MakeEndpoint(), 1);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);
}

TEST(ConnectionManagerTest, SelectConnectionRoundRobinCyclesThroughChannels)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
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
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
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
    StubTransProvider provider;
    provider.forceReturnCount = 0;
    ConnectionManager mgr(provider, "", 5000);

    auto ch = mgr.SelectConnection();
    EXPECT_EQ(ch, nullptr);
}

TEST(ConnectionManagerTest, SelectConnectionReturnsNullptrAfterShutdown)
{
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 3);
    mgr.Shutdown();

    auto ch = mgr.SelectConnection();
    EXPECT_EQ(ch, nullptr);
}

TEST(ConnectionManagerTest, ReportFailureBelowThresholdDoesNotDrain)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 1);

    auto ch = mgr.SelectConnection();
    ASSERT_NE(ch, nullptr);

    mgr.ReportFailure(ch);
    EXPECT_EQ(ch->GetState(), ChannelState::ACTIVE);
}

TEST(ConnectionManagerTest, ReportFailureAtThresholdMarksForDrain)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
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
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 3);

    auto status = mgr.Shutdown();
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);
}

TEST(ConnectionManagerTest, TotalInflightCountSumsCorrectly)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
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
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 2);

    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);
    auto rr1 = mgr.SelectConnection();
    auto rr2 = mgr.SelectConnection();
    ASSERT_NE(rr1, nullptr);
    ASSERT_NE(rr2, nullptr);
    EXPECT_NE(rr1->GetChannelId(), rr2->GetChannelId());

    mgr.SetRoutingPolicy(RoutingPolicy::LEAST_LOADED);
    auto ll1 = mgr.SelectConnection();
    ASSERT_NE(ll1, nullptr);
    EXPECT_EQ(ll1->GetInflightCount(), 2u);

    auto ll2 = mgr.SelectConnection();
    ASSERT_NE(ll2, nullptr);
    EXPECT_NE(ll1->GetChannelId(), ll2->GetChannelId());
    EXPECT_EQ(ll2->GetInflightCount(), 2u);
}

// ─── ConnectionManager Multi-Group Tests ───

TEST(ConnectionManagerTest, SelectConnectionRoundRobinAcrossMultipleGroups)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2);
    mgr.AddGroup(MakeEndpoint("10.0.0.2"), 2);

    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);

    std::set<std::uint32_t> selectedIds;
    for (int i = 0; i < 4; ++i) {
        auto ch = mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);
        selectedIds.insert(ch->GetChannelId());
    }
    EXPECT_EQ(selectedIds.size(), 2u);

    auto ch5 = mgr.SelectConnection();
    ASSERT_NE(ch5, nullptr);
}

TEST(ConnectionManagerTest, SelectConnectionReturnsNullptrWhenAllChannelsAtMaxInflight)
{
    g_createCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 2);
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);

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
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 3);

    auto status1 = mgr.Shutdown();
    EXPECT_TRUE(status1.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);

    auto status2 = mgr.Shutdown();
    EXPECT_TRUE(status2.ok());
    EXPECT_EQ(g_deleteCount.load(), 3);
}

TEST(ConnectionManagerTest, ReportFailureMultipleTimesDoesNotDuplicateInDrainList)
{
    g_createCount = 0;
    g_deleteCount = 0;
    StubTransProvider provider;
    ConnectionManager mgr(provider, "", 5000);
    mgr.AddGroup(MakeEndpoint(), 1);

    {
        auto ch = mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);

        for (int i = 0; i < 5; ++i) { mgr.ReportFailure(ch); }

        EXPECT_EQ(ch->GetState(), ChannelState::DRAINING);
    }

    mgr.Shutdown();
    EXPECT_EQ(g_deleteCount.load(), 1);
}

}  // namespace UC::ASU
