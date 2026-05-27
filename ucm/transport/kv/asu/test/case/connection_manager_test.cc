#include "connection_manager.h"
#include <cstdint>
#include <gtest/gtest.h>
#include "connection_internal.h"
#include "test_helper.h"

namespace UC::ASU {
namespace {

using UC::ASU::test::MakeEndpoint;
using UC::ASU::test::StubCreateConnection;
using UC::ASU::test::StubDeleteConnections;

}  // namespace

// Purpose: Verify AddGroup creates channels and SelectConnection works with different routing
// policies.
TEST(ConnectionManagerTest, AddGroupAndSelectConnection)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.2"), 2).ok());

    // Verify Round Robin distributes across groups
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);
    auto* ch1 = mgr.SelectConnection();
    ASSERT_NE(ch1, nullptr);
    ch1->ReleaseInflight();
    auto* ch2 = mgr.SelectConnection();
    ASSERT_NE(ch2, nullptr);
    ch2->ReleaseInflight();

    // Verify Least Loaded balances inflight
    mgr.SetRoutingPolicy(RoutingPolicy::LEAST_LOADED);
    auto* ch3 = mgr.SelectConnection();
    ASSERT_NE(ch3, nullptr);
    EXPECT_EQ(ch3->GetInflightCount(), 1u);
    ch3->ReleaseInflight();

    mgr.Shutdown();
}

// Purpose: Verify ReportFailure triggers drain at threshold and RecoverLoop rebuilds channels.
TEST(ConnectionManagerTest, ReportFailureAndRecoverLoop)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto* channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();

    // Below threshold: stays ACTIVE
    mgr.ReportFailure(channel);
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);

    // At threshold: triggers DRAINING
    mgr.ReportFailure(channel);
    EXPECT_EQ(channel->GetState(), ChannelState::DRAINING);

    // RecoverLoop will see inflight is already 0 and complete drain
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (channel->GetGroup()->HasActiveChannel()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(channel->GetGroup()->HasActiveChannel());
    EXPECT_EQ(channel->GetGroup()->GetChannels().size(), 4u);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    mgr.Shutdown();
}

// Purpose: Verify Drain lifecycle: BeginDrain CAS, FinishDrain sets FAILED, HasActiveChannel logic.
TEST(ConnectionManagerTest, DrainLifecycle)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto* ch0 = mgr.SelectConnection();
    ch0->ReleaseInflight();
    const auto& channels = ch0->GetGroup()->GetChannels();
    auto* ch1 = channels[1].get();

    // BeginDrain CAS
    EXPECT_TRUE(ch0->BeginDrain());
    EXPECT_FALSE(ch0->BeginDrain());  // Second call fails
    EXPECT_EQ(ch0->GetState(), ChannelState::DRAINING);
    EXPECT_TRUE(ch0->GetGroup()->HasActiveChannel());  // ch1 is still active

    // FinishDrain
    ch0->FinishDrain();
    EXPECT_EQ(ch0->GetState(), ChannelState::FAILED);
    EXPECT_EQ(ch0->GetNativeQp(), nullptr);

    // Drain all
    ch1->BeginDrain();
    ch1->FinishDrain();
    EXPECT_FALSE(ch0->GetGroup()->HasActiveChannel());

    mgr.Shutdown();
}

// Purpose: Verify SelectConnection returns nullptr when no active channels exist.
TEST(ConnectionManagerTest, SelectConnection_NoActiveChannel)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto* ch0 = mgr.SelectConnection();
    auto* ch1 = mgr.SelectConnection();
    ch0->ReleaseInflight();
    ch1->ReleaseInflight();

    ch0->BeginDrain();
    ch0->FinishDrain();
    ch1->BeginDrain();
    ch1->FinishDrain();

    EXPECT_EQ(mgr.SelectConnection(), nullptr);
    mgr.Shutdown();
}

// Purpose: Verify Shutdown cleans up resources and prevents further selection.
TEST(ConnectionManagerTest, Shutdown_CleansUp)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    auto s = mgr.Shutdown();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(mgr.SelectConnection(), nullptr);
}

// Purpose: Verify RecoverLoop handles multiple simultaneous failures and recovers.
TEST(ConnectionManagerTest, RecoverLoop_MultipleFailures)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    std::vector<ConnectionChannel*> channels;
    for (int i = 0; i < 4; ++i) {
        auto* ch = mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);
        channels.push_back(ch);
        mgr.ReportFailure(ch);
        mgr.ReportFailure(ch);
        ch->ReleaseInflight();
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (channels[0]->GetGroup()->HasActiveChannel()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_TRUE(channels[0]->GetGroup()->HasActiveChannel());
    EXPECT_EQ(channels[0]->GetGroup()->GetChannels().size(), 4u);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    mgr.Shutdown();
}

}  // namespace UC::ASU
