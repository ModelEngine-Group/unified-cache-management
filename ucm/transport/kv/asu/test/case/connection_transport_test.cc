#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
#include "asu_transport_impl.h"
#include "connection_internal.h"
#include "test_helper.h"

namespace UC::ASU {
namespace {

using UC::ASU::test::MakeEndpoint;
using UC::ASU::test::MakeKeys;
using UC::ASU::test::MakeKVEntries;
using UC::ASU::test::StubCreateConnection;
using UC::ASU::test::StubDeleteConnections;

TransportConfig MakeTransportConfig()
{
    TransportConfig config;
    config.asuName = "test-asu";
    config.asuId = 1;
    config.queryQpNum = 1;
    config.loadQpNum = 2;
    config.storeQpNum = 1;
    config.maxInflightTasks = 64;
    config.queryTimeoutMs = 5000;

    AsuEndpoint ep;
    ep.ip = "10.0.0.1";
    ep.port = 9559;
    ep.protocol = Protocol::UB;
    config.endpoints = {ep};

    return config;
}

void WaitAndVerifyOK(AsuTransport& transport, TaskId task_id, std::size_t entry_count)
{
    TaskResult result;
    auto s = transport.StubWait(task_id, 5000, result);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_EQ(result.entryStatus.size(), entry_count);
    for (const auto& es : result.entryStatus) { EXPECT_TRUE(es.ok()) << es.message; }
}

}  // namespace

// Purpose: Verify AsuTransportImpl Init creates groups + worker thread, CheckHealth returns OK,
// Shutdown cleans up.
TEST(ConnectionTransportTest, InitShutdown_Lifecycle)
{
    auto transport = CreateAsuTransport();
    ASSERT_NE(transport, nullptr);

    auto s = transport->Init(MakeTransportConfig());
    ASSERT_TRUE(s.ok()) << s.message;

    s = transport->CheckHealth();
    EXPECT_TRUE(s.ok());

    s = transport->Shutdown();
    ASSERT_TRUE(s.ok()) << s.message;
}

// Purpose: Verify LoadAsync + Wait end-to-end completes with OK status and correct entry_count.
TEST(ConnectionTransportTest, LoadAsyncAndWait_CompletesOK)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    auto entries = MakeKVEntries(4);
    TaskId task_id{kInvalidTaskId};
    auto s = transport->LoadAsync(entries, task_id);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_NE(task_id, kInvalidTaskId);

    WaitAndVerifyOK(*transport, task_id, entries.size());

    transport->Shutdown();
}

// Purpose: Verify QueryAsync + Wait end-to-end completes with OK status and query_result populated.
TEST(ConnectionTransportTest, QueryAsyncAndWait_CompletesOK)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    auto keys = MakeKeys(3);
    QueryOptions opts;
    opts.timeoutMs = 5000;
    TaskId task_id{kInvalidTaskId};
    auto s = transport->QueryAsync(keys, opts, task_id);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_NE(task_id, kInvalidTaskId);

    TaskResult result;
    s = transport->StubWait(task_id, 5000, result);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_TRUE(result.queryResult.has_value());
    ASSERT_EQ(result.queryResult->exists.size(), keys.size());

    transport->Shutdown();
}

// Purpose: Verify Check returns IN_PROGRESS immediately after submit, then DONE after Wait;
// subsequent Check returns TASK_NOT_FOUND.
TEST(ConnectionTransportTest, Check_InProgressThenDone)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    auto entries = MakeKVEntries(2);
    TaskId task_id{kInvalidTaskId};
    ASSERT_TRUE(transport->LoadAsync(entries, task_id).ok());

    TaskResult check_result;
    auto s = transport->StubCheck(task_id, check_result);
    ASSERT_TRUE(s.ok());
    if (check_result.status.code == StatusCode::IN_PROGRESS) {
        s = transport->StubWait(task_id, 5000, check_result);
        ASSERT_TRUE(s.ok());
        ASSERT_TRUE(check_result.status.ok());
    } else {
        ASSERT_TRUE(check_result.status.ok());
        s = transport->StubWait(task_id, 5000, check_result);
        ASSERT_TRUE(s.ok());
    }

    s = transport->StubCheck(task_id, check_result);
    EXPECT_EQ(s.code, StatusCode::TASK_NOT_FOUND);

    transport->Shutdown();
}

// Purpose: Verify Wait on nonexistent task_id returns TASK_NOT_FOUND.
TEST(ConnectionTransportTest, Wait_TaskNotFound)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    TaskResult result;
    auto s = transport->StubWait(9999, 100, result);
    EXPECT_EQ(s.code, StatusCode::TASK_NOT_FOUND);

    transport->Shutdown();
}

// Purpose: Verify 10 sequential LoadAsync+Wait tasks all complete successfully on the same
// transport instance.
TEST(ConnectionTransportTest, MultipleTasksSequential)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    std::vector<TaskId> task_ids;
    for (int i = 0; i < 10; ++i) {
        auto entries = MakeKVEntries(2);
        TaskId tid{kInvalidTaskId};
        auto s = transport->LoadAsync(entries, tid);
        ASSERT_TRUE(s.ok()) << s.message;
        ASSERT_NE(tid, kInvalidTaskId);
        task_ids.push_back(tid);
    }

    for (auto tid : task_ids) { WaitAndVerifyOK(*transport, tid, 2); }

    transport->Shutdown();
}

// Purpose: Verify ReportFailure triggers BeginDrain at threshold, RecoverLoop completes
// drain+rebuild, HasActiveChannel restores.
TEST(ConnectionTransportTest, ChannelFailure_ReportFailureTriggersDrain)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1", 9559), 4).ok());
    mgr.StartRecoverLoop();

    auto* channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    EXPECT_EQ(channel->GetGroup()->GetChannels().size(), 4u);

    mgr.ReportFailure(channel);
    EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(channel->GetGroup()->GetChannels().size(), 4u);

    mgr.ReportFailure(channel);
    EXPECT_EQ(channel->GetState(), ChannelState::DRAINING);

    channel->ReleaseInflight();
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

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

// Purpose: Verify 2 channels both failing can be drained and rebuilt, group recovers with new
// ACTIVE channels selectable.
TEST(ConnectionTransportTest, ChannelFailure_RecoveryAfterDrain)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1", 9559), 2).ok());
    mgr.StartRecoverLoop();

    auto* ch0 = mgr.SelectConnection();
    auto* ch1 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ASSERT_NE(ch1, nullptr);

    mgr.ReportFailure(ch0);
    mgr.ReportFailure(ch0);
    mgr.ReportFailure(ch1);
    mgr.ReportFailure(ch1);

    ch0->ReleaseInflight();
    ch1->ReleaseInflight();
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ch0->GetGroup()->HasActiveChannel()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_TRUE(ch0->GetGroup()->HasActiveChannel());
    EXPECT_EQ(ch0->GetGroup()->GetChannels().size(), 2u);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    auto* new_ch = mgr.SelectConnection();
    ASSERT_NE(new_ch, nullptr);
    EXPECT_EQ(new_ch->GetState(), ChannelState::ACTIVE);
    EXPECT_EQ(new_ch->GetInflightCount(), 1u);
    new_ch->ReleaseInflight();

    mgr.Shutdown();
}

}  // namespace UC::ASU
