#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
#include "asu_transport_impl.h"
#include "connection_internal.h"
#include "connection_manager.h"
#include "test_helper.h"

namespace UC::ASU {
namespace {

using UC::ASU::test::MakeEndpoint;
using UC::ASU::test::MakeKeys;
using UC::ASU::test::MakeKVEntries;
using UC::ASU::test::TestCreateConnection;

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

TransportConfig MakeConcurrentTransportConfig()
{
    TransportConfig config;
    config.asuName = "concurrent-test";
    config.asuId = 1;
    config.queryQpNum = 2;
    config.loadQpNum = 4;
    config.storeQpNum = 2;
    config.maxInflightTasks = 256;
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
    auto s = transport.Wait(task_id, 5000, result);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_TRUE(result.status.ok()) << result.status.message;
    ASSERT_EQ(result.entryStatus.size(), entry_count);
    for (const auto& es : result.entryStatus) { EXPECT_TRUE(es.ok()) << es.message; }
}

}  // namespace

// ─── ConnectionManagerTest ───

// Purpose: Verify AddGroup creates channels and SelectConnection works with different routing
// policies.
TEST(ConnectionManagerTest, AddGroupAndSelectConnection)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.2"), 2).ok());

    // Verify Round Robin distributes across groups
    mgr.SetRoutingPolicy(RoutingPolicy::ROUND_ROBIN);
    auto ch1 = mgr.SelectConnection();
    ASSERT_NE(ch1, nullptr);
    ch1->ReleaseInflight();
    auto ch2 = mgr.SelectConnection();
    ASSERT_NE(ch2, nullptr);
    ch2->ReleaseInflight();

    // Verify Least Loaded balances inflight
    mgr.SetRoutingPolicy(RoutingPolicy::LEAST_LOADED);
    auto ch3 = mgr.SelectConnection();
    ASSERT_NE(ch3, nullptr);
    EXPECT_EQ(ch3->GetInflightCount(), 1u);
    ch3->ReleaseInflight();

    mgr.Shutdown();
}

// Purpose: Verify ReportFailure triggers drain at threshold and RecoverLoop rebuilds channels.
TEST(ConnectionManagerTest, ReportFailureAndRecoverLoop)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto channel = mgr.SelectConnection();
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

// Purpose: Verify Drain lifecycle: MarkForDrain CAS, state transitions, HasActiveChannel logic.
TEST(ConnectionManagerTest, DrainLifecycle)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto ch0 = mgr.SelectConnection();
    ch0->ReleaseInflight();
    const auto& channels = ch0->GetGroup()->GetChannels();
    auto ch1 = channels[1];

    // MarkForDrain CAS
    EXPECT_TRUE(ch0->MarkForDrain());
    EXPECT_FALSE(ch0->MarkForDrain());  // Second call fails
    EXPECT_EQ(ch0->GetState(), ChannelState::DRAINING);
    EXPECT_TRUE(ch0->GetGroup()->HasActiveChannel());  // ch1 is still active

    // Drain all
    ch0->SetState(ChannelState::FAILED);
    ch1->MarkForDrain();
    ch1->SetState(ChannelState::FAILED);
    EXPECT_FALSE(ch0->GetGroup()->HasActiveChannel());

    mgr.Shutdown();
}

// Purpose: Verify SelectConnection returns nullptr when no active channels exist.
TEST(ConnectionManagerTest, SelectConnection_NoActiveChannel)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 2).ok());

    auto ch0 = mgr.SelectConnection();
    auto ch1 = mgr.SelectConnection();
    ch0->ReleaseInflight();
    ch1->ReleaseInflight();

    ch0->MarkForDrain();
    ch0->SetState(ChannelState::FAILED);
    ch1->MarkForDrain();
    ch1->SetState(ChannelState::FAILED);

    EXPECT_EQ(mgr.SelectConnection(), nullptr);
    mgr.Shutdown();
}

// Purpose: Verify Shutdown cleans up resources and prevents further selection.
TEST(ConnectionManagerTest, Shutdown_CleansUp)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    auto s = mgr.Shutdown();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(mgr.SelectConnection(), nullptr);
}

// Purpose: Verify RecoverLoop handles multiple simultaneous failures and recovers.
TEST(ConnectionManagerTest, RecoverLoop_MultipleFailures)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    std::vector<std::shared_ptr<ConnectionChannel>> channels;
    for (int i = 0; i < 4; ++i) {
        auto ch = mgr.SelectConnection();
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

// ─── ConnectionTransportTest ───

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
    s = transport->Wait(task_id, 5000, result);
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
    auto s = transport->Check(task_id, check_result);
    ASSERT_TRUE(s.ok());
    if (check_result.status.code == StatusCode::IN_PROGRESS) {
        s = transport->Wait(task_id, 5000, check_result);
        ASSERT_TRUE(s.ok());
        ASSERT_TRUE(check_result.status.ok());
    } else {
        ASSERT_TRUE(check_result.status.ok());
        s = transport->Wait(task_id, 5000, check_result);
        ASSERT_TRUE(s.ok());
    }

    s = transport->Check(task_id, check_result);
    EXPECT_EQ(s.code, StatusCode::TASK_NOT_FOUND);

    transport->Shutdown();
}

// Purpose: Verify Wait on nonexistent task_id returns TASK_NOT_FOUND.
TEST(ConnectionTransportTest, Wait_TaskNotFound)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

    TaskResult result;
    auto s = transport->Wait(9999, 100, result);
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

// ─── ConnectionConcurrentTest ───

// Purpose: Verify concurrent SelectConnection does not corrupt inflight_count; after all threads
// release inflight, every channel must have inflight_count==0 and TotalInflightCount==0.
TEST(ConnectionConcurrentTest, ConcurrentSelectConnection_InflightConsistency)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    std::set<ConnectionChannel*> all_channels;
    while (all_channels.size() < 4) {
        auto channel = mgr.SelectConnection();
        ASSERT_NE(channel, nullptr);
        all_channels.insert(channel.get());
        channel->ReleaseInflight();
    }
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    constexpr int kThreads = 8;
    constexpr int kIterations = 100;
    std::mutex select_mu;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIterations; ++i) {
                std::shared_ptr<ConnectionChannel> channel;
                {
                    std::lock_guard<std::mutex> lock(select_mu);
                    channel = mgr.SelectConnection();
                }
                if (channel) { channel->ReleaseInflight(); }
            }
        });
    }
    for (auto& th : threads) { th.join(); }

    for (auto* channel : all_channels) { EXPECT_EQ(channel->GetInflightCount(), 0); }
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    mgr.Shutdown();
}

// Purpose: Verify concurrent ReportFailure on same channel triggers MarkForDrain CAS exactly once.
TEST(ConnectionConcurrentTest, ConcurrentReportFailure_MarkForDrainCAS)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    auto channel = mgr.SelectConnection();
    ASSERT_NE(channel, nullptr);
    channel->ReleaseInflight();

    constexpr int kThreads = 4;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 2; ++i) { mgr.ReportFailure(channel); }
        });
    }
    for (auto& th : threads) { th.join(); }

    EXPECT_EQ(channel->GetState(), ChannelState::DRAINING);
    EXPECT_GE(channel->FetchAddErrorCount(0), 8u);

    mgr.Shutdown();
}

// Purpose: Verify 50 concurrent SubmitAsync+Wait tasks all complete successfully.
TEST(ConnectionConcurrentTest, ConcurrentSubmitAndWait_MultipleTasks)
{
    auto transport = CreateAsuTransport();
    ASSERT_TRUE(transport->Init(MakeConcurrentTransportConfig()).ok());

    constexpr int kTasks = 50;
    std::atomic<int> completed{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < kTasks; ++i) {
        threads.emplace_back([&, i]() {
            TaskId tid{kInvalidTaskId};
            Status s;
            if (i % 3 == 0) {
                auto entries = MakeKVEntries(2);
                s = transport->LoadAsync(entries, tid);
            } else if (i % 3 == 1) {
                auto entries = MakeKVEntries(2);
                s = transport->StoreAsync(entries, tid);
            } else {
                auto keys = MakeKeys(2);
                QueryOptions opts;
                opts.timeoutMs = 5000;
                s = transport->QueryAsync(keys, opts, tid);
            }

            if (s.ok() && tid != kInvalidTaskId) {
                TaskResult result;
                s = transport->Wait(tid, 10000, result);
                if (s.ok() && result.status.ok()) { completed.fetch_add(1); }
            }
        });
    }
    for (auto& th : threads) { th.join(); }

    EXPECT_EQ(completed.load(), kTasks);

    auto verify_entries = MakeKVEntries(2);
    TaskId verify_tid{kInvalidTaskId};
    auto s = transport->LoadAsync(verify_entries, verify_tid);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_NE(verify_tid, kInvalidTaskId);

    TaskResult verify_result;
    s = transport->Wait(verify_tid, 10000, verify_result);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_TRUE(verify_result.status.ok()) << verify_result.status.message;

    transport->Shutdown();
}

// Purpose: Verify concurrent drain (ReportFailure) and SelectConnection do not deadlock or crash.
TEST(ConnectionConcurrentTest, ConcurrentDrainAndSelect)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto ch0 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ch0->ReleaseInflight();

    std::thread drain_thread([&]() {
        mgr.ReportFailure(ch0);
        mgr.ReportFailure(ch0);
    });

    std::atomic<int> select_ok{0};
    std::mutex select_mu;
    std::vector<std::thread> select_threads;
    for (int i = 0; i < 20; ++i) {
        select_threads.emplace_back([&]() {
            std::shared_ptr<ConnectionChannel> channel;
            {
                std::lock_guard<std::mutex> lock(select_mu);
                channel = mgr.SelectConnection();
            }
            if (channel) {
                select_ok.fetch_add(1);
                channel->ReleaseInflight();
            }
        });
    }

    drain_thread.join();
    for (auto& th : select_threads) { th.join(); }

    EXPECT_EQ(select_ok.load(), 20);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    mgr.Shutdown();
}

// Purpose: Verify 2 channels failing concurrently from separate threads are drained and rebuilt.
TEST(ConnectionConcurrentTest, ConcurrentChannelFailureAndRecovery)
{
    ConnectionManager mgr(TestCreateConnection);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto ch0 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ch0->ReleaseInflight();
    const auto& channels = ch0->GetGroup()->GetChannels();
    ASSERT_EQ(channels.size(), 4u);

    auto ch1 = channels[1];

    std::thread fail0([&]() {
        mgr.ReportFailure(ch0);
        mgr.ReportFailure(ch0);
    });
    std::thread fail1([&]() {
        mgr.ReportFailure(ch1);
        mgr.ReportFailure(ch1);
    });

    fail0.join();
    fail1.join();

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (ch0->GetGroup()->HasActiveChannel()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_TRUE(ch0->GetGroup()->HasActiveChannel());
    EXPECT_EQ(ch0->GetGroup()->GetChannels().size(), 4u);

    std::atomic<int> select_after{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 20; ++i) {
        threads.emplace_back([&]() {
            auto channel = mgr.SelectConnection();
            if (channel) {
                EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
                select_after.fetch_add(1);
                channel->ReleaseInflight();
            }
        });
    }
    for (auto& th : threads) { th.join(); }

    EXPECT_EQ(select_after.load(), 20);
    EXPECT_EQ(mgr.TotalInflightCount(), 0);

    mgr.Shutdown();
}

}  // namespace UC::ASU
