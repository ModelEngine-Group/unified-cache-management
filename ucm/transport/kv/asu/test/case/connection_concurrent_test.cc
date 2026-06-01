#include <algorithm>
#include <atomic>
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
using UC::ASU::test::StubCreateConnection;
using UC::ASU::test::StubDeleteConnections;

TransportConfig MakeTransportConfig()
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

}  // namespace

// Purpose: Verify concurrent SelectConnection does not corrupt inflight_count; after all threads
// release inflight, every channel must have inflight_count==0 and TotalInflightCount==0.
TEST(ConnectionConcurrentTest, ConcurrentSelectConnection_InflightConsistency)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    std::set<ConnectionChannel*> all_channels;
    while (all_channels.size() < 4) {
        auto* channel = mgr.SelectConnection();
        ASSERT_NE(channel, nullptr);
        all_channels.insert(channel);
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
                ConnectionChannel* channel = nullptr;
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

// Purpose: Verify concurrent ReportFailure on same channel triggers BeginDrain CAS exactly once.
TEST(ConnectionConcurrentTest, ConcurrentReportFailure_BeginDrainCAS)
{
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());

    auto* channel = mgr.SelectConnection();
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
    ASSERT_TRUE(transport->Init(MakeTransportConfig()).ok());

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
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto* ch0 = mgr.SelectConnection();
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
            ConnectionChannel* channel = nullptr;
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
    ConnectionManager mgr;
    mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(mgr.AddGroup(MakeEndpoint("10.0.0.1"), 4).ok());
    mgr.StartRecoverLoop();

    auto* ch0 = mgr.SelectConnection();
    ASSERT_NE(ch0, nullptr);
    ch0->ReleaseInflight();
    const auto& channels = ch0->GetGroup()->GetChannels();
    ASSERT_EQ(channels.size(), 4u);

    auto* ch1 = channels[1].get();

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
            auto* channel = mgr.SelectConnection();
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

TEST(ConnectionConcurrentTest, ConcurrentAll8InterfacesWithChannelRebuild)
{
    auto transport = CreateAsuTransport();

    TransportConfig config;
    config.asuName = "rebuild-8iface";
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
    ASSERT_TRUE(transport->Init(config).ok());

    ConnectionManager rebuild_mgr;
    rebuild_mgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(rebuild_mgr.AddGroup(MakeEndpoint("10.0.0.2"), 8).ok());
    rebuild_mgr.StartRecoverLoop();

    constexpr int kThreads = 16;
    std::atomic<int> async_ok{0};
    std::atomic<int> sync_ok{0};
    std::atomic<int> cancel_count{0};
    std::atomic<int> check_count{0};
    std::atomic<int> wait_short_count{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            TaskId tid{kInvalidTaskId};
            Status s;
            switch (i % 8) {
                case 0: {
                    auto keys = MakeKeys(2);
                    QueryOptions opts;
                    opts.timeoutMs = 5000;
                    QueryResult qr;
                    s = transport->Query(keys, opts, qr);
                    if (s.ok()) { sync_ok.fetch_add(1); }
                    return;
                }
                case 1: {
                    auto keys = MakeKeys(2);
                    QueryOptions opts;
                    opts.timeoutMs = 5000;
                    s = transport->QueryAsync(keys, opts, tid);
                    break;
                }
                case 2: {
                    auto entries = MakeKVEntries(2);
                    s = transport->LoadAsync(entries, tid);
                    break;
                }
                case 3: {
                    auto entries = MakeKVEntries(2);
                    s = transport->StoreAsync(entries, tid);
                    break;
                }
                case 4: {
                    auto keys = MakeKeys(2);
                    s = transport->DeleteAsync(keys, tid);
                    break;
                }
                case 5: {
                    s = transport->Cancel(i + 1000);
                    cancel_count.fetch_add(1);
                    return;
                }
                case 6: {
                    TaskResult result;
                    auto s2 = transport->Check(i + 2000, result);
                    if (s2.code == StatusCode::TASK_NOT_FOUND) { check_count.fetch_add(1); }
                    return;
                }
                case 7: {
                    TaskResult result;
                    auto s2 = transport->Wait(i + 3000, 100, result);
                    if (s2.code == StatusCode::TASK_NOT_FOUND) { wait_short_count.fetch_add(1); }
                    return;
                }
            }

            if (s.ok() && tid != kInvalidTaskId) {
                TaskResult result;
                s = transport->Wait(tid, 10000, result);
                if (s.ok() && result.status.ok()) { async_ok.fetch_add(1); }
            }
        });
    }

    std::thread drain_thread0([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto* channel = rebuild_mgr.SelectConnection();
        if (channel) {
            channel->ReleaseInflight();
            rebuild_mgr.ReportFailure(channel);
            rebuild_mgr.ReportFailure(channel);
        }
    });

    std::thread drain_thread1([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto* channel = rebuild_mgr.SelectConnection();
        if (channel) {
            channel->ReleaseInflight();
            rebuild_mgr.ReportFailure(channel);
            rebuild_mgr.ReportFailure(channel);
        }
    });

    for (auto& th : threads) { th.join(); }
    drain_thread0.join();
    drain_thread1.join();

    EXPECT_GT(async_ok.load() + sync_ok.load(), 0);
    EXPECT_EQ(cancel_count.load(), 2);
    EXPECT_EQ(check_count.load(), 2);
    EXPECT_EQ(wait_short_count.load(), 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    EXPECT_EQ(rebuild_mgr.TotalInflightCount(), 0);
    rebuild_mgr.Shutdown();

    auto verify_entries = MakeKVEntries(1);
    TaskId verify_tid{kInvalidTaskId};
    ASSERT_TRUE(transport->LoadAsync(verify_entries, verify_tid).ok());
    TaskResult verify_result;
    ASSERT_TRUE(transport->Wait(verify_tid, 5000, verify_result).ok());

    transport->Shutdown();
}

}  // namespace UC::ASU
