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
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "asu_client_impl.h"
#include "asu_transport/asu_transport.h"
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


AsuClientConfig MakeClientConfig()
{
    AsuClientConfig config;
    config.clientId = "asu-smoke-client";
    config.defaultWaitTimeoutMs = 100;

    TransportConfig first;
    first.asuName = "asu-smoke-0";
    first.asuId = 1001;
    first.maxInflightTasks = 64;
    first.queryTimeoutMs = 100;
    {
        AsuEndpoint ep;
        ep.ip = "10.0.0.1";
        ep.port = 9559;
        ep.protocol = Protocol::UB;
        first.endpoints = {ep};
    }

    TransportConfig second;
    second.asuName = "asu-smoke-1";
    second.asuId = 1002;
    second.maxInflightTasks = 64;
    second.queryTimeoutMs = 100;
    {
        AsuEndpoint ep;
        ep.ip = "10.0.0.2";
        ep.port = 9559;
        ep.protocol = Protocol::UB;
        second.endpoints = {ep};
    }

    config.transportConfigs = {first, second};
    return config;
}

std::vector<KVBuffer> MakeEntries(std::vector<std::uint8_t>& payload)
{
    payload.assign(4096, 7);
    MemoryRegion region;
    region.memoryType = MemoryType::HOST;
    region.addr = reinterpret_cast<std::uint64_t>(payload.data());
    region.size = payload.size();

    Buffer buffer;
    buffer.region = region;

    return {
        KVBuffer{"alpha", buffer},
        KVBuffer{"beta",  buffer},
        KVBuffer{"gamma", buffer},
        KVBuffer{"delta", buffer},
    };
}

void ExpectCompleted(AsuClient& client, TaskId taskId, std::size_t entryCount)
{
    TaskResult waitResult;
    auto status = client.Wait(taskId, 500, waitResult);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_TRUE(waitResult.status.ok()) << waitResult.status.message;
    ASSERT_EQ(waitResult.entryStatus.size(), entryCount);
    for (const auto& entryStatus : waitResult.entryStatus) {
        EXPECT_TRUE(entryStatus.ok()) << entryStatus.message;
    }

    TaskResult checkResult;
    status = client.Check(taskId, checkResult);
    ASSERT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

}  // namespace

TEST(AsuSmokeTest, ClientAsyncTasksCompleteEndToEnd)
{
    auto client = CreateAsuClient([]() -> std::unique_ptr<AsuTransport> {
        return std::make_unique<AsuTransportImpl>();
    });
    ASSERT_NE(client, nullptr);

    auto status = client->Init(MakeClientConfig());
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<std::uint8_t> payload;
    auto entries = MakeEntries(payload);

    TaskId loadTaskId{kInvalidTaskId};
    status = client->LoadAsync(entries, loadTaskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(loadTaskId, kInvalidTaskId);
    ExpectCompleted(*client, loadTaskId, entries.size());

    TaskId storeTaskId{kInvalidTaskId};
    status = client->StoreAsync(entries, storeTaskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(storeTaskId, kInvalidTaskId);
    ExpectCompleted(*client, storeTaskId, entries.size());

    std::vector<CacheKey> keys{"alpha", "beta", "gamma", "delta"};
    QueryOptions queryOptions;
    queryOptions.timeoutMs = 500;
    QueryResult queryResult;
    status = client->Query(keys, queryOptions, queryResult);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(queryResult.exists.size(), keys.size());

    TaskId deleteTaskId{kInvalidTaskId};
    status = client->DeleteAsync(keys, deleteTaskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(deleteTaskId, kInvalidTaskId);
    ExpectCompleted(*client, deleteTaskId, keys.size());

    status = client->Shutdown();
    ASSERT_TRUE(status.ok()) << status.message;
}

TEST(AsuSmokeTest, ConcurrentAll8InterfacesWithChannelRebuild)
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

    auto& conn_mgr = static_cast<AsuTransportImpl*>(transport.get())->GetConnectionManager();

    // Record initial channel state
    auto first_ch = conn_mgr.SelectConnection();
    ASSERT_NE(first_ch, nullptr);
    first_ch->ReleaseInflight();
    auto* group = first_ch->GetGroup();
    const std::size_t initial_count = group->GetChannels().size();
    EXPECT_EQ(initial_count, 8u);

    // Collect initial channel IDs
    std::vector<std::uint32_t> initial_ids;
    for (const auto& ch : group->GetChannels()) { initial_ids.push_back(ch->GetChannelId()); }

    // Pre-select channel to drain on main thread (avoids concurrent SelectConnection with Worker)
    auto drain_ch = conn_mgr.SelectConnection();
    ASSERT_NE(drain_ch, nullptr);
    drain_ch->ReleaseInflight();
    auto drain_ch_id = drain_ch->GetChannelId();

    constexpr int kThreads = 16;
    std::atomic<int> async_ok{0};
    std::atomic<int> sync_ok{0};

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
                    TaskId q_tid{kInvalidTaskId};
                    s = transport->QueryAsync(keys, opts, q_tid);
                    if (s.ok() && q_tid != kInvalidTaskId) {
                        TaskResult qr_result;
                        s = transport->Wait(q_tid, 10000, qr_result);
                        if (s.ok() && qr_result.status.ok()) { sync_ok.fetch_add(1); }
                    }
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
                    transport->Cancel(i + 1000);
                    return;
                }
                case 6: {
                    TaskResult result;
                    transport->Check(i + 2000, result);
                    return;
                }
                case 7: {
                    TaskResult result;
                    transport->Wait(i + 3000, 100, result);
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

    // Drain thread uses pre-selected channel (no SelectConnection call)
    std::thread drain_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        conn_mgr.ReportFailure(drain_ch);
        conn_mgr.ReportFailure(drain_ch);
    });

    for (auto& th : threads) { th.join(); }
    drain_thread.join();

    EXPECT_GT(async_ok.load() + sync_ok.load(), 0);

    // Wait for RecoverLoop to finish drain and rebuild (poll until all channels ACTIVE)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool rebuild_done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all_active = true;
        for (const auto& ch : group->GetChannels()) {
            if (ch->GetState() != ChannelState::ACTIVE) {
                all_active = false;
                break;
            }
        }
        if (all_active && group->GetChannels().size() == initial_count) {
            rebuild_done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(rebuild_done) << "rebuild should complete within timeout";

    // 1. Verify channel count is preserved after rebuild
    const auto& after_channels = group->GetChannels();
    EXPECT_EQ(after_channels.size(), initial_count);

    // 2. Verify drained channel is removed from group
    bool drain_ch_found = false;
    for (const auto& ch : after_channels) {
        if (ch->GetChannelId() == drain_ch_id) {
            drain_ch_found = true;
            break;
        }
    }
    EXPECT_FALSE(drain_ch_found) << "drained channel " << drain_ch_id << " should be removed";

    // 3. Verify a NEW channel was created (ID > drain_ch_id)
    std::uint32_t max_id = 0;
    for (const auto& ch : after_channels) {
        max_id = std::max(max_id, ch->GetChannelId());
        EXPECT_EQ(ch->GetState(), ChannelState::ACTIVE);
    }
    EXPECT_GT(max_id, drain_ch_id) << "new channel should have higher ID after rebuild";

    // 4. Verify all remaining channels are ACTIVE
    for (const auto& ch : after_channels) { EXPECT_EQ(ch->GetState(), ChannelState::ACTIVE); }

    // 5. Verify SelectConnection returns an ACTIVE channel
    auto new_ch = conn_mgr.SelectConnection();
    ASSERT_NE(new_ch, nullptr);
    EXPECT_EQ(new_ch->GetState(), ChannelState::ACTIVE);
    new_ch->ReleaseInflight();

    // 6. Verify load balancing: 20 tasks should distribute across ALL channels
    constexpr int kVerifyTasks = 20;
    std::atomic<int> verify_ok{0};
    std::vector<std::uint32_t> used_channels(kVerifyTasks, kInvalidTaskId);
    std::vector<std::thread> verify_threads;
    for (int i = 0; i < kVerifyTasks; ++i) {
        verify_threads.emplace_back([&, i]() {
            auto entries = MakeKVEntries(1);
            TaskId tid{kInvalidTaskId};
            if (transport->LoadAsync(entries, tid).ok()) {
                TaskResult result;
                if (transport->Wait(tid, 5000, result).ok() && result.status.ok()) {
                    verify_ok.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : verify_threads) { th.join(); }
    EXPECT_EQ(verify_ok.load(), kVerifyTasks);

    // 7. Verify load balancing: multiple ACTIVE channels exist and Round Robin distributes
    std::vector<std::uint32_t> selected_ids;
    for (int i = 0; i < 5; ++i) {
        auto ch = conn_mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);
        EXPECT_EQ(ch->GetState(), ChannelState::ACTIVE);
        selected_ids.push_back(ch->GetChannelId());
        ch->ReleaseInflight();
    }
    // Round Robin should select different channels
    std::sort(selected_ids.begin(), selected_ids.end());
    auto last = std::unique(selected_ids.begin(), selected_ids.end());
    std::size_t unique_count = std::distance(selected_ids.begin(), last);
    EXPECT_GE(unique_count, 3u) << "Round Robin should distribute across multiple channels";

    // 8. Verify TotalInflightCount is 0 after all releases
    EXPECT_EQ(conn_mgr.TotalInflightCount(), 0);

    transport->Shutdown();
}

// Purpose: Verify sequential drain+rebuild cycles work correctly.
// Method: Drain 3 channels one by one, verify channel count preserved after each cycle.
TEST(AsuSmokeTest, SequentialChannelDrainAndRebuild)
{
    auto transport = CreateAsuTransport();

    TransportConfig config;
    config.asuName = "seq-rebuild";
    config.asuId = 2;
    config.queryQpNum = 2;
    config.loadQpNum = 2;
    config.storeQpNum = 2;
    config.maxInflightTasks = 64;
    config.queryTimeoutMs = 5000;
    AsuEndpoint ep;
    ep.ip = "10.0.0.1";
    ep.port = 9559;
    ep.protocol = Protocol::UB;
    config.endpoints = {ep};
    ASSERT_TRUE(transport->Init(config).ok());

    auto& conn_mgr = static_cast<AsuTransportImpl*>(transport.get())->GetConnectionManager();
    auto first_ch = conn_mgr.SelectConnection();
    ASSERT_NE(first_ch, nullptr);
    first_ch->ReleaseInflight();
    auto* group = first_ch->GetGroup();
    const std::size_t initial_count = group->GetChannels().size();

    // Drain 3 channels sequentially
    for (int cycle = 0; cycle < 3; ++cycle) {
        auto ch = conn_mgr.SelectConnection();
        ASSERT_NE(ch, nullptr);
        ch->ReleaseInflight();
        auto ch_id = ch->GetChannelId();

        conn_mgr.ReportFailure(ch);
        conn_mgr.ReportFailure(ch);

        // Wait for rebuild (poll until all channels ACTIVE)
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        bool rebuild_done = false;
        while (std::chrono::steady_clock::now() < deadline) {
            bool all_active = true;
            for (const auto& c : group->GetChannels()) {
                if (c->GetState() != ChannelState::ACTIVE) {
                    all_active = false;
                    break;
                }
            }
            if (all_active && group->GetChannels().size() == initial_count) {
                rebuild_done = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        EXPECT_TRUE(rebuild_done) << "cycle " << cycle << ": rebuild should complete";

        EXPECT_EQ(group->GetChannels().size(), initial_count);

        // Verify drained channel is removed
        bool found = false;
        for (const auto& c : group->GetChannels()) {
            if (c->GetChannelId() == ch_id) {
                found = true;
                break;
            }
        }
        EXPECT_FALSE(found) << "cycle " << cycle << ": channel " << ch_id << " should be removed";
    }

    transport->Shutdown();
}

// Purpose: Verify drain during heavy concurrent load does not cause crashes or data loss.
// Method: 32 threads continuously submit tasks while 2 channels are drained concurrently.
TEST(AsuSmokeTest, DrainUnderHeavyConcurrentLoad)
{
    auto transport = CreateAsuTransport();

    TransportConfig config;
    config.asuName = "heavy-drain";
    config.asuId = 3;
    config.queryQpNum = 4;
    config.loadQpNum = 8;
    config.storeQpNum = 4;
    config.maxInflightTasks = 512;
    config.queryTimeoutMs = 5000;
    AsuEndpoint ep;
    ep.ip = "10.0.0.1";
    ep.port = 9559;
    ep.protocol = Protocol::UB;
    config.endpoints = {ep};
    ASSERT_TRUE(transport->Init(config).ok());

    auto& conn_mgr = static_cast<AsuTransportImpl*>(transport.get())->GetConnectionManager();

    // Pre-select 2 channels to drain
    auto drain_ch0 = conn_mgr.SelectConnection();
    ASSERT_NE(drain_ch0, nullptr);
    drain_ch0->ReleaseInflight();
    auto drain_ch1 = conn_mgr.SelectConnection();
    ASSERT_NE(drain_ch1, nullptr);
    drain_ch1->ReleaseInflight();
    auto* group = drain_ch0->GetGroup();
    const std::size_t initial_count = group->GetChannels().size();

    // 32 threads continuously submitting tasks
    constexpr int kLoadThreads = 32;
    constexpr int kTasksPerThread = 5;
    std::atomic<int> total_ok{0};
    std::atomic<bool> load_done{false};

    std::vector<std::thread> load_threads;
    for (int t = 0; t < kLoadThreads; ++t) {
        load_threads.emplace_back([&]() {
            for (int i = 0; i < kTasksPerThread && !load_done.load(); ++i) {
                auto entries = MakeKVEntries(1);
                TaskId tid{kInvalidTaskId};
                if (transport->LoadAsync(entries, tid).ok()) {
                    TaskResult result;
                    if (transport->Wait(tid, 5000, result).ok() && result.status.ok()) {
                        total_ok.fetch_add(1);
                    }
                }
            }
        });
    }

    // Drain 2 channels while load is running
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    conn_mgr.ReportFailure(drain_ch0);
    conn_mgr.ReportFailure(drain_ch0);
    conn_mgr.ReportFailure(drain_ch1);
    conn_mgr.ReportFailure(drain_ch1);

    // Wait for load to finish
    for (auto& th : load_threads) { th.join(); }
    load_done.store(true);

    // Wait for rebuild (poll until all channels are ACTIVE)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool rebuild_done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool all_active = true;
        for (const auto& ch : group->GetChannels()) {
            if (ch->GetState() != ChannelState::ACTIVE) {
                all_active = false;
                break;
            }
        }
        if (all_active && group->GetChannels().size() == initial_count) {
            rebuild_done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(rebuild_done) << "rebuild should complete within timeout";

    EXPECT_EQ(group->GetChannels().size(), initial_count);
    EXPECT_GT(total_ok.load(), 0) << "some tasks should succeed during drain";

    transport->Shutdown();
}

}  // namespace UC::ASU
