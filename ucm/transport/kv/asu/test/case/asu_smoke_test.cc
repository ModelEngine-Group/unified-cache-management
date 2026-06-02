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
using UC::ASU::test::StubCreateConnection;
using UC::ASU::test::StubDeleteConnections;

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
    auto client = CreateAsuClient(
        []() -> std::unique_ptr<AsuTransport> { return std::make_unique<AsuTransportImpl>(); });
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

    ConnectionManager connMgr;
    connMgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(connMgr.AddGroup(MakeEndpoint("10.0.0.2"), 8).ok());
    connMgr.StartRecoverLoop();

    auto* firstChannel = connMgr.SelectConnection();
    ASSERT_NE(firstChannel, nullptr);
    firstChannel->ReleaseInflight();
    auto* group = firstChannel->GetGroup();
    const std::size_t initialCount = group->GetChannels().size();
    EXPECT_EQ(initialCount, 8u);

    auto* drainChannel = connMgr.SelectConnection();
    ASSERT_NE(drainChannel, nullptr);
    drainChannel->ReleaseInflight();
    auto drainChannelId = drainChannel->GetChannelId();

    constexpr int kThreads = 16;
    std::atomic<int> asyncOk{0};
    std::atomic<int> syncOk{0};

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i]() {
            TaskId taskId{kInvalidTaskId};
            Status status;
            switch (i % 8) {
                case 0: {
                    auto keys = MakeKeys(2);
                    QueryOptions options;
                    options.timeoutMs = 5000;
                    QueryResult result;
                    status = transport->Query(keys, options, result);
                    if (status.ok()) { syncOk.fetch_add(1); }
                    return;
                }
                case 1: {
                    auto keys = MakeKeys(2);
                    QueryOptions options;
                    options.timeoutMs = 5000;
                    status = transport->QueryAsync(keys, options, taskId);
                    break;
                }
                case 2: {
                    auto entries = MakeKVEntries(2);
                    status = transport->LoadAsync(entries, taskId);
                    break;
                }
                case 3: {
                    auto entries = MakeKVEntries(2);
                    status = transport->StoreAsync(entries, taskId);
                    break;
                }
                case 4: {
                    auto keys = MakeKeys(2);
                    status = transport->DeleteAsync(keys, taskId);
                    break;
                }
                case 5: transport->Cancel(i + 1000); return;
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

            if (status.ok() && taskId != kInvalidTaskId) {
                TaskResult result;
                status = transport->Wait(taskId, 10000, result);
                if (status.ok() && result.status.ok()) { asyncOk.fetch_add(1); }
            }
        });
    }

    std::thread drainThread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        connMgr.ReportFailure(drainChannel);
        connMgr.ReportFailure(drainChannel);
    });

    for (auto& thread : threads) { thread.join(); }
    drainThread.join();

    EXPECT_GT(asyncOk.load() + syncOk.load(), 0);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool rebuildDone = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool allActive = true;
        for (const auto& channel : group->GetChannels()) {
            if (channel->GetState() != ChannelState::ACTIVE) {
                allActive = false;
                break;
            }
        }
        if (allActive && group->GetChannels().size() == initialCount) {
            rebuildDone = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(rebuildDone) << "rebuild should complete within timeout";

    const auto& afterChannels = group->GetChannels();
    EXPECT_EQ(afterChannels.size(), initialCount);

    bool drainedChannelFound = false;
    std::uint32_t maxId = 0;
    for (const auto& channel : afterChannels) {
        if (channel->GetChannelId() == drainChannelId) { drainedChannelFound = true; }
        maxId = std::max(maxId, channel->GetChannelId());
        EXPECT_EQ(channel->GetState(), ChannelState::ACTIVE);
    }
    EXPECT_FALSE(drainedChannelFound);
    EXPECT_GT(maxId, drainChannelId);

    auto* selected = connMgr.SelectConnection();
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->GetState(), ChannelState::ACTIVE);
    selected->ReleaseInflight();

    EXPECT_EQ(connMgr.TotalInflightCount(), 0);
    connMgr.Shutdown();
    transport->Shutdown();
}

TEST(AsuSmokeTest, SequentialChannelDrainAndRebuild)
{
    ConnectionManager connMgr;
    connMgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(connMgr.AddGroup(MakeEndpoint("10.0.0.1"), 6).ok());
    connMgr.StartRecoverLoop();

    auto* firstChannel = connMgr.SelectConnection();
    ASSERT_NE(firstChannel, nullptr);
    firstChannel->ReleaseInflight();
    auto* group = firstChannel->GetGroup();
    const std::size_t initialCount = group->GetChannels().size();

    for (int cycle = 0; cycle < 3; ++cycle) {
        auto* channel = connMgr.SelectConnection();
        ASSERT_NE(channel, nullptr);
        channel->ReleaseInflight();
        auto channelId = channel->GetChannelId();

        connMgr.ReportFailure(channel);
        connMgr.ReportFailure(channel);

        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        bool rebuildDone = false;
        while (std::chrono::steady_clock::now() < deadline) {
            bool allActive = true;
            for (const auto& current : group->GetChannels()) {
                if (current->GetState() != ChannelState::ACTIVE) {
                    allActive = false;
                    break;
                }
            }
            if (allActive && group->GetChannels().size() == initialCount) {
                rebuildDone = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        EXPECT_TRUE(rebuildDone) << "cycle " << cycle << ": rebuild should complete";
        EXPECT_EQ(group->GetChannels().size(), initialCount);

        bool found = false;
        for (const auto& current : group->GetChannels()) {
            if (current->GetChannelId() == channelId) {
                found = true;
                break;
            }
        }
        EXPECT_FALSE(found) << "cycle " << cycle << ": channel " << channelId
                            << " should be removed";
    }

    EXPECT_EQ(connMgr.TotalInflightCount(), 0);
    connMgr.Shutdown();
}

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

    ConnectionManager connMgr;
    connMgr.SetConnectionOps(StubCreateConnection, StubDeleteConnections);
    ASSERT_TRUE(connMgr.AddGroup(MakeEndpoint("10.0.0.2"), 16).ok());
    connMgr.StartRecoverLoop();

    auto* drainChannel0 = connMgr.SelectConnection();
    ASSERT_NE(drainChannel0, nullptr);
    drainChannel0->ReleaseInflight();
    auto* drainChannel1 = connMgr.SelectConnection();
    ASSERT_NE(drainChannel1, nullptr);
    drainChannel1->ReleaseInflight();
    auto* group = drainChannel0->GetGroup();
    const std::size_t initialCount = group->GetChannels().size();

    constexpr int kLoadThreads = 32;
    constexpr int kTasksPerThread = 5;
    std::atomic<int> totalOk{0};

    std::vector<std::thread> loadThreads;
    for (int thread = 0; thread < kLoadThreads; ++thread) {
        loadThreads.emplace_back([&]() {
            for (int i = 0; i < kTasksPerThread; ++i) {
                auto entries = MakeKVEntries(1);
                TaskId taskId{kInvalidTaskId};
                if (transport->LoadAsync(entries, taskId).ok()) {
                    TaskResult result;
                    if (transport->Wait(taskId, 5000, result).ok() && result.status.ok()) {
                        totalOk.fetch_add(1);
                    }
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    connMgr.ReportFailure(drainChannel0);
    connMgr.ReportFailure(drainChannel0);
    connMgr.ReportFailure(drainChannel1);
    connMgr.ReportFailure(drainChannel1);

    for (auto& thread : loadThreads) { thread.join(); }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool rebuildDone = false;
    while (std::chrono::steady_clock::now() < deadline) {
        bool allActive = true;
        for (const auto& channel : group->GetChannels()) {
            if (channel->GetState() != ChannelState::ACTIVE) {
                allActive = false;
                break;
            }
        }
        if (allActive && group->GetChannels().size() == initialCount) {
            rebuildDone = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    EXPECT_TRUE(rebuildDone) << "rebuild should complete within timeout";
    EXPECT_EQ(group->GetChannels().size(), initialCount);
    EXPECT_GT(totalOk.load(), 0) << "some tasks should succeed during drain";
    EXPECT_EQ(connMgr.TotalInflightCount(), 0);

    connMgr.Shutdown();
    transport->Shutdown();
}

}  // namespace UC::ASU
