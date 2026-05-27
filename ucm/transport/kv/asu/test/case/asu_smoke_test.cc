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
#include <gtest/gtest.h>
#include "asu_client_impl.h"

namespace UC::ASU {
namespace {

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

    TransportConfig second;
    second.asuName = "asu-smoke-1";
    second.asuId = 1002;
    second.maxInflightTasks = 64;
    second.queryTimeoutMs = 100;

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
    auto client = CreateAsuClient();
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

}  // namespace UC::ASU
