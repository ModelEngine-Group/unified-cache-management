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
#include "asu_client/asu_client.h"

namespace UC::ASU {
namespace {

AsuClientConfig MakeClientConfig()
{
    AsuClientConfig config;
    config.client_id = "asu-smoke-client";
    config.default_wait_timeout_ms = 100;

    TransportConfig first;
    first.asu_name = "asu-smoke-0";
    first.asu_id = 1001;
    first.max_inflight_tasks = 64;
    first.query_timeout_ms = 100;

    TransportConfig second;
    second.asu_name = "asu-smoke-1";
    second.asu_id = 1002;
    second.max_inflight_tasks = 64;
    second.query_timeout_ms = 100;

    config.transport_configs = {first, second};
    return config;
}

std::vector<KVBuffer> MakeEntries(std::vector<std::uint8_t>& payload)
{
    payload.assign(4096, 7);
    MemoryRegion region;
    region.memory_type = MemoryType::HOST;
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

void ExpectCompleted(AsuClient& client, TaskId task_id, std::size_t entry_count)
{
    TaskResult wait_result;
    auto status = client.Wait(task_id, 500, wait_result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_TRUE(wait_result.status.ok()) << wait_result.status.message;
    ASSERT_EQ(wait_result.entry_status.size(), entry_count);
    for (const auto& entry_status : wait_result.entry_status) {
        EXPECT_TRUE(entry_status.ok()) << entry_status.message;
    }

    TaskResult check_result;
    status = client.Check(task_id, check_result);
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

    TaskId load_task_id{kInvalidTaskId};
    status = client->LoadAsync(entries, load_task_id);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(load_task_id, kInvalidTaskId);
    ExpectCompleted(*client, load_task_id, entries.size());

    TaskId store_task_id{kInvalidTaskId};
    status = client->StoreAsync(entries, store_task_id);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(store_task_id, kInvalidTaskId);
    ExpectCompleted(*client, store_task_id, entries.size());

    std::vector<CacheKey> keys{"alpha", "beta", "gamma", "delta"};
    QueryOptions query_options;
    query_options.timeout_ms = 500;
    QueryResult query_result;
    status = client->Query(keys, query_options, query_result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(query_result.exists.size(), keys.size());

    TaskId delete_task_id{kInvalidTaskId};
    status = client->DeleteAsync(keys, delete_task_id);
    ASSERT_EQ(status.code, StatusCode::UNSUPPORTED);
    ASSERT_EQ(delete_task_id, kInvalidTaskId);

    status = client->Shutdown();
    ASSERT_TRUE(status.ok()) << status.message;
}

}  // namespace UC::ASU
