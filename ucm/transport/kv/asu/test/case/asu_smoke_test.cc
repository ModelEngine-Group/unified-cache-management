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
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string_view>
#include "asu_client_impl.h"

namespace UC::ASU {
namespace {

CacheKey MakeCacheKey(std::string_view text)
{
    CacheKey key{};
    const auto size = std::min(text.size(), key.size());
    if (size > 0) { std::memcpy(key.data(), text.data(), size); }
    return key;
}

class StubTransport : public AsuTransport {
public:
    Status Init(const TransportConfig& config) override
    {
        config_ = config;
        initialized_ = true;
        return Status::OK();
    }

    Status Init(const std::string&) override
    {
        return Status::Error(StatusCode::UNSUPPORTED, "stub transport config path unsupported");
    }

    Status Shutdown() override
    {
        initialized_ = false;
        return Status::OK();
    }

    Status CheckHealth() override { return Status::OK(); }

    Status Query(const std::vector<CacheKey>& keys, const QueryOptions&,
                 QueryResult& result) override
    {
        result.exists.assign(keys.size(), true);
        result.prefixHitKeys = 0;
        return Status::OK();
    }

    Status QueryAsync(const std::vector<CacheKey>&, const QueryOptions&, TaskId& taskId) override
    {
        taskId = 0;
        return Status::OK();
    }

    Status LoadAsync(const std::vector<KVBuffer>&, TaskId& taskId) override
    {
        taskId = nextTaskId_++;
        return Status::OK();
    }

    Status StoreAsync(const std::vector<KVBuffer>&, TaskId& taskId) override
    {
        taskId = nextTaskId_++;
        return Status::OK();
    }

    Status DeleteAsync(const std::vector<CacheKey>&, TaskId& taskId) override
    {
        taskId = nextTaskId_++;
        return Status::OK();
    }

    Status Cancel(TaskId) override { return Status::OK(); }

    Status Check(TaskId taskId, TaskResult& result) override
    {
        if (taskId == kInvalidTaskId) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found");
        }
        result.status = Status::OK();
        result.entryStatus.assign(1, Status::OK());
        result.queryResult.reset();
        return Status::OK();
    }

    Status Wait(TaskId taskId, std::uint64_t, TaskResult& result) override
    {
        return Check(taskId, result);
    }

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisterResult>& results) override
    {
        results.clear();
        for (std::size_t i = 0; i < regions.size(); ++i) {
            results.emplace_back(RegisterResult{Status::OK(), static_cast<MRHandle>(i + 1)});
        }
        return Status::OK();
    }

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                 std::vector<RegisterResult>& results) override
    {
        results.clear();
        for (const auto& region : regions) {
            results.emplace_back(RegisterResult{Status::OK(), region.handle});
        }
        return Status::OK();
    }

    Status UnregisterRegions(const std::vector<MRHandle>&) override { return Status::OK(); }

private:
    TransportConfig config_;
    bool initialized_{false};
    TaskId nextTaskId_{1000};
};

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
        KVBuffer{MakeCacheKey("alpha"), buffer},
        KVBuffer{MakeCacheKey("beta"),  buffer},
        KVBuffer{MakeCacheKey("gamma"), buffer},
        KVBuffer{MakeCacheKey("delta"), buffer},
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
    auto client = CreateAsuClient([] { return std::make_unique<StubTransport>(); });
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

    std::vector<CacheKey> keys{MakeCacheKey("alpha"), MakeCacheKey("beta"), MakeCacheKey("gamma"),
                               MakeCacheKey("delta")};
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
