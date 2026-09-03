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
#include <unordered_map>
#include "kv_client_impl.h"

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
    Status Init(const TransportConfig& config,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)transProvider;
        config_ = config;
        initialized_ = true;
        return Status::OK();
    }

    Status Init(const std::string&, std::shared_ptr<TransProvider>) override
    {
        return Status::Error(StatusCode::UNSUPPORTED, "stub transport config path unsupported");
    }

    Status Shutdown() override
    {
        initialized_ = false;
        return Status::OK();
    }

    Status CheckHealth() override { return Status::OK(); }

    Status RunQuery(BatchView<CacheKey> keys, QueryResult& result)
    {
        result.exists.assign(keys.size, true);
        result.prefixHitKeys = 0;
        return Status::OK();
    }

    Status Submit(const TransportTaskPtr& task) override
    {
        if (!task) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "stub transport task is null");
        }
        if (task->opType == AsuOpType::QUERY) {
            QueryResult queryResult;
            auto status = RunQuery({task->keys.data(), task->keys.size()}, queryResult);
            if (!status.ok()) {
                task->taskId = kInvalidTaskId;
                return status;
            }

            task->taskId = nextTaskId_++;
            if (task->onComplete) {
                TaskResult result;
                result.status = Status::OK();
                result.entryStatus.assign(task->keys.size(), Status::OK());
                result.queryResult = std::move(queryResult);
                task->onComplete(std::move(result));
            }
            return Status::OK();
        }

        switch (task->opType) {
            case AsuOpType::LOAD:
            case AsuOpType::STORE:
            case AsuOpType::BATCH_LOAD:
            case AsuOpType::BATCH_STORE:
            case AsuOpType::DELETE: break;
            default:
                task->taskId = kInvalidTaskId;
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "unsupported stub transport operation");
        }
        task->taskId = nextTaskId_++;
        if (task->onComplete) {
            TaskResult result;
            result.status = Status::OK();
            task->onComplete(std::move(result));
        }
        return Status::OK();
    }

    Status Cancel(TaskId) override { return Status::OK(); }

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
    first.providerType = TransProviderType::FAKE;
    first.maxInflightTasks = 64;
    first.timeoutMs = 100;

    TransportConfig second;
    second.asuName = "asu-smoke-1";
    second.asuId = 1002;
    second.providerType = TransProviderType::FAKE;
    second.maxInflightTasks = 64;
    second.timeoutMs = 100;

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

    ASSERT_TRUE(client.Check(taskId));
}

Status QueryAndWait(AsuClient& client, const std::vector<CacheKey>& keys, QueryResult& result,
                    std::uint64_t timeoutMs)
{
    TaskId taskId{kInvalidTaskId};
    auto status = client.QueryAsync(keys, taskId);
    if (!status.ok()) { return status; }

    TaskResult taskResult;
    status = client.Wait(taskId, timeoutMs, taskResult);
    if (taskResult.queryResult.has_value()) {
        result = std::move(*taskResult.queryResult);
    } else if (status.ok()) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "client query result is missing");
    }
    return status;
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
    QueryResult queryResult;
    status = QueryAndWait(*client, keys, queryResult, 500);
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
