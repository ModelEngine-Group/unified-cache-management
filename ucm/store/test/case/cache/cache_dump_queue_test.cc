/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include "cache/cc/dump_queue.h"
#include "detail/data_generator.h"
#include "detail/mock_store.h"
#include "detail/random.h"
#include "detail/types_helper.h"

class UCCacheDumpQueueTest : public testing::Test {
public:
    UC::Test::Detail::Random rd;
    static UC::Detail::TaskHandle NextId()
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
};

TEST_F(UCCacheDumpQueueTest, DumpOneBlock)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(testing::Invoke(NextId));
    UC::Latch finish{};
    finish.Up();
    EXPECT_CALL(backend, Wait).WillOnce(testing::Invoke([&finish]() {
        finish.Done();
        return UC::Status::OK();
    }));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_FALSE(failureSet.Contains(task->id));
    finish.Wait();
}

TEST_F(UCCacheDumpQueueTest, DumpBlockWhileBackendSubmitFailed)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Status::Error()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_TRUE(failureSet.Contains(task->id));
}

TEST_F(UCCacheDumpQueueTest, DumpBlockWhileBackendUnhealthy)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(testing::Return(UC::Status::StoreUnhealthy()));
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    size_t tensorSize = 32768;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    TransBuffer buffer;
    DumpQueue dumpQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = dumpQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_TRUE(failureSet.Contains(task->id));
    EXPECT_EQ(task->FailureStatus(), UC::Status::StoreUnhealthy());
}

TEST_F(UCCacheDumpQueueTest, HostToHostGatherSupportsVariableTensorSizes)
{
    using namespace UC::CacheStore;
    using namespace testing;
    constexpr size_t kSize = 16;
    constexpr size_t vSize = 8;
    std::array<uint8_t, kSize> k{};
    std::array<uint8_t, vSize> v{};
    for (size_t i = 0; i < k.size(); ++i) { k[i] = static_cast<uint8_t>(i + 1); }
    for (size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<uint8_t>(i + 33); }

    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Dump).WillOnce(Invoke([&](UC::Detail::TaskDesc desc) {
        EXPECT_EQ(desc.size(), 1);
        auto* data = static_cast<uint8_t*>(desc[0].addrs[0]);
        EXPECT_EQ(std::memcmp(data, k.data(), k.size()), 0);
        EXPECT_EQ(std::memcmp(data + k.size(), v.data(), v.size()), 0);
        return NextId();
    }));
    EXPECT_CALL(backend, Wait).WillOnce(Return(UC::Status::OK()));

    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {kSize, vSize};
    config.shardSize = kSize + vSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    config.cacheUseHostBuffer = true;
    config.cacheSdmaDirect = false;

    TransBuffer buffer;
    DumpQueue dumpQ;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    ASSERT_EQ(dumpQ.Setup(config, &failureSet, &buffer), UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId(
        "a1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc desc{
        {blockId, 0, {k.data(), v.data()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::DUMP, desc);
    auto waiter = std::make_shared<UC::Latch>();
    dumpQ.Submit(task, waiter);
    waiter->Wait();
    ASSERT_FALSE(failureSet.Contains(task->id));
}
