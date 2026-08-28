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
#include "cache/cc/load_queue.h"
#include "detail/data_generator.h"
#include "detail/mock_store.h"
#include "detail/random.h"
#include "detail/types_helper.h"

class UCCacheLoadQueueTest : public testing::Test {
public:
    UC::Test::Detail::Random rd;
    static UC::Detail::TaskHandle NextId()
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    }
};

TEST_F(UCCacheLoadQueueTest, LoadSameBlockTwice)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).WillOnce(testing::Return(UC::Status::OK()));
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
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task1 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter1 = std::make_shared<UC::Latch>();
    loadQ.Submit(task1, waiter1);
    waiter1->Wait();
    ASSERT_FALSE(failureSet.Contains(task1->id));
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter2 = std::make_shared<UC::Latch>();
    loadQ.Submit(task2, waiter2);
    waiter2->Wait();
    ASSERT_FALSE(failureSet.Contains(task2->id));
}

TEST_F(UCCacheLoadQueueTest, SharedFailureStopsNonOwnerWait)
{
    using namespace UC::CacheStore;
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).Times(0);
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
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    auto owner = buffer.Get(blockId, shardIdx, true, true);
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);

    owner.MarkFailed(UC::Status::NotFound());

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, LoadWhileBackendSubmitFailed)
{
    using namespace UC::CacheStore;
    using namespace testing;
    std::promise<void> submitEntered;
    std::promise<void> allowSubmitFailure;
    auto allowSubmitFailureFuture = allowSubmitFailure.get_future().share();
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load)
        .WillOnce(Invoke([&](UC::Detail::TaskDesc) -> UC::Expected<UC::Detail::TaskHandle> {
            submitEntered.set_value();
            allowSubmitFailureFuture.wait();
            return UC::Status::NotFound();
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
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    submitEntered.get_future().wait();
    auto observer = buffer.Get(blockId, shardIdx, true, true);

    allowSubmitFailure.set_value();

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(observer.GetState(), TransBuffer::State::FAILED);
    ASSERT_EQ(observer.FailureStatus(), UC::Status::NotFound());
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, LoadWhileBackendWaitFailed)
{
    using namespace UC::CacheStore;
    using namespace testing;
    std::promise<void> waitEntered;
    std::promise<void> allowWaitFailure;
    auto allowWaitFailureFuture = allowWaitFailure.get_future().share();
    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load).WillOnce(testing::Invoke(NextId));
    EXPECT_CALL(backend, Wait).WillOnce(Invoke([&](UC::Detail::TaskHandle) {
        waitEntered.set_value();
        allowWaitFailureFuture.wait();
        return UC::Status::NotFound();
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
    LoadQueue loadQ;
    auto s = buffer.Setup(config);
    ASSERT_EQ(s, UC::Status::OK());
    s = loadQ.Setup(config, &failureSet, &buffer);
    ASSERT_EQ(s, UC::Status::OK());
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t shardIdx = 0;
    UC::Test::Detail::DataGenerator data{1, config.blockSize};
    data.Generate();
    UC::Detail::TaskDesc desc{
        {blockId, shardIdx, {data.Buffer()}}
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);
    waitEntered.get_future().wait();
    auto observer = buffer.Get(blockId, shardIdx, true, true);

    allowWaitFailure.set_value();

    ASSERT_TRUE(waiter->WaitForDuration(1000));
    ASSERT_TRUE(failureSet.Contains(task->id));
    ASSERT_EQ(observer.GetState(), TransBuffer::State::FAILED);
    ASSERT_EQ(observer.FailureStatus(), UC::Status::NotFound());
    ASSERT_EQ(task->FailureStatus(), UC::Status::NotFound());
}

TEST_F(UCCacheLoadQueueTest, HostToHostScatterRunsShardsInParallel)
{
    using namespace UC::CacheStore;
    using namespace testing;
    constexpr size_t kSize = 16;
    constexpr size_t vSize = 8;
    constexpr uint8_t value = 0x5a;
    UC::Latch workersEntered;
    workersEntered.Set(2);

    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load)
        .Times(2)
        .WillRepeatedly(Invoke([&](UC::Detail::TaskDesc desc) {
            std::memset(desc[0].addrs[0], value, kSize + vSize);
            return NextId();
        }));
    EXPECT_CALL(backend, Wait)
        .Times(2)
        .WillRepeatedly(Invoke([&](UC::Detail::TaskHandle) {
            workersEntered.Done();
            return workersEntered.WaitForDuration(1000) ? UC::Status::OK()
                                                        : UC::Status::Timeout();
        }));

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
    config.h2hWorkerNumber = 2;
    config.h2hQueueDepth = 8;

    TransBuffer buffer;
    LoadQueue loadQ;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    ASSERT_EQ(loadQ.Setup(config, &failureSet, &buffer), UC::Status::OK());

    std::array<uint8_t, kSize> k0{}, k1{};
    std::array<uint8_t, vSize> v0{}, v1{};
    auto block0 = UC::Test::Detail::TypesHelper::MakeBlockId(
        "a1b2c3d4e5f6789012345678901234ab");
    auto block1 = UC::Test::Detail::TypesHelper::MakeBlockId(
        "b1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc desc{
        {block0, 0, {k0.data(), v0.data()}},
        {block1, 0, {k1.data(), v1.data()}},
    };
    auto task = std::make_shared<TransTask>(TransTask::Type::LOAD, desc);
    auto waiter = std::make_shared<UC::Latch>();
    loadQ.Submit(task, waiter);

    ASSERT_TRUE(waiter->WaitForDuration(2000));
    ASSERT_FALSE(failureSet.Contains(task->id));
    for (auto byte : k0) { EXPECT_EQ(byte, value); }
    for (auto byte : k1) { EXPECT_EQ(byte, value); }
    for (auto byte : v0) { EXPECT_EQ(byte, value); }
    for (auto byte : v1) { EXPECT_EQ(byte, value); }
}

TEST_F(UCCacheLoadQueueTest, HostToHostConcurrentLoadWaitsBackendOnlyOnce)
{
    using namespace UC::CacheStore;
    using namespace testing;
    constexpr size_t tensorSize = 32;
    constexpr uint8_t value = 0x6b;
    std::promise<void> waitEntered;
    std::promise<void> allowBackendReady;
    auto allowBackendReadyFuture = allowBackendReady.get_future().share();

    UC::Test::Detail::MockStore backend;
    EXPECT_CALL(backend, Load)
        .Times(1)
        .WillOnce(Invoke([&](UC::Detail::TaskDesc desc) {
            std::memset(desc[0].addrs[0], value, tensorSize);
            return NextId();
        }));
    EXPECT_CALL(backend, Wait)
        .Times(1)
        .WillOnce(Invoke([&](UC::Detail::TaskHandle) {
            waitEntered.set_value();
            allowBackendReadyFuture.wait();
            return UC::Status::OK();
        }));

    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    Config config;
    config.storeBackend = &backend;
    config.tensorSizes = {tensorSize};
    config.shardSize = tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferCapacity = config.shardSize * 1024;
    config.uniqueId = rd.RandomString(10);
    config.shareBufferEnable = true;
    config.cacheUseHostBuffer = true;
    config.cacheSdmaDirect = false;
    config.h2hWorkerNumber = 2;
    config.h2hQueueDepth = 8;

    TransBuffer buffer;
    LoadQueue loadQ;
    ASSERT_EQ(buffer.Setup(config), UC::Status::OK());
    ASSERT_EQ(loadQ.Setup(config, &failureSet, &buffer), UC::Status::OK());

    std::array<uint8_t, tensorSize> dst0{}, dst1{};
    auto blockId = UC::Test::Detail::TypesHelper::MakeBlockId(
        "c1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc desc0{
        {blockId, 0, {dst0.data()}}
    };
    UC::Detail::TaskDesc desc1{
        {blockId, 0, {dst1.data()}}
    };
    auto task0 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc0);
    auto task1 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc1);
    auto waiter0 = std::make_shared<UC::Latch>();
    auto waiter1 = std::make_shared<UC::Latch>();

    loadQ.Submit(task0, waiter0);
    waitEntered.get_future().wait();
    // Keep a non-owner reference alive so the second request must share the
    // in-flight buffer instead of becoming a new backend-load owner.
    auto observer = buffer.Get(blockId, 0, true, true);
    ASSERT_FALSE(observer.Owner());
    loadQ.Submit(task1, waiter1);
    allowBackendReady.set_value();

    ASSERT_TRUE(waiter0->WaitForDuration(2000));
    ASSERT_TRUE(waiter1->WaitForDuration(2000));
    ASSERT_FALSE(failureSet.Contains(task0->id));
    ASSERT_FALSE(failureSet.Contains(task1->id));
    for (auto byte : dst0) { EXPECT_EQ(byte, value); }
    for (auto byte : dst1) { EXPECT_EQ(byte, value); }
}
