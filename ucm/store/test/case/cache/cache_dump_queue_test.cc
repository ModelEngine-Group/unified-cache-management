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
    config.storeBackend = (uintptr_t)(void*)&backend;
    config.tensorSize = 32768;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferNumber = 2048;
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
    config.storeBackend = (uintptr_t)(void*)&backend;
    config.tensorSize = 32768;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.deviceId = 0;
    config.bufferNumber = 2048;
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
