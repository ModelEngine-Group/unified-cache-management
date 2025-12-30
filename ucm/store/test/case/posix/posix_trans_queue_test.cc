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
#include "detail/data_generator.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/global_config.h"
#include "posix/cc/space_layout.h"
#include "posix/cc/trans_queue.h"
#include "template/hashset.h"

class UCPosixTransQueueTest : public UC::Test::Detail::PathBase {};

TEST_F(UCPosixTransQueueTest, TransBlock)
{
    using namespace UC::PosixStore;
    Config config;
    config.tensorSize = 32768 * 64;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize;
    config.storageBackends.push_back(Path());
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    UC::PosixStore::SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    TransQueue queue;
    auto s = queue.Setup(config, &failureSet, &layout);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    constexpr size_t nBlocks = 1;
    UC::Test::Detail::DataGenerator data1{nBlocks, config.blockSize};
    data1.GenerateRandom();
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    desc1.push_back(UC::Detail::Shard{block, 0, {data1.Buffer()}});
    auto task1 = std::make_shared<TransTask>(TransTask::Type::DUMP, desc1);
    auto waiter1 = std::make_shared<UC::Latch>();
    queue.Push(task1, waiter1);
    waiter1->Wait();
    UC::Test::Detail::DataGenerator data2{nBlocks, config.blockSize};
    data2.Generate();
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    desc2.push_back(UC::Detail::Shard{block, 0, {data2.Buffer()}});
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc2);
    auto waiter2 = std::make_shared<UC::Latch>();
    queue.Push(task2, waiter2);
    waiter2->Wait();
    ASSERT_EQ(data1.Compare(data2), 0);
}

TEST_F(UCPosixTransQueueTest, TransBlockLayerWise)
{
    using namespace UC::PosixStore;
    constexpr size_t nShards = 8;
    Config config;
    config.tensorSize = 4096;
    config.shardSize = config.tensorSize;
    config.blockSize = config.shardSize * nShards;
    config.storageBackends.push_back(Path());
    UC::HashSet<UC::Detail::TaskHandle> failureSet;
    UC::PosixStore::SpaceLayout layout;
    ASSERT_TRUE(layout.Setup(config).Success());
    TransQueue queue;
    auto s = queue.Setup(config, &failureSet, &layout);
    ASSERT_EQ(s, UC::Status::OK());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    auto data1 = UC::Test::Detail::TypesHelper::MakeArray<UC::Test::Detail::DataGenerator, nShards>(
        size_t(1), config.tensorSize);
    UC::Detail::TaskDesc desc1;
    desc1.brief = "Dump";
    for (size_t i = 0; i < nShards; i++) {
        auto& d = data1[i];
        d.GenerateRandom();
        desc1.push_back(UC::Detail::Shard{block, i, {d.Buffer()}});
    }
    auto task1 = std::make_shared<TransTask>(TransTask::Type::DUMP, desc1);
    auto waiter1 = std::make_shared<UC::Latch>();
    queue.Push(task1, waiter1);
    waiter1->Wait();
    auto data2 = UC::Test::Detail::TypesHelper::MakeArray<UC::Test::Detail::DataGenerator, nShards>(
        size_t(1), config.tensorSize);
    UC::Detail::TaskDesc desc2;
    desc2.brief = "Load";
    for (size_t i = 0; i < nShards; i++) {
        auto& d = data2[i];
        d.Generate();
        desc2.push_back(UC::Detail::Shard{block, i, {d.Buffer()}});
    }
    auto task2 = std::make_shared<TransTask>(TransTask::Type::LOAD, desc2);
    auto waiter2 = std::make_shared<UC::Latch>();
    queue.Push(task2, waiter2);
    waiter2->Wait();
    for (size_t i = 0; i < nShards; i++) { ASSERT_EQ(data1[i].Compare(data2[i]), 0); }
}
