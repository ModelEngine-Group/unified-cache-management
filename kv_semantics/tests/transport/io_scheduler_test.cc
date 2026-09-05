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
#include "task/io_scheduler.h"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace kv::test {
namespace {

CacheKey MakeCacheKey(const std::string& text)
{
    CacheKey key{};
    const auto size = std::min(text.size(), key.size());
    if (size > 0) { std::memcpy(key.data(), text.data(), size); }
    return key;
}

TEST(IoSchedulerTest, SplitEntryBatchPreservesOrderAndUsesViews)
{
    std::vector<KVBuffer> entries(5);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        entries[index].key = MakeCacheKey("key_" + std::to_string(index));
    }

    TransportConfig config;
    config.asuBatchLoadIoNum = 2;
    IoScheduler scheduler(config);
    const auto batches = scheduler.SplitForAsu(BatchView<KVBuffer>{entries.data(), entries.size()},
                                               AsuOpType::BATCH_LOAD);

    ASSERT_EQ(batches.size(), std::size_t{3});
    EXPECT_EQ(batches[0].entries.size, std::size_t{2});
    EXPECT_EQ(batches[1].entries.size, std::size_t{2});
    EXPECT_EQ(batches[2].entries.size, std::size_t{1});
    EXPECT_EQ(&batches[0].entries[0], &entries[0]);
    EXPECT_EQ(&batches[1].entries[0], &entries[2]);
    EXPECT_EQ(&batches[2].entries[0], &entries[4]);
    EXPECT_EQ(batches[1].entries[1].key, MakeCacheKey("key_3"));
}

TEST(IoSchedulerTest, GetSqeIoNumMatchesOperationKind)
{
    TransportConfig config;
    config.asuBatchLoadIoNum = 3;
    config.asuBatchStoreIoNum = 4;
    config.asuDeleteIoNum = 5;
    config.asuQueryIoNum = 6;
    IoScheduler scheduler(config);

    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::LOAD), std::size_t{1});
    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::STORE), std::size_t{1});
    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::BATCH_LOAD), std::size_t{3});
    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::BATCH_STORE), std::size_t{4});
    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::DELETE), std::size_t{5});
    EXPECT_EQ(scheduler.GetSqeIoNum(AsuOpType::QUERY), std::size_t{6});
}

TEST(IoSchedulerTest, SplitByOperationUsesHeldConfig)
{
    TransportConfig config;
    config.asuBatchLoadIoNum = 2;
    IoScheduler scheduler(config);
    std::vector<KVBuffer> entries(5);

    const auto batches = scheduler.SplitForAsu(BatchView<KVBuffer>{entries.data(), entries.size()},
                                               AsuOpType::BATCH_LOAD);

    ASSERT_EQ(batches.size(), std::size_t{3});
    EXPECT_EQ(batches[0].entries.size, std::size_t{2});
    EXPECT_EQ(batches[1].entries.size, std::size_t{2});
    EXPECT_EQ(batches[2].entries.size, std::size_t{1});
}

}  // namespace
}  // namespace kv::test
