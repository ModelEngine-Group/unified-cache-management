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
#include "io_scheduler.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace UC::ASU {
namespace {

TEST(IoSchedulerTest, SplitEntryBatchPreservesOrderAndUsesViews)
{
    std::vector<KVBuffer> entries(5);
    for (std::size_t index = 0; index < entries.size(); ++index) {
        entries[index].key = "key_" + std::to_string(index);
    }

    IoScheduler scheduler;
    const auto batches =
        scheduler.SplitForAsu(BatchView<KVBuffer>{entries.data(), entries.size()}, 2);

    ASSERT_EQ(batches.size(), std::size_t{3});
    EXPECT_EQ(batches[0].entries.size, std::size_t{2});
    EXPECT_EQ(batches[1].entries.size, std::size_t{2});
    EXPECT_EQ(batches[2].entries.size, std::size_t{1});
    EXPECT_EQ(&batches[0].entries[0], &entries[0]);
    EXPECT_EQ(&batches[1].entries[0], &entries[2]);
    EXPECT_EQ(&batches[2].entries[0], &entries[4]);
    EXPECT_EQ(batches[1].entries[1].key, "key_3");
}

TEST(IoSchedulerTest, SplitKeyBatchReturnsEmptyForEmptyInputOrZeroLimit)
{
    IoScheduler scheduler;
    std::vector<CacheKey> keys = {"a", "b"};

    EXPECT_TRUE(scheduler.SplitForAsu(BatchView<CacheKey>{keys.data(), 0}, 2).empty());
    EXPECT_TRUE(scheduler.SplitForAsu(BatchView<CacheKey>{keys.data(), keys.size()}, 0).empty());
}

TEST(IoSchedulerTest, GetSqeBatchLimitMatchesOperationKind)
{
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::LOAD), std::size_t{1});
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::STORE), std::size_t{1});
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::BATCH_LOAD), kAsuBatchLoadMaxIoNum);
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::BATCH_STORE), kAsuBatchStoreMaxIoNum);
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::DELETE), kAsuDeleteMaxIoNum);
    EXPECT_EQ(GetSqeBatchLimit(TransportOpType::QUERY), kAsuQueryMaxIoNum);
}

}  // namespace
}  // namespace UC::ASU
