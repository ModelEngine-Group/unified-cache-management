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

#include "template/topn_heap.h"
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

class UCTopNHeapTest : public testing::Test {};

namespace {

template <typename Heap>
std::vector<typename Heap::ValueType> DrainHeap(Heap& heap)
{
    std::vector<typename Heap::ValueType> values;
    while (!heap.Empty()) {
        values.push_back(heap.Top());
        heap.Pop();
    }
    return values;
}

struct BlockInfo {
    std::string name;
    size_t timestamp;
};

struct CmpTimestamp {
    bool operator()(const BlockInfo& lhs, const BlockInfo& rhs) const noexcept
    {
        return lhs.timestamp > rhs.timestamp;
    }
};

}  // namespace

TEST_F(UCTopNHeapTest, BasicProperties)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(3);
    EXPECT_TRUE(heap.Empty());
    EXPECT_EQ(heap.Size(), 0);
    EXPECT_EQ(heap.Capacity(), 3);
}

TEST_F(UCTopNHeapTest, ZeroCapacityStaysEmpty)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(0);
    heap.Push({"a", 1});
    heap.Push({"b", 2});

    EXPECT_TRUE(heap.Empty());
    EXPECT_EQ(heap.Size(), 0);
    EXPECT_EQ(heap.Capacity(), 0);
}

TEST_F(UCTopNHeapTest, CapacityOneKeepsEarliestValue)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(1);
    heap.Push({"late", 30});
    EXPECT_EQ(heap.Top().timestamp, 30U);

    heap.Push({"later", 40});
    EXPECT_EQ(heap.Top().timestamp, 30U);

    heap.Push({"early", 10});
    EXPECT_EQ(heap.Top().timestamp, 10U);
    EXPECT_EQ(heap.Size(), 1);
}

TEST_F(UCTopNHeapTest, FullHeapDropsLaterValues)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(3);
    heap.Push({"a", 10});
    heap.Push({"b", 20});
    heap.Push({"c", 30});

    EXPECT_EQ(heap.Top().timestamp, 30U);

    heap.Push({"later", 50});
    EXPECT_EQ(heap.Top().timestamp, 30U);
    EXPECT_EQ(heap.Size(), 3);
}

TEST_F(UCTopNHeapTest, FullHeapReplacesRootWhenEarlierValueArrives)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(3);
    heap.Push({"late", 80});
    heap.Push({"early", 10});
    heap.Push({"middle", 40});
    heap.Push({"earlier", 5});
    heap.Push({"almost-late", 60});

    EXPECT_EQ(heap.Size(), 3);
    EXPECT_EQ(heap.Top().timestamp, 40U);

    std::vector<size_t> timestamps;
    while (!heap.Empty()) {
        timestamps.push_back(heap.Top().timestamp);
        heap.Pop();
    }
    EXPECT_EQ(timestamps, (std::vector<size_t>{40U, 10U, 5U}));
}

TEST_F(UCTopNHeapTest, PreservesDuplicateTimestamps)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(3);
    heap.Push({"a", 2});
    heap.Push({"b", 2});
    heap.Push({"c", 2});
    heap.Push({"d", 2});

    EXPECT_EQ(heap.Size(), 3);
    EXPECT_EQ(heap.Top().timestamp, 2U);

    std::vector<size_t> timestamps;
    while (!heap.Empty()) {
        timestamps.push_back(heap.Top().timestamp);
        heap.Pop();
    }
    EXPECT_EQ(timestamps, (std::vector<size_t>{2U, 2U, 2U}));
}

TEST_F(UCTopNHeapTest, PopMaintainsHeapOrder)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(4);
    heap.Push({"9", 9});
    heap.Push({"1", 1});
    heap.Push({"7", 7});
    heap.Push({"5", 5});
    heap.Push({"6", 6});
    heap.Push({"8", 8});

    EXPECT_EQ(heap.Size(), 4);
    EXPECT_EQ(heap.Top().timestamp, 7U);

    heap.Pop();
    EXPECT_EQ(heap.Top().timestamp, 6U);
    heap.Pop();
    EXPECT_EQ(heap.Top().timestamp, 5U);
    heap.Pop();
    EXPECT_EQ(heap.Top().timestamp, 1U);
    heap.Pop();
    EXPECT_TRUE(heap.Empty());
}

TEST_F(UCTopNHeapTest, ClearResetsLogicalStateAndAllowsReuse)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(3);
    heap.Push({"a", 4});
    heap.Push({"b", 2});
    heap.Push({"c", 9});

    heap.Clear();
    EXPECT_TRUE(heap.Empty());
    EXPECT_EQ(heap.Size(), 0);

    heap.Push({"1", 1});
    heap.Push({"8", 8});
    heap.Push({"6", 6});
    heap.Push({"7", 7});
    EXPECT_EQ(heap.Size(), 3);
    std::vector<size_t> timestamps;
    while (!heap.Empty()) {
        timestamps.push_back(heap.Top().timestamp);
        heap.Pop();
    }
    EXPECT_EQ(timestamps, (std::vector<size_t>{7U, 6U, 1U}));
}

TEST_F(UCTopNHeapTest, SupportsLvalueAndRvaluePushWithBlockInfo)
{
    UC::TopNHeap<BlockInfo, CmpTimestamp> heap(2);
    BlockInfo oldest{"oldest", 10};

    heap.Push(oldest);
    heap.Push(BlockInfo{"newest", 30});
    heap.Push(BlockInfo{"middle", 20});

    ASSERT_EQ(heap.Size(), 2);
    EXPECT_EQ(heap.Top().name, "middle");
    EXPECT_EQ(heap.Top().timestamp, 20U);
}

TEST_F(UCTopNHeapTest, FixedHeapWrapperUsesCompileTimeCapacity)
{
    UC::TopNFixedHeap<BlockInfo, 2, CmpTimestamp> heap;
    heap.Push({"late", 100});
    heap.Push({"early", 10});
    heap.Push({"middle", 50});

    EXPECT_EQ(heap.Capacity(), 2);
    EXPECT_EQ(heap.Size(), 2);

    std::vector<size_t> timestamps;
    while (!heap.Empty()) {
        timestamps.push_back(heap.Top().timestamp);
        heap.Pop();
    }
    EXPECT_EQ(timestamps, (std::vector<size_t>{50U, 10U}));
}
