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

#include <chrono>
#include <random>
#include <gtest/gtest.h>
#include "template/topn_heap.h"

class UCTopNHeapTest : public testing::Test {
protected:
    template <typename Heap, typename T>
    void RandomPush(Heap& heap, T number)
    {
        std::vector<T> a(number);
        for (T i = 0; i < number; i++) { a[i] = i; }
        std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::shuffle(a.begin(), a.end(), rng);
        for (auto& v : a) { heap.Push(v); }
    }
};

TEST_F(UCTopNHeapTest, TopMaxNHeap)
{
    UC::TopNHeap<size_t> heap{5};
    ASSERT_EQ(heap.Capacity(), 5);
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
    RandomPush(heap, size_t(10));
    ASSERT_EQ(heap.Size(), heap.Capacity());
    ASSERT_FALSE(heap.Empty());
    for (size_t x : {5, 6, 7, 8, 9}) {
        auto v = heap.Top();
        heap.Pop();
        EXPECT_EQ(v, x);
    }
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
}

TEST_F(UCTopNHeapTest, TopMaxNFixedHeap)
{
    UC::TopNFixedHeap<size_t, 5> heap;
    ASSERT_EQ(heap.Capacity(), 5);
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
    RandomPush(heap, size_t(10));
    ASSERT_EQ(heap.Size(), heap.Capacity());
    ASSERT_FALSE(heap.Empty());
    for (size_t x : {5, 6, 7, 8, 9}) {
        auto v = heap.Top();
        heap.Pop();
        EXPECT_EQ(v, x);
    }
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
}

TEST_F(UCTopNHeapTest, TopMinNHeap)
{
    UC::TopNHeap<size_t, std::greater<size_t>> heap{5};
    ASSERT_EQ(heap.Capacity(), 5);
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
    RandomPush(heap, size_t(10));
    ASSERT_EQ(heap.Size(), heap.Capacity());
    ASSERT_FALSE(heap.Empty());
    for (size_t x : {4, 3, 2, 1, 0}) {
        auto v = heap.Top();
        heap.Pop();
        EXPECT_EQ(v, x);
    }
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
}

TEST_F(UCTopNHeapTest, CustomStructTopN)
{
    struct Data {
        size_t val;
        size_t prop;
    };
    struct CmpProp {
        bool operator() (const Data& a, const Data& b) const {return a.prop > b.prop;}
    };
    UC::TopNHeap<Data, CmpProp> heap{5};
    Data data[] = {
        Data{100, 7},
        Data{200, 3},
        Data{300, 9},
        Data{400, 1},
        Data{500, 5},
        Data{600, 8},
        Data{700, 2}
    };
    for (auto& d : data) { heap.Push(d); }
    ASSERT_EQ(heap.Size(), heap.Capacity());
    ASSERT_FALSE(heap.Empty());
    for (size_t x : {100, 500, 200, 700, 400}) {
        auto v = heap.Top();
        heap.Pop();
        EXPECT_EQ(v.val, x);
    }
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
}

TEST_F(UCTopNHeapTest, BlockFilePathTopN) {
    struct BlockFileInfo {
        std::string path;
        size_t timestamp;
    };
    struct CmpTimestamp {
        bool operator() (const BlockFileInfo& a, const BlockFileInfo& b) const { return a.timestamp > b.timestamp; }
    };
    UC::TopNHeap<BlockFileInfo, CmpTimestamp> heap{3};
    BlockFileInfo data[] = {
        {"block1", 1000003},
        {"block2", 1000000},
        {"block3", 1000001},
        {"block4", 1000001},
        {"block5", 1000002}
    };
    for (auto& d : data) { heap.Push(d); }
    ASSERT_EQ(heap.Size(), heap.Capacity());
    ASSERT_FALSE(heap.Empty());
    for (auto& x : {"block4", "block3", "block2"}) {
        EXPECT_EQ(heap.Top().path, x);
        heap.Pop();
    }
    ASSERT_EQ(heap.Size(), 0);
    ASSERT_TRUE(heap.Empty());
}