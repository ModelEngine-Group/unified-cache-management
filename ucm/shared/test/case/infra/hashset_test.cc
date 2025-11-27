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
#include <thread>
#include "template/hashset.h"

class UCHashSetTest : public testing::Test {};

TEST_F(UCHashSetTest, Basic)
{
    UC::HashSet<int> hs;
    EXPECT_FALSE(hs.Contains(1));
    hs.Insert(1);
    EXPECT_TRUE(hs.Contains(1));
    hs.Insert(1);
    EXPECT_TRUE(hs.Contains(1));
    hs.Remove(1);
    EXPECT_FALSE(hs.Contains(1));
    hs.Remove(1);
    EXPECT_FALSE(hs.Contains(1));
}

TEST_F(UCHashSetTest, ManyKeys)
{
    constexpr int N = 100'000;
    UC::HashSet<int> hs;
    for (int i = 0; i < N; ++i) { hs.Insert(i); }
    for (int i = 0; i < N; ++i) { EXPECT_TRUE(hs.Contains(i)); }
    for (int i = 0; i < N; i += 2) { hs.Remove(i); }
    for (int i = 0; i < N; ++i) { EXPECT_EQ(hs.Contains(i), i % 2 != 0); }
}

TEST_F(UCHashSetTest, ConcurrentInsert)
{
    constexpr int ThreadN = 8;
    constexpr int PerThread = 50'000;
    UC::HashSet<int> hs;
    auto worker = [&](int offset) {
        for (int i = 0; i < PerThread; ++i) { hs.Insert(offset + i); }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < ThreadN; ++t) { threads.emplace_back(worker, t * PerThread); }
    for (auto& t : threads) t.join();
    for (int i = 0; i < ThreadN * PerThread; ++i) { EXPECT_TRUE(hs.Contains(i)); }
}

TEST_F(UCHashSetTest, StringKey)
{
    UC::HashSet<std::string> hs;
    hs.Insert("hello");
    EXPECT_TRUE(hs.Contains("hello"));
    EXPECT_FALSE(hs.Contains("world"));
    hs.Remove("hello");
    EXPECT_FALSE(hs.Contains("hello"));
}
