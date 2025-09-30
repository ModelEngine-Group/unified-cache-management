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
#include "cache/cache_index.h"

class UCCacheIndexTest : public testing::Test {};

TEST_F(UCCacheIndexTest, AcquireAllIndex)
{
    constexpr uint32_t capacity = 10;
    UC::CacheIndex index;
    index.Setup(capacity);
    auto size = index.MemorySize();
    ASSERT_GT(size, 0);
    auto buffer = new (std::nothrow) uint8_t[size];
    ASSERT_NE(buffer, nullptr);
    index.Setup(buffer);
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(index.Acquire(), i); }
    ASSERT_EQ(index.Acquire(), UC::CacheIndex::npos);
    delete[] buffer;
}

TEST_F(UCCacheIndexTest, AtSharedMemory)
{
    constexpr uint32_t capacity = 10;
    UC::CacheIndex index1;
    index1.Setup(capacity);
    auto size = index1.MemorySize();
    ASSERT_GT(size, 0);
    auto buffer = new (std::nothrow) uint8_t[size];
    ASSERT_NE(buffer, nullptr);
    index1.Setup(buffer);
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(index1.Acquire(), i); }
    ASSERT_EQ(index1.Acquire(), UC::CacheIndex::npos);
    UC::CacheIndex index2;
    index2.Setup(capacity);
    ASSERT_EQ(index1.MemorySize(), index2.MemorySize());
    index2.Setup(buffer);
    ASSERT_EQ(index2.Acquire(), UC::CacheIndex::npos);
    for (uint32_t i = 0; i < capacity; i++) { index2.Release(i); }
    delete[] buffer;
}
