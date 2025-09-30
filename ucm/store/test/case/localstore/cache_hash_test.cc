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
#include "cache/cache_hash.h"

class UCCacheHashTest : public testing::Test {};

TEST_F(UCCacheHashTest, InsertAndFind)
{
    constexpr uint32_t capacity = 10;
    const std::string id = "1234567890123456";
    UC::CacheHash hash;
    hash.Setup(capacity);
    auto buffer = new (std::nothrow) uint8_t[hash.MemorySize()];
    ASSERT_NE(buffer, nullptr);
    hash.Setup(buffer);
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(hash.Find(id, i), UC::CacheHash::npos); }
    for (uint32_t i = 0; i < capacity; i++) { hash.Insert(id, i, i); }
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_NE(hash.Find(id, i), UC::CacheHash::npos); }
    for (uint32_t i = 0; i < capacity; i++) { hash.Remove(id, i); }
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(hash.Find(id, i), UC::CacheHash::npos); }
    delete[] buffer;
}

TEST_F(UCCacheHashTest, SetupAtSharedMemory)
{
    constexpr uint32_t capacity = 10;
    const std::string id = "1234567890123456";
    UC::CacheHash hash1;
    hash1.Setup(capacity);
    UC::CacheHash hash2;
    hash2.Setup(capacity);
    auto buffer = new (std::nothrow) uint8_t[hash1.MemorySize()];
    ASSERT_NE(buffer, nullptr);
    hash1.Setup(buffer);
    hash2.Setup(buffer);
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(hash2.Find(id, i), UC::CacheHash::npos); }
    for (uint32_t i = 0; i < capacity; i++) { hash1.Insert(id, i, i); }
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(hash2.Find(id, i), i); }
    delete[] buffer;
}

TEST_F(UCCacheHashTest, Evict)
{
    constexpr uint32_t capacity = 10;
    const std::string id = "1234567890123456";
    UC::CacheHash hash;
    hash.Setup(capacity);
    auto buffer = new (std::nothrow) uint8_t[hash.MemorySize()];
    ASSERT_NE(buffer, nullptr);
    hash.Setup(buffer);
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_EQ(hash.Find(id, i), UC::CacheHash::npos); }
    for (uint32_t i = 0; i < capacity; i++) {
        hash.Insert(id, i, i);
        hash.PutRef(i);
    }
    for (uint32_t i = 0; i < capacity; i++) { ASSERT_NE(hash.Find(id, i), UC::CacheHash::npos); }
    ASSERT_EQ(hash.Evict(), UC::CacheHash::npos);
    hash.PutRef(0);
    ASSERT_EQ(hash.Evict(), 0);
    hash.PutRef(id, 2);
    ASSERT_EQ(hash.Evict(), 2);
    ASSERT_EQ(hash.Evict(), UC::CacheHash::npos);
    for (uint32_t i = 0; i < capacity; i++) { hash.PutRef(i); }
    ASSERT_EQ(hash.Evict(), 1);
    delete[] buffer;
}
