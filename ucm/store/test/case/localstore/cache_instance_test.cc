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
#include <mockcpp/mockcpp.hpp>
#include "cache/cache_instance.h"

class UCCacheInstanceTest : public testing::Test {};

TEST_F(UCCacheInstanceTest, StoreAndLoad)
{
    using Data = size_t;
    constexpr Data data = 2048;
    constexpr auto ioSize = sizeof(Data);
    constexpr uint32_t capacity = 1024;
    constexpr size_t totalSize = ioSize * capacity;
    MOCKER(UC::CacheSegment::TotalSize).stubs().will(returnValue(totalSize));
    {
        UC::CacheInstance instance;
        ASSERT_TRUE(instance.Setup(ioSize, capacity).Success());
        const std::string id = "1234567890987654";
        const size_t offset = 1024;
        ASSERT_EQ(instance.Find(id, offset), nullptr);
        auto addr1 = instance.Alloc(id, offset);
        ASSERT_NE(addr1, nullptr);
        *(Data*)addr1 = data;
        instance.PutRef(id, offset);
        auto addr2 = instance.Find(id, offset);
        ASSERT_NE(addr2, nullptr);
        ASSERT_EQ(*(Data*)addr2, data);
        instance.PutRef(id, offset);
    }
    GlobalMockObject::verify();
}

TEST_F(UCCacheInstanceTest, CacheShare)
{
    using Data = size_t;
    constexpr Data data = 2048;
    constexpr auto ioSize = sizeof(Data);
    constexpr uint32_t capacity = 1024;
    constexpr size_t totalSize = ioSize * capacity;
    MOCKER(UC::CacheSegment::TotalSize).stubs().will(returnValue(totalSize));
    {
        UC::CacheInstance instance1;
        ASSERT_TRUE(instance1.Setup(ioSize, capacity).Success());
        UC::CacheInstance instance2;
        ASSERT_TRUE(instance2.Setup(ioSize, capacity).Success());
        const std::string id = "1234567890987654";
        const size_t offset = 1024;
        ASSERT_EQ(instance2.Find(id, offset), nullptr);
        auto addr1 = instance1.Alloc(id, offset);
        ASSERT_NE(addr1, nullptr);
        *(Data*)addr1 = data;
        instance1.PutRef(id, offset);
        auto addr2 = instance2.Find(id, offset);
        ASSERT_NE(addr2, nullptr);
        ASSERT_EQ(*(Data*)addr2, data);
        instance2.PutRef(id, offset);
    }
    GlobalMockObject::verify();
}
