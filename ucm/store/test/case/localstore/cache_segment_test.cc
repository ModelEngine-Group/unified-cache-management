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
#include <sys/mman.h>
#include "cache/cache_segment.h"

class UCCacheSegmentTest : public testing::Test {};

TEST_F(UCCacheSegmentTest, SetupOnSharedMemory)
{
    using T = size_t;
    constexpr T data = 100;
    constexpr size_t ioSize = sizeof(T);
    constexpr size_t index = 15;
    constexpr size_t ioNumber = index + 1;
    constexpr size_t totalSize = ioSize * ioNumber;
    MOCKER(UC::CacheSegment::TotalSize).stubs().will(returnValue(totalSize));
    auto seg1 = new (std::nothrow) UC::CacheSegment();
    auto seg2 = new (std::nothrow) UC::CacheSegment();
    ASSERT_NE(seg1, nullptr);
    ASSERT_NE(seg2, nullptr);
    ASSERT_TRUE(seg1->Setup(10, ioSize).Success());
    ASSERT_TRUE(seg2->Setup(10, ioSize).Success());
    ASSERT_NE(*(T*)seg2->At(index), data);
    *(T*)(seg1->At(index)) = data;
    delete seg1;
    ASSERT_EQ(*(T*)seg2->At(index), data);
    delete seg2;
    GlobalMockObject::verify();
}

TEST_F(UCCacheSegmentTest, SetupWhileShmOpenFailed)
{
    using T = size_t;
    constexpr size_t ioSize = sizeof(T);
    constexpr size_t index = 15;
    constexpr size_t ioNumber = index + 1;
    constexpr size_t totalSize = ioSize * ioNumber;
    MOCKER(UC::CacheSegment::TotalSize).stubs().will(returnValue(totalSize));
    MOCKER(shm_open).stubs().will(returnValue(-1));
    UC::CacheSegment seg;
    ASSERT_TRUE(seg.Setup(0, ioSize).Failure());
    GlobalMockObject::verify();
}
