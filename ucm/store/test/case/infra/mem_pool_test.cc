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

#include "infra/memory/memory_pool.h"
#include <gtest/gtest.h>

class UCMemoryPoolTest : public ::testing::Test {};

TEST_F(UCMemoryPoolTest, NewBlockAllocateAndCommit)
{
    UC::MemoryPool memPool(10, 2); // 初始化内存池
    const std::string block1 = "block1";
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_EQ(memPool.GetAddress(block1), nullptr);
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_NE(memPool.GetAddress(block1), nullptr);
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::DuplicateKey());
    ASSERT_EQ(memPool.CommitBlock(block1), UC::Status::OK());
    ASSERT_TRUE(memPool.LookupBlock(block1));
}

TEST_F(UCMemoryPoolTest, OutOfCapacity)
{
    UC::MemoryPool memPool(10, 5); // 初始化内存池
    const std::string block1 = "block1";
    const std::string block2 = "block2";
    const std::string block3 = "block3";
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block2), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block3), UC::Status::Error());
}