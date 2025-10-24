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
    UC::MemoryPool memPool(-1, 10, 2); // 初始化内存池
    const std::string block1 = "block1";
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_EQ(memPool.GetOffest(block1), nullptr);
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_NE(memPool.GetOffest(block1), nullptr);
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::DuplicateKey());
    ASSERT_EQ(memPool.CommitBlock(block1, true), UC::Status::OK());
    ASSERT_TRUE(memPool.LookupBlock(block1));
}

TEST_F(UCMemoryPoolTest, EvictOldBlock)
{
    UC::MemoryPool memPool(-1, 10, 5); // 初始化内存池
    const std::string block1 = "block1";
    const std::string block2 = "block2";
    const std::string block3 = "block3";
    size_t* offset = nullptr;
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block1), nullptr);
    ASSERT_EQ(memPool.GetOffest(block1, offset), true);
    ASSERT_EQ(memPool.NewBlock(block2), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block2), nullptr);
    ASSERT_EQ(memPool.GetOffest(block2, offset), true);
    memPool.CommitBlock(block1, true);
    memPool.CommitBlock(block2, true);
    ASSERT_EQ(memPool.NewBlock(block3), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block3), nullptr);
    ASSERT_EQ(memPool.GetOffest(block3, offset), true);
    // ASSERT_EQ(memPool.GetOffest(block1), nullptr);
    ASSERT_EQ(memPool.GetOffest(block1, offset), false);
    // ASSERT_NE(memPool.GetOffest(block2), nullptr);
    ASSERT_EQ(memPool.GetOffest(block2, offset), true);
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_TRUE(memPool.LookupBlock(block2));
    ASSERT_FALSE(memPool.LookupBlock(block3));
}

TEST_F(UCMemoryPoolTest, OldBlockCommitFalse)
{
    UC::MemoryPool memPool(-1, 32, 8); // 初始化内存池
    const std::string block1 = "block1";
    const std::string block2 = "block2";
    const std::string block3 = "block3";
    const std::string block4 = "block4";
    const std::string block5 = "block5";
    size_t* offset = nullptr;
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block1), nullptr);
    ASSERT_EQ(memPool.GetOffest(block1, offset), true);
    ASSERT_EQ(memPool.NewBlock(block2), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block2), nullptr);
    ASSERT_EQ(memPool.GetOffest(block2, offset), true);
    ASSERT_EQ(memPool.NewBlock(block3), UC::Status::OK());
    // ASSERT_NE(memPool.GetOffest(block3), nullptr);
    ASSERT_EQ(memPool.GetOffest(block3, offset), true);
    memPool.CommitBlock(block1, true);
    memPool.CommitBlock(block2, false);
    ASSERT_TRUE(memPool.LookupBlock(block1));
    ASSERT_FALSE(memPool.LookupBlock(block2));
    ASSERT_FALSE(memPool.LookupBlock(block3));
    ASSERT_EQ(memPool.NewBlock(block4), UC::Status::OK());
    // ASSERT_EQ(memPool.GetOffest(block4), 8);
    ASSERT_EQ(memPool.GetOffest(block4, offset), true);
    ASSERT_EQ(*offset, 8);
    ASSERT_EQ(memPool.NewBlock(block5), UC::Status::OK());
    // ASSERT_EQ(memPool.GetOffest(block5), 24);
    ASSERT_EQ(memPool.GetOffest(block5, offset), true);
    ASSERT_EQ(*offset, 24);
    memPool.CommitBlock(block3, true);
    memPool.CommitBlock(block4, true);
    memPool.CommitBlock(block5, true);
    ASSERT_TRUE(memPool.LookupBlock(block1));
    ASSERT_FALSE(memPool.LookupBlock(block2));
    ASSERT_TRUE(memPool.LookupBlock(block3));
    ASSERT_TRUE(memPool.LookupBlock(block4));
    ASSERT_TRUE(memPool.LookupBlock(block5));

    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::DuplicateKey());
    ASSERT_EQ(memPool.NewBlock(block2), UC::Status::OK());
    // ASSERT_EQ(memPool.GetOffest(block2), 0);
    ASSERT_EQ(memPool.GetOffest(block2, offset), true);
    ASSERT_EQ(*offset, 0);
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_FALSE(memPool.LookupBlock(block2));
    memPool.CommitBlock(block2, true);
    ASSERT_TRUE(memPool.LookupBlock(block2));
}

TEST_F(UCMemoryPoolTest, NoCommittedBlock)
{
    UC::MemoryPool memPool(-1, 32, 8); // 初始化内存池
    const std::string block1 = "block1";
    const std::string block2 = "block2";
    const std::string block3 = "block3";
    const std::string block4 = "block4";
    const std::string block5 = "block5";
    const std::string block6 = "block6";
    size_t* offset = nullptr;
    ASSERT_EQ(memPool.NewBlock(block1), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block2), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block3), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block4), UC::Status::OK());
    ASSERT_EQ(memPool.NewBlock(block5), UC::Status::Error());
    memPool.CommitBlock(block1, true);
    ASSERT_TRUE(memPool.LookupBlock(block1));
    ASSERT_EQ(memPool.NewBlock(block5), UC::Status::OK());
    // ASSERT_EQ(memPool.GetOffest(block5), 0);
    ASSERT_EQ(memPool.GetOffest(block5, offset), true);
    ASSERT_EQ(*offset, 0);
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_EQ(memPool.NewBlock(block6), UC::Status::Error());
    // ASSERT_EQ(memPool.GetOffest(block2), 8);
    ASSERT_EQ(memPool.GetOffest(block2, offset), true);
    ASSERT_EQ(*offset, 8);
    memPool.CommitBlock(block2, false);
    // ASSERT_EQ(memPool.GetOffest((block2)), nullptr);
    ASSERT_EQ(memPool.GetOffest(block2, offset), false);
    ASSERT_FALSE(memPool.LookupBlock(block1));
    ASSERT_EQ(memPool.NewBlock(block6), UC::Status::OK());
    // ASSERT_EQ(memPool.GetOffest(block6), 8);
    ASSERT_EQ(memPool.GetOffest(block6, offset), true);
    ASSERT_EQ(*offset, 8);
    ASSERT_FALSE(memPool.LookupBlock(block6));
    memPool.CommitBlock(block6, true);
    ASSERT_TRUE(memPool.LookupBlock(block6));
    // ASSERT_EQ(memPool.GetOffest(block6), 8);
    ASSERT_EQ(memPool.GetOffest(block6, offset), true);
    ASSERT_EQ(*offset, 8);
}