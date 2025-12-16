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
#include "space/space_manager.h"
#include <gtest/gtest.h>

class PCSpaceManagerTest : public ::testing::Test {};

TEST_F(PCSpaceManagerTest, NewBlock)
{
    UC::SpaceManager spaceMgr;
    const size_t lookupTimeoutMs = 1000;
    ASSERT_EQ(spaceMgr.Setup({"./"}, 1024 * 1024, lookupTimeoutMs), UC::Status::OK());
    const std::string block1 = "block1";
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::OK());
    ASSERT_EQ(spaceMgr.CommitBlock(block1, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.LookupBlock(block1), UC::Status::OK());
}

TEST_F(PCSpaceManagerTest, LookupTimeout)
{
    UC::SpaceManager spaceMgr;
    const size_t lookupTimeoutMs = 0;  // look up timeout
    ASSERT_EQ(spaceMgr.Setup({"./"}, 1024 * 1024, lookupTimeoutMs), UC::Status::OK());
    const std::string block1 = "block2";
    ASSERT_EQ(spaceMgr.NewBlock(block1), UC::Status::OK());
    ASSERT_EQ(spaceMgr.CommitBlock(block1, true), UC::Status::OK());
    ASSERT_EQ(spaceMgr.LookupBlock(block1), UC::Status::Timeout());
}