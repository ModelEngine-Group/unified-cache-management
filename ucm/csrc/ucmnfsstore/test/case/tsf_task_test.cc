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
#include "tsf_task/tsf_task.h"

class UCTaskTest : public testing::Test {};

TEST_F(UCTaskTest, AppendTask)
{
    UC::TsfTask task;
    ASSERT_EQ(task.number, 0);
    ASSERT_EQ(task.size, 0);
    ASSERT_TRUE(task.Append("xxx", 0, 100, 1024).Success());
    ASSERT_TRUE(task.Append("yyy", 0, 100, 1024).Success());
    ASSERT_FALSE(task.Append("yyy", 0, 100, 2048).Success());
    ASSERT_EQ(task.number, 2);
    ASSERT_EQ(task.size, 1024);
}
