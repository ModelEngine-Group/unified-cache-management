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
#include <future>
#include <gtest/gtest.h>
#include "tsf_task/tsf_task_waiter.h"

class UCTsfTaskWaiterTest : public ::testing::Test {};

TEST_F(UCTsfTaskWaiterTest, TaskTimeout)
{
    UC::TsfTaskWaiter waiter{1, 1024, 1, "xxx"};
    auto fut = std::async([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        waiter.Done();
    });
    ASSERT_FALSE(waiter.Wait(1));
    fut.get();
    ASSERT_TRUE(waiter.Finish());
}

TEST_F(UCTsfTaskWaiterTest, TaskSuccess)
{
    UC::TsfTaskWaiter waiter{2, 1024, 1, "xxx"};
    auto fut = std::async([&] { waiter.Done(); });
    ASSERT_TRUE(waiter.Wait(1000));
    ASSERT_TRUE(waiter.Finish());
    fut.get();
}

TEST_F(UCTsfTaskWaiterTest, TaskTimeoutButSuccess)
{
    UC::TsfTaskWaiter waiter{3, 1024, 1, "xxx"};
    auto fut = std::async([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        waiter.Done();
    });
    fut.get();
    ASSERT_TRUE(waiter.Finish());
    ASSERT_TRUE(waiter.Wait(1));
}
