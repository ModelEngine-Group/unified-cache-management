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
#include "cmn/path_base.h"
#include "space/space_manager.h"
#include "tsf_task/tsf_task_queue.h"

class UCTsfTaskQueueTest : public UC::PathBase {};

TEST_F(UCTsfTaskQueueTest, TransBetweenHostAndSSD)
{
    constexpr size_t blockSize = sizeof(size_t);
    constexpr size_t number = 1024;
    size_t host1[number];
    size_t host2[number];
    UC::SpaceManager spaceMgr;
    ASSERT_TRUE(spaceMgr.Setup({this->Path()}, blockSize).Success());
    UC::TsfTaskSet failureSet;
    UC::TsfTaskQueue q;
    ASSERT_TRUE(q.Setup(-1, 128, blockSize, number, &failureSet, spaceMgr.GetSpaceLayout()).Success());
    UC::TsfTask task1(UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST, "H2S");
    UC::TsfTask task2(UC::TsfTask::Type::LOAD, UC::TsfTask::Location::HOST, "S2H");
    UC::Random rd;
    std::vector<std::string> blockIds(number);
    for (size_t i = 0; i < number; i++) {
        host1[i] = i;
        host2[i] = 0;
        blockIds[i] = rd.RandomString(16);
        ASSERT_TRUE(spaceMgr.NewBlock(blockIds[i]).Success());
        ASSERT_TRUE(task1.Append(blockIds[i], 0, (uintptr_t)(host1 + i), blockSize).Success());
        ASSERT_TRUE(task2.Append(blockIds[i], 0, (uintptr_t)(host2 + i), blockSize).Success());
    }
    UC::TsfTaskWaiter waiter1{1};
    task1.id = 1;
    task1.waiter = &waiter1;
    q.Push(std::move(task1));
    waiter1.Wait();
    ASSERT_FALSE(failureSet.Contains(1));
    for (size_t i = 0; i < number; i++) { spaceMgr.CommitBlock(blockIds[i], true); }
    UC::TsfTaskWaiter waiter2{1};
    task2.id = 2;
    task2.waiter = &waiter2;
    q.Push(std::move(task2));
    waiter2.Wait();
    ASSERT_FALSE(failureSet.Contains(2));
    for (size_t i = 0; i < number; i++) { ASSERT_EQ(host1[i], host2[i]); }
}

TEST_F(UCTsfTaskQueueTest, TransBetweenDeviceAndSSD)
{
    constexpr size_t blockSize = sizeof(size_t);
    constexpr size_t number = 1024;
    size_t host1[number];
    size_t host2[number];
    UC::SpaceManager spaceMgr;
    ASSERT_TRUE(spaceMgr.Setup({this->Path()}, blockSize).Success());
    UC::TsfTaskSet failureSet;
    UC::TsfTaskQueue q;
    ASSERT_TRUE(q.Setup(0, 128, blockSize, number / 2, &failureSet, spaceMgr.GetSpaceLayout()).Success());
    UC::TsfTask task1(UC::TsfTask::Type::DUMP, UC::TsfTask::Location::DEVICE, "D2S");
    UC::TsfTask task2(UC::TsfTask::Type::LOAD, UC::TsfTask::Location::DEVICE, "S2D");
    UC::Random rd;
    std::vector<std::string> blockIds(number);
    for (size_t i = 0; i < number; i++) {
        host1[i] = i;
        host2[i] = 0;
        blockIds[i] = rd.RandomString(16);
        ASSERT_TRUE(spaceMgr.NewBlock(blockIds[i]).Success());
        ASSERT_TRUE(task1.Append(blockIds[i], 0, (uintptr_t)(host1 + i), blockSize).Success());
        ASSERT_TRUE(task2.Append(blockIds[i], 0, (uintptr_t)(host2 + i), blockSize).Success());
    }
    UC::TsfTaskWaiter waiter1{1};
    task1.id = 1;
    task1.waiter = &waiter1;
    q.Push(std::move(task1));
    waiter1.Wait();
    ASSERT_FALSE(failureSet.Contains(1));
    for (size_t i = 0; i < number; i++) { spaceMgr.CommitBlock(blockIds[i], true); }
    UC::TsfTaskWaiter waiter2{1};
    task2.id = 2;
    task2.waiter = &waiter2;
    q.Push(std::move(task2));
    waiter2.Wait();
    ASSERT_FALSE(failureSet.Contains(2));
    for (size_t i = 0; i < number; i++) { ASSERT_EQ(host1[i], host2[i]); }
}
