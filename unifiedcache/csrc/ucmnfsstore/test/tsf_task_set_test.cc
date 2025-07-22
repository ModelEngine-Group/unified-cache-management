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
 #include "tsf_task/tsf_task_set.h"
 #include <vector>
 #include <thread>

class TsfTaskSetUnitTest : public ::testing::Test {};
/**
 * @brief Test case for single TSF task set.
**/
TEST_F(TsfTaskSetUnitTest, TsfTaskSetSingle)
{
    UC::TsfTaskSet runningSet;
    const size_t taskId = 1;

    ASSERT_FALSE(runningSet.Exist(taskId));
    runningSet.Remove(taskId);

    runningSet.Insert(taskId);
    ASSERT_TRUE(runningSet.Exist(taskId));

    runningSet.Insert(taskId);
    ASSERT_TRUE(runningSet.Exist(taskId));

    runningSet.Remove(taskId);
    ASSERT_FALSE(runningSet.Exist(taskId));
}

/**
 * @brief Test case for multi-threaded TSF task set.
**/
TEST_F(TsfTaskSetUnitTest, TsfTaskSetMultiThread)
{
    UC::TsfTaskSet runningSet;
    const size_t THREAD_NUM = 8;
    const size_t INSERT_PER_THREAD = 1000;
    const size_t REMOVE_PER_THREAD = 1000;

    // Insert tasks in multiple threads
    auto insertTask = [&](size_t start){
        for (size_t i = 0; i < start + INSERT_PER_THREAD; ++i) {
            size_t taskId = start + i;
            runningSet.Insert(taskId);
        }
    };
    std::vector<std::thread> threadsInsert;
    for (size_t i = 0; i < THREAD_NUM; ++i) {
        threadsInsert.emplace_back(insertTask, i * INSERT_PER_THREAD);
    }

    for (auto& thread : threadsInsert) {
        thread.join();
    }

    for (size_t i = 0; i < THREAD_NUM * INSERT_PER_THREAD; ++i) {
        ASSERT_TRUE(runningSet.Exist(i));
    }

    // Remove tasks in multiple threads
    auto removeTask = [&](size_t start){
        for (size_t i = 0; i < start + REMOVE_PER_THREAD; ++i) {
            size_t taskId = start + i;
            runningSet.Remove(taskId);
        }
    };
    std::vector<std::thread> threadsRemove;
    for (size_t i = 0; i < THREAD_NUM; ++i) {
        threadsRemove.emplace_back(removeTask, i * REMOVE_PER_THREAD);
    }

    for (auto& thread : threadsRemove) {
        thread.join();
    }

    for (size_t i = 0; i < THREAD_NUM * REMOVE_PER_THREAD; ++i) {
        ASSERT_FALSE(runningSet.Exist(i));
    }
}
