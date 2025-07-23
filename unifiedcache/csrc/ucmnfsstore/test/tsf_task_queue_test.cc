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
/**
 * @brief tsf task queue unit test
 */
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tsf_task/tsf_task_queue.h"

class TsfTaskQueueUnitTest : public ::testing::Test {};

TEST_F(TsfTaskQueueUnitTest, Setup)
{
    UC::TsfTaskQueue taskQueue;
    UC::TsfTaskSet failureSet;
    auto status = taskQueue.Setup(-1, 0, 0, &failureSet);
    ASSERT_TRUE(status.Success());
    constexpr size_t taskId = 1;        // owner初始化为0， 必须等待manager给他分配，跟原来的逻辑不同
    ASSERT_FALSE(failureSet.Exist(taskId));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).expects(once()).will(returnValue(UC::Status::Error()));
    constexpr size_t nTasks = 10;
    constexpr size_t dataSize = 4096;
    for (size_t i = 0; i < nTasks; i++) {
        UC::TsfTask task{UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST, 
                            "blockId", 0, 0, dataSize, taskId
        };
        taskQueue.Push(std::move(task));
    }
    while (!taskQueue.Finish(taskId)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    GlobalMockObject::verify();
    ASSERT_TRUE(failureSet.Exist(taskId));
}

TEST_F(TsfTaskQueueUnitTest, TaskSuccess)
{
    UC::TsfTaskQueue taskQueue;
    UC::TsfTaskSet failureSet;
    auto status = taskQueue.Setup(-1, 0, 0, &failureSet);
    ASSERT_TRUE(status.Success());
    constexpr size_t taskId = 1;
    constexpr size_t nTasks = 10;
    constexpr size_t dataSize = 4096;
    ASSERT_FALSE(failureSet.Exist(taskId));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).expects(exactly(nTasks)).will(returnValue(UC::Status::OK()));
    for (size_t i = 0; i < nTasks; i++) {
        UC::TsfTask task{
            UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST,"blockId",0, dataSize * i, dataSize, taskId
        };
        taskQueue.Push(std::move(task));
    }
    while (!taskQueue.Finish(taskId)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    GlobalMockObject::verify();
    ASSERT_FALSE(failureSet.Exist(taskId));
}
