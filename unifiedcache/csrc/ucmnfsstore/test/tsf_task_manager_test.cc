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
 * @brief tsf task manager unit test
 */
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "tsf_task/configurator.h"
#include "tsf_task/tsf_task_manager.h"

class TsfTaskManagerUnitTest : public testing::Test {
protected:
    std::list<UC::TsfTask> MakeTasks(const size_t& number, const uint32_t& dataSize)
    {
        std::list<UC::TsfTask> tasks;

        for (size_t i = 0; i < number; i++) {
            tasks.emplace_back(UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST, "1", 0, dataSize * i, dataSize);
        }
        return tasks;
    }
};

TEST_F(TsfTaskManagerUnitTest, SetupWhileTaskQueueFailed)
{
    UC::TsfTaskManager taskMgr;
    MOCKER_CPP(&UC::Configurator::StreamNumber, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(128));
    MOCKER_CPP(&UC::TsfTaskQueue::Setup).expects(once()).will(returnValue(UC::Status::Error()));
    ASSERT_TRUE(taskMgr.Setup().Failure());
    GlobalMockObject::verify();
}

TEST_F(TsfTaskManagerUnitTest, LotsOfTasks)
{
    constexpr size_t queueDepth = 10;
    constexpr size_t nQueues = 20;
    constexpr size_t nTasks = 2000;
    constexpr uint32_t dataSize = 4096;
    auto tasks = this->MakeTasks(nTasks, dataSize);
    ASSERT_EQ(tasks.size(), nTasks);

    UC::TsfTaskManager taskMgr;
    MOCKER_CPP(&UC::Configurator::StreamNumber, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(nQueues));
    ASSERT_TRUE(taskMgr.Setup().Success());
    GlobalMockObject::verify();
    MOCKER_CPP(&UC::Configurator::QueueDepth, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(queueDepth));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).stubs().will(returnValue(UC::Status::OK()));
    size_t taskId = 0;

    ASSERT_TRUE(taskMgr.SubmitTask(tasks, taskId).Success());
    ASSERT_GT(taskId, 0);
    auto taskStatus = taskMgr.GetStatus(taskId);
    while (taskStatus == UC::TsfTaskStatus::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        taskStatus = taskMgr.GetStatus(taskId);
    }
    ASSERT_EQ(taskStatus, UC::TsfTaskStatus::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TsfTaskManagerUnitTest, FewTasks)
{
    constexpr size_t queueDepth = 10;
    constexpr size_t nQueues = 512;
    constexpr size_t nTasks = 61;
    constexpr uint32_t dataSize = 4096;
    auto tasks = this->MakeTasks(nTasks, dataSize);
    ASSERT_EQ(tasks.size(), nTasks);

    UC::TsfTaskManager taskMgr;
    MOCKER_CPP(&UC::Configurator::StreamNumber, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(nQueues));
    ASSERT_TRUE(taskMgr.Setup().Success());
    GlobalMockObject::verify();
    MOCKER_CPP(&UC::Configurator::QueueDepth, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(queueDepth));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).stubs().will(returnValue(UC::Status::OK()));
    size_t taskId = 0;

    ASSERT_TRUE(taskMgr.SubmitTask(tasks, taskId).Success());
    ASSERT_GT(taskId, 0);
    auto taskStatus = taskMgr.GetStatus(taskId);
    while (taskStatus == UC::TsfTaskStatus::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        taskStatus = taskMgr.GetStatus(taskId);
    }
    ASSERT_EQ(taskStatus, UC::TsfTaskStatus::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TsfTaskManagerUnitTest, TooManyTasks)
{
    constexpr size_t queueDepth = 1;
    constexpr size_t nQueues = 512;
    constexpr size_t nTasks = 61;
    constexpr uint32_t dataSize = 4096;
    auto tasks = this->MakeTasks(nTasks, dataSize);
    ASSERT_EQ(tasks.size(), nTasks);

    UC::TsfTaskManager taskMgr;
    MOCKER_CPP(&UC::Configurator::StreamNumber, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(nQueues));
    ASSERT_TRUE(taskMgr.Setup().Success());
    GlobalMockObject::verify();
    MOCKER_CPP(&UC::Configurator::QueueDepth, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(queueDepth));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).stubs().will(returnValue(UC::Status::OK()));
    size_t taskId = 0;

    ASSERT_TRUE(taskMgr.SubmitTask(tasks, taskId).Success());
    ASSERT_GT(taskId, 0);
    auto taskStatus = taskMgr.GetStatus(taskId);
    while (taskStatus == UC::TsfTaskStatus::RUNNING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        taskStatus = taskMgr.GetStatus(taskId);
    }
    ASSERT_EQ(taskStatus, UC::TsfTaskStatus::SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TsfTaskManagerUnitTest, InvalidTasks)
{
    constexpr size_t queueDepth = 10;
    constexpr size_t nQueues = 512;
    constexpr size_t nTasks = 61;
    constexpr uint32_t dataSize = 4096;
    auto tasks = this->MakeTasks(nTasks, dataSize);
    ASSERT_EQ(tasks.size(), nTasks);

    tasks.emplace_back(UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST, "", 0, 0, 0);

    UC::TsfTaskManager taskMgr;
    MOCKER_CPP(&UC::Configurator::StreamNumber, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(nQueues));
    ASSERT_TRUE(taskMgr.Setup().Success());
    GlobalMockObject::verify();
    MOCKER_CPP(&UC::Configurator::QueueDepth, size_t (UC::Configurator::*)() const)
        .stubs()
        .will(returnValue(queueDepth));
    MOCKER_CPP(&UC::TsfTaskRunner::Run).stubs().will(returnValue(UC::Status::OK()));
    size_t taskId = 0;
    ASSERT_FALSE(taskMgr.SubmitTask(tasks, taskId).Success());
    GlobalMockObject::verify();
}
