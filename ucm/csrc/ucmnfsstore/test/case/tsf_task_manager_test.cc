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
#include "device/idevice.h"
#include "space/space_manager.h"
#include "tsf_task/tsf_task_manager.h"

class UCTsfTaskManagerTest : public UC::PathBase {};

TEST_F(UCTsfTaskManagerTest, SameTaskWaitTwice)
{
    constexpr size_t blockSize = sizeof(size_t);
    constexpr size_t number = 1;
    size_t host[number];
    UC::SpaceManager spaceMgr;
    ASSERT_TRUE(spaceMgr.Setup({this->Path()}, blockSize).Success());
    UC::TsfTaskManager transMgr;
    ASSERT_TRUE(transMgr.Setup(-1, 128, 0, 0, 0, spaceMgr.GetSpaceLayout()).Success());
    UC::TsfTask task(UC::TsfTask::Type::DUMP, UC::TsfTask::Location::HOST, "H2S");
    UC::Random rd;
    for (size_t i = 0; i < number; i++) {
        host[i] = i;
        auto blockId = rd.RandomString(16);
        ASSERT_TRUE(spaceMgr.NewBlock(blockId).Success());
        ASSERT_TRUE(task.Append(blockId, 0, (uintptr_t)(host + i), blockSize).Success());
    }
    size_t taskId = 0;
    ASSERT_TRUE(transMgr.Submit(std::move(task), taskId).Success());
    ASSERT_GT(taskId, 0);
    ASSERT_TRUE(transMgr.Wait(taskId).Success());
    ASSERT_TRUE(transMgr.Wait(taskId).Failure());
}
