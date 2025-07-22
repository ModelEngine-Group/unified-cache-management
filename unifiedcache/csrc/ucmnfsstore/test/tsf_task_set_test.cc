/**
/* MIT License
/*
/* Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
/*
/* Permission is hereby granted, free of charge, to any person obtaining a copy
/* of this software and associated documentation files (the "Software"), to deal
/* in the Software without restriction, including without limitation the rights
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/* copies of the Software, and to permit persons to whom the Software is
/* furnished to do so, subject to the following conditions:
/*
/* The above copyright notice and this permission notice shall be included in all
/* copies or substantial portions of the Software.
/*
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/* SOFTWARE.
 * */

 #include <gtest/gtest.h>
 #include "tsf_task/tsf_task_set.h"
 #include <vector>
 #include <thread>

class TsfTaskSetTest : public ::testing::Test {};
/**
 * @brief Test case for single TSF task set.
**/
TEST_F(TsfTaskSetTest, TsfTaskSetSingle)
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
