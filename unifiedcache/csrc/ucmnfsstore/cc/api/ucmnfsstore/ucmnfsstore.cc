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
#include "ucmnfsstore.h"
#include "template/singleton.h"
#include "tsf_task/tsf_task_manager.h"

namespace UC {

int32_t Setup(const SetupParam& param)
{
    return 0;
}

int32_t Alloc(const std::string& blockId)
{
    return 0;
}

bool Lookup(const std::string& blockId)
{
    return false;
}

size_t Submit(std::list<TsfTask> tasks)
{
    auto& taskMgr = Singleton<TsfTaskManager>::Instance();
    size_t taskId;
    if(taskMgr.SubmitTask(tasks, taskId).Failure()){
        return TRANSFER_INVALID_TASK_ID;
    }
    return taskId;
}

int32_t Wait(const size_t taskId)
{
    auto& taskMgr = Singleton<TsfTaskManager>::Instance();
    auto status = taskMgr.GetStatus(taskId);
    if (status == TsfTaskStatus::RUNNING) {
        // wait for the task to finish
        // todo
        while (status == TsfTaskStatus::RUNNING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            status = taskMgr.GetStatus(taskId);
        }
    }
    return 0;
}

void Commit(const std::string& blockId, const bool success)
{
}

} // namespace UC
