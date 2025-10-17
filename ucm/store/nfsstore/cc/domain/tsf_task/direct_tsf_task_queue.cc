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
#include "direct_tsf_task_queue.h"
#include "file/file.h"
#include "logger/logger.h"
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

namespace UC {

#define UC_TASK_ERROR(s, t)                                                                        \
    do {                                                                                           \
        UC_ERROR("Failed({}) to run task({},{},{},{}).", (s), (t).owner, (t).blockId, (t).offset,  \
                 (t).length);                                                                      \
    } while (0)


Status DirectTsfTaskQueue::Setup(const int32_t deviceId, const size_t bufferSize,
                           const size_t bufferNumber, TsfTaskSet* failureSet,
                           const SpaceLayout* layout, bool transferUseDirect)
{
    this->_failureSet = failureSet;
    this->_layout = layout;
    if (deviceId >= 0) {
        this->_device = DeviceFactory::Make(deviceId, bufferSize, bufferNumber);
        if (!this->_device) { return Status::OutOfMemory(); }
        if (!this->_device->Setup(transferUseDirect).Success()) { return Status::Error(); }
    }
    const size_t nWorkers = 8; 
    if (!this->_directOper.Setup([this](TsfTask& task) { this->DirectOper(task); },
                            [] { return true; }, [] {}, nWorkers)) {
        UC_ERROR("Failed to setup GDS operation thread pool");
        return Status::Error();
    }

    return Status::OK();
}

Status DirectTsfTaskQueue::InitializeCuFile() {
    CUfileError_t cuFileErr = cuFileDriverOpen();
    if (cuFileErr.err != CU_FILE_SUCCESS) {
        UC_ERROR("Failed to open cuFile driver: {}", static_cast<int>(cuFileErr.err));
        return Status::Error();
    }

    UC_INFO("cuFile driver open");
    return Status::OK();
}

void DirectTsfTaskQueue::Push(std::list<TsfTask>& tasks)
{
    this->_directOper.Push(tasks);
}

void DirectTsfTaskQueue::DirectOper(TsfTask& task) {
    if (this->_failureSet->Contains(task.owner)) {
        this->Done(task, false);
        return;
    }
    if (task.type == TsfTask::Type::DUMP) {
        D2S(task);
    } else {
        S2D(task);
    }
}

void DirectTsfTaskQueue::S2D(TsfTask& task) {
    auto filePath = _layout->DataFilePath(task.blockId, false);
    auto status = this->_device->S2D(filePath, (void*)task.address, task.length, task.offset, 0);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        Done(task, false);
        return;
    } 
    Done(task, true);
}

void DirectTsfTaskQueue::D2S(TsfTask& task) {
    auto filePath = _layout->DataFilePath(task.blockId, true);
    auto status = this->_device->D2S(filePath, (void*)task.address, task.length, task.offset, 0);
    if (status.Failure()) {
        UC_TASK_ERROR(status, task);
        Done(task, false);
        return;
    } 
    Done(task, true);
}

void DirectTsfTaskQueue::Done(const TsfTask& task, bool success) {
    if (!success) {
        _failureSet->Insert(task.owner);
    }
    task.waiter->Done();
}

} // namespace UC