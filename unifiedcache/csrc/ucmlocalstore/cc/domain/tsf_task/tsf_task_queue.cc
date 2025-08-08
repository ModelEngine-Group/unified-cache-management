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

#include "tsf_task_queue.h"
#include "cache/cache_manager.h"
#include "device/device.h"
#include "file/file.h"
#include "template/singleton.h"

namespace UCM {

#define UCM_TASK_ERROR(s, t)                                                                                            \
    do {                                                                                                               \
        UCM_ERROR("Failed({}) to run cache task({},{},{},{}).", (s), (t).owner, (t).blockId, (t).offset, (t).length);         \
    } while (0)

Status TsfTaskQueue::Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
{
    if (deviceId >= 0) {
        this->_device = Device::Make(deviceId, bufferSize, bufferNumber);
        if (!this->_device) { return Status::OutOfMemory(); }
        if (!this->_streamOper.Setup([this](TsfTask& task) { this->Host2Device(task); },
                                     [this] { return this->_device->Setup().Success(); })) {
            return Status::Error();
        }
    }
    if (!this->_cacheOper.Setup([this](TsfTask& task) { this->Cache2Host(task); })) { return Status::Error(); }
    return Status::OK();
}

void TsfTaskQueue::Push(std::list<TsfTask>& tasks)
{
    this->_cacheOper.Push(tasks);
}

void TsfTaskQueue::Cache2Host(TsfTask& task)
{
    if (task.buffer == nullptr) {
        task.waiter->Done();
        return;
    }
    if (!(task.hub = this->_device->GetBuffer(task.length))) {
        auto status = Status::OutOfMemory();
        UCM_TASK_ERROR(status, task);
        task.waiter->Done();
        return;
    }
    std::byte* src = (std::byte*)task.buffer + task.offset;
    std::memcpy(task.hub.get(), src, task.length);
    this->_streamOper.Push(std::move(task));
}

void TsfTaskQueue::Host2Device(TsfTask& task)
{
    auto status = this->_device->H2DAsync((std::byte*)task.address, task.hub.get(), task.length);
    if (status.Failure()) {
        UCM_TASK_ERROR(status, task);
        task.waiter->Done();
        return;
    }
    status = this->_device->AppendCallback([this, task](bool success) mutable {
        if (!success) { UCM_TASK_ERROR(Status::Error(), task); }
        task.waiter->Done();
        auto cacheMgr = Singleton<CacheManager>::Instance();
        cacheMgr->ReadDone(task.buffer);
    });
    if (status.Failure()) {
        UCM_TASK_ERROR(status, task);
        task.waiter->Done();
        return;
    }
}

} // namespace UCM
