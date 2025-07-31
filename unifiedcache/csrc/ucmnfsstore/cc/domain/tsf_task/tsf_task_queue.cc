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
#include "device/device.h" 
#include "tsf_task_runner.h"

namespace UC{

TsfTaskQueue::~TsfTaskQueue()
{
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        if (!this->_running) { return; }
        this->_running = false;
    }
    if (this->_worker.joinable()){
        this->_cv.notify_all();
        this->_worker.join();
    }
}

Status TsfTaskQueue::Setup(const int32_t deviceId, const size_t ioSize, TsfTaskSet* failureSet)
{
    this->_deviceId = deviceId;
    this->_ioSize = ioSize;
    this->_failureSet = failureSet;
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_running = true;
    }
    std::promise<Status> started;
    auto fut = started.get_future();
    this->_worker = std::thread([&]{ this->Worker(started); });
    return fut.get();
}

void TsfTaskQueue::Push(std::list<TsfTask>& tasks)
{
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_taskQ.splice(this->_taskQ.end(), tasks);
    }
    this->_cv.notify_all();
}

void TsfTaskQueue::Worker(std::promise<Status>& started)
{
    auto status = Status::OK();
    std::unique_ptr<IDevice> device = nullptr;
    if (this -> _deviceId >= 0){
        if (!(device = Device::Make(this->_deviceId, this->_ioSize))){
            started.set_value(Status::OutOfMemory());
            return;
        }
        if ((status = device->Setup()).Failure()){
            started.set_value(status);
            return;
        }
    }
    TsfTaskRunner runner(device ? device.get() : nullptr);
    started.set_value(status);
    for(;;){
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_cv.wait(lk, [this] { return !this->_taskQ.empty() || !this->_running; });
        if (!this->_running) { return; }
        if (this->_taskQ.empty()) { continue; }
        auto task = std::move(this->_taskQ.front());
        this->_taskQ.pop_front();
        bool lastTask = this->_taskQ.empty() || this->_taskQ.front().owner != task.owner;
        lk.unlock();
        if (!this->_failureSet->Exist(task.owner)) {
            if ((status = runner.Run(task)).Failure()) {
                UC_ERROR("Failed({}) to run transfer task({}).", status, task.owner);
                this->_failureSet->Insert(task.owner);
            }
        }
        if (device && lastTask) {
            if ((status = device->WaitFinish()).Failure()) { this->_failureSet->Insert(task.owner); }
        }
        task.waiter->Done();
    }
}

} // namespace UC
