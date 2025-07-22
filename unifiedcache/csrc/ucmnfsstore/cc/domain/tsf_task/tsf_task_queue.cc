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

#include <tsf_task/tsf_task_queue.h>
#include "device.h" 

namespace UC{

TsfTaskQueue::~TsfTaskQueue()
{
    {
        std::unique_lock<std::mutex> lock(this->_mutex);
        if(!this->_running){
            return;
        }
        this->_running = false;
    }
    if(this->_worker.joinable()){
        this->_cv.notify_all();
        this->_worker.join();
    }
}

Status TsfTaskQueue::Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber, TsfTaskSet* failureSet)
{
    this->_deviceId = deviceId;
    this->_bufferSize = bufferSize;
    this->_bufferNumber = bufferNumber;
    this->_failureSet = failureSet;
    {
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_running = true;
    }
    auto fut = this ->_started.get_future();
    this->_worker = std::thread([&]{this->Worker();});
    return fut.get();
}

void TsfTaskQueue::Push(std::list<TsfTask>& tasks)
{
    {
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_q.splice(this->_q.end(),tasks);
    }
    this->_cv.notify_one();
}

void TsfTaskQueue::Push(TsfTask&& task)
{
    {
        std::unique_lock<std::mutex> lock(this->_mutex);
        this->_q.push_back(task);
    }
    this->_cv.notify_one();
}

bool TsfTaskQueue::Finish(const size_t& taskId) const
{
    if (this->_lastId > taskId){
        return true;
    }
    if(this->_q.empty()){
        return true;
    }
    return false;
}

void TsfTaskQueue::Worker()
{
    std::unique_ptr<IDevice> device = nullptr;
    if (this -> _deviceId >=0 ){
        device = Device::Make(this->_deviceId, this->_bufferSize, this -> _bufferNumber);
        if (!device){
            this->_started.set_value(Status::OutOfMemory());
            return;
        }
        auto status = device->Setup();
        if (status.Failure()){
            this -> _started.set_value(status);
            return;
        }
    }
    this->_started.set_value(Status::OK());
    for (;;){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_cv.wait(lock, [this] {return !this->_running || !this->_q.empty();});
            if(!this->_running){
                return;
            }
        }
        if (this->_q.empty()){
            continue;
        }
        const auto& task = this->_q.front();
        if (!this->_failureSet->Exist(task.owner)){
            auto status = this->_runner.Run(task, device.get());
            if (status.Failure()){
                this->_failureSet->Insert(task.owner);
            }
        }
        if (device && (this->_lastId != task.owner || this->_q.size()==1)){
            auto status = Status::OK(); //device->WaitFinish(); // todo ssd2device
            if (status.Failure()){
                this->_failureSet->Insert(task.owner);
            }
        }
        this->_lastId = task.owner;
        this->_q.pop_front();
    }
}
}
