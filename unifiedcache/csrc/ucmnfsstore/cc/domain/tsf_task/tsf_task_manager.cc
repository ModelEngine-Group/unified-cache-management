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
#include "tsf_task_manager.h"
#include <list>
#include <map>
#include <algorithm>
#include <thread>
#include "template/singleton.h"
#include "configurator.h"

namespace UC {

Status TsfTaskManager::Setup()
{
    auto configer = Singleton<Configurator>::Instance();
    auto deviceId = configer.DeviceId();
    auto streamNumber = configer.StreamNumber();
    auto bufferSize = configer.BufferSize();
    constexpr size_t bufferNumber = 2048; 
    for (size_t i = 0; i < streamNumber; ++i) {
        auto queue = std::make_unique<TsfTaskQueue>();
        auto status = queue->Setup(deviceId, bufferSize, bufferNumber, &this->_failureSet);
        if (status.Failure()) {
            return status;
        }
        this->_queues.emplace_back(std::move(queue));
    }
    return Status::OK();
}

TsfTaskStatus TsfTaskManager::GetStatus(const size_t& taskId) 
{
    auto configer = Singleton<Configurator>::Instance();
    auto timeout = configer.Timeout();
    std::unique_lock<std::mutex> lock(this->_mutex);
    auto iter = this->_tasks.find(taskId);
    if (iter == this->_tasks.end()) {
        UC_DEBUG("Not found task({}).", taskId);
        return TsfTaskStatus::CANCELLED;
    }
    auto elapsed = iter->second.Elapsed();
    if (this->Finish(taskId)) {
        auto failure = this->_failureSet.Exist(taskId);
        this->_failureSet.Remove(taskId);
        UC_INFO("Task({}) finish, failure={}, elapsed={:.6f}s.", iter->second.Str(), failure, elapsed);
        this->_tasks.erase(iter);
        if (this->_maxFinishedId < taskId) {
            this->_maxFinishedId = taskId;
        }
        return failure ? TsfTaskStatus::FAILURE : TsfTaskStatus::SUCCESS;
    } else {
        if (timeout != 0 && elapsed > timeout) {
            UC_ERROR("Task({}) timeout({:.6f}s > {}s).", iter->second.Str(), elapsed, timeout);
            this->Remove(iter);
            if (this->_maxFinishedId < taskId) {
                this->_maxFinishedId = taskId;
            }
            return TsfTaskStatus::FAILURE;
        }
        return TsfTaskStatus::RUNNING;
    }
}

void TsfTaskManager::Cancel(const size_t& taskId)
{
    std::unique_lock<std::mutex> lock(this->_mutex);
    auto iter = this->_tasks.find(taskId);
    if (iter == this->_tasks.end()){
        return;
    }
    this->Remove(iter);
}

Status TsfTaskManager::Precheck()
{
    auto configer = Singleton<Configurator>::Instance();
    if (this->_idSeed - this->_maxFinishedId >= configer.QueueDepth()){
        return Status::Retry(); 
    }
    return Status::OK();
}

std::string TsfTaskManager::GetTaskBrief(TsfTask& task)
{
    static std::map<std::pair<TsfTask::Type, TsfTask::Location>, std::string> briefs = {
        {{TsfTask::Type::DUMP, TsfTask::Location::HOST},    "Host2SSD"  },
        {{TsfTask::Type::LOAD, TsfTask::Location::HOST},    "SSD2Host"  },
        {{TsfTask::Type::DUMP, TsfTask::Location::DEVICE},  "Device2SSD"},
        {{TsfTask::Type::LOAD, TsfTask::Location::DEVICE},  "SSD2Device"},
    };
    auto brief = briefs.find({task.type, task.location});
    return brief == briefs.end() ? std::string() : brief->second;
}

void TsfTaskManager::Remove(std::unordered_map<size_t, TsfTaskGroup>::iterator iter)
{
    this->_failureSet.Insert(iter->first);
    constexpr auto interval = std::chrono::milliseconds(20);
    while (!this->Finish(iter->first)) {
        std::this_thread::sleep_for(interval);
    }
    this->_failureSet.Remove(iter->first);
    this->_tasks.erase(iter);
}

bool TsfTaskManager::Finish(const size_t& id) const
{

    for (const auto& q : this->_queues) {
        if (q->Finish(id)) {
            return true;
        }
    }
    return false;
} 
} // namespace UC
