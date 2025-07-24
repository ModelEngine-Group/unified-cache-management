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
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR O THER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_TSF_TASK_MANAGER
#define UNIFIEDCACHE_TSF_TASK_MANAGER

#include "tsf_task_queue.h"
#include <unordered_map>

namespace UC {

class TsfTaskManager{
public:
    Status Setup();
    Status SubmitTask(std::list<TsfTask>tasks, size_t& taskId){
        Status status;
        std::unique_lock<std::mutex> lock(this->_mutex);
        taskId = ++this->_idSeed;
        size_t number = 0;
        size_t size = 0;
        std::string brief;
        auto qSize = _queues.size();
        std::list<TsfTask> lists[qSize];
        auto [iter, success] = this->_tasks.emplace(taskId, TsfTaskGroup());
        if (!success){
            UC_ERROR("Failed to insert tsaks({}) into set.", taskId);
            return Status::OutOfMemory();
        }
        status = this ->MakeTasks(tasks, lists, taskId, qSize, number, size, brief);
        if (status.Failure()) {
            return status;
        }
        for (size_t i = 0; i < qSize; ++i) {
            if (lists[i].empty()) {
                continue;
            }
            auto& q = this->_queues[this->_qIdx];
            q->Push(std::move(lists[i].front()));
            this->_qIdx = (this->_qIdx + 1) % qSize;
        }
        iter->second.Set(taskId, brief, number, size);
        return status;
    };
    TsfTaskStatus GetStatus(const size_t& taskId);
    void Cancel(const size_t& taskId);
    
private:
    Status Precheck();
    Status MakeTasks(std::list<TsfTask>tasks, std::list<TsfTask> lists[], 
                     const size_t& taskId, const size_t& qSize, size_t& number, size_t& size, std::string& brief)
    {
        for (auto& task : tasks) {
            number++;
            brief = this->GetTaskBrief(task);
            if (brief.empty()) {
                UC_ERROR("Unsupported task ({}-{}-{}-{}-{}) with action({}-{})", task.blockId, task.offset, task.address, task.length, fmt::underlying(task.type), fmt::underlying(task.location));
                return Status::Unsupported();
            }
            if(task.length==0){
                UC_ERROR("Invalid task ({}-{}-{}-{}-{}) with action({}-{})", task.blockId, task.offset, task.address, task.length, fmt::underlying(task.type), fmt::underlying(task.location));
                return Status::InvalidParam();
            }
            task.owner = taskId;
            lists[number % qSize].emplace_back(std::move(task));
            size += task.length;
        }
        return Status::OK();
    }
    std::string GetTaskBrief(TsfTask& task);
    void Remove(std::unordered_map<size_t, TsfTaskGroup>::iterator iter);
    bool Finish(const size_t& id) const;

private:
    std::mutex _mutex;
    TsfTaskSet _failureSet;
    std::unordered_map<size_t, TsfTaskGroup> _tasks;
    size_t _idSeed{0};
    size_t _maxFinishedId{0};
    std::vector<std::unique_ptr<TsfTaskQueue>> _queues;
    size_t _qIdx{0};
};

} // namespace UC

#endif
