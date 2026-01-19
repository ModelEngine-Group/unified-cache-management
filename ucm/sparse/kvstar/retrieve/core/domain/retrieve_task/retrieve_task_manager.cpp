#include "retrieve_task_manager.h"

namespace KVStar {
Status RetrieveTaskManager::Setup(const size_t threadNum,
                                  const std::vector<std::pair<int, int>>& bindInfo)
{
    if (threadNum != bindInfo.size()) {
        UC_ERROR("Thread count ({}) does not match the size of bind-core-ID list ({}).", threadNum,
                 bindInfo.size());
        return Status::InvalidParam();
    }

    this->_queues.reserve(threadNum);
    for (size_t i = 0; i < threadNum; ++i) {
        const int targetCoreId = bindInfo[i].first;
        const int targetNumaId = bindInfo[i].second;

        auto& queue = this->_queues.emplace_back(std::make_unique<RetrieveTaskQueue>());
        auto status = queue->Setup(targetNumaId, targetCoreId, &this->_failureSet);
        if (status.Failure()) {
            UC_ERROR("Init and setup thread id {} (to core {}) in pool failed.", i, targetCoreId);
            return status;
        }
        UC_DEBUG("Init and setup thread id {} in pool to core {} success.", i, targetCoreId);
    }
    return Status::OK();
}

Status RetrieveTaskManager::SubmitSingleTask(RetrieveTask&& task, size_t& taskId)
{
    std::unique_lock<std::mutex> lk(this->_mutex);
    taskId = ++this->_taskIdSeed;
    UC_DEBUG("Retrieve task manager allocate id to task: {}.", taskId);
    auto [waiter_iter, success1] =
        this->_waiters.emplace(taskId, std::make_shared<RetrieveTaskWaiter>(taskId, 1));
    if (!success1) { return Status::OutOfMemory(); }

    auto resultPtr = std::make_shared<TaskResult>();
    auto [result_iter, success2] = this->_resultMap.emplace(taskId, resultPtr);
    if (!success2) {
        this->_waiters.erase(waiter_iter);
        return Status::OutOfMemory();
    }

    task.allocTaskId = taskId;
    task.waiter = waiter_iter->second;
    UC_DEBUG("Set task id to retrieve task waiter success.");

    this->_queues[this->_lastTimeScheduledQueueIdx]->Push({std::move(task), resultPtr});

    UC_DEBUG("Push task and set task scheduled queue idx success, queue idx: {}.",
             this->_lastTimeScheduledQueueIdx);

    this->_lastTimeScheduledQueueIdx =
        (this->_lastTimeScheduledQueueIdx + 1) % this->_queues.size();

    return Status::OK();
}

Status RetrieveTaskManager::Wait(const size_t taskId)
{
    std::shared_ptr<RetrieveTaskWaiter> waiter = nullptr;
    {  // lock area
        std::unique_lock<std::mutex> lk(this->_mutex);
        auto iter = this->_waiters.find(taskId);
        if (iter == this->_waiters.end()) { return Status::NotFound(); }
        waiter = iter->second;
        this->_waiters.erase(iter);
    }
    waiter->Wait();
    bool failure = this->_failureSet.Exist(taskId);
    this->_failureSet.Remove(taskId);
    if (failure) { UC_ERROR("Retrieve task({}) failed.", taskId); }
    return failure ? Status::Error() : Status::OK();
}

Status RetrieveTaskManager::GetResult(size_t taskId, std::shared_ptr<TaskResult>& result)
{
    std::unique_lock<std::mutex> lk(this->_mutex);
    auto it = _resultMap.find(taskId);
    if (it == _resultMap.end()) { return Status::NotFound(); }
    result = it->second;
    return Status::OK();
}

}  // namespace KVStar
