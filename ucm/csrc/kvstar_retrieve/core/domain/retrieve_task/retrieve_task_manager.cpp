#include <spdlog/fmt/ranges.h>
#include "retrieve_task_manager.h"

namespace KVStar {
Status RetrieveTaskManager::Setup(const size_t threadNum, const std::vector<int>& cpuNumaIds, const std::vector<std::vector<int>>& bindCoreId) {
    // 一些必要检测
    const size_t numaNodeCount = cpuNumaIds.size();
    if (numaNodeCount == 0) {
        KVSTAR_ERROR("Retrieve task manager get error numa id info {}.", cpuNumaIds);
        return Status::InvalidParam();
    }

    if (threadNum % numaNodeCount != 0) {
        KVSTAR_ERROR("Retrieve task manager can not split threads into each numa, thread num {}, numa id info {}.", threadNum, cpuNumaIds);
        return Status::InvalidParam();
    }

    if (bindCoreId.size() != numaNodeCount) {
        KVSTAR_ERROR("Bind core ids {} can not match numa id info {}.", bindCoreId, cpuNumaIds);
        return Status::InvalidParam();
    }

    const size_t threadsPerNuma = threadNum / numaNodeCount;

    this->_queues.reserve(threadNum);
    for (size_t i = 0; i < threadNum; ++i) {
        // 1. 计算当前线程 i 应该落在哪个 NUMA 节点的索引上
        const size_t numaListIndex = i / threadsPerNuma;

        // 2. 计算当前线程 i 在其 NUMA 节点内部的核心列表中的索引
        const size_t coreListIndex = i % threadsPerNuma;

        // 安全检查：确保 bindCoreId 内部的向量大小是足够的
        if (coreListIndex >= bindCoreId[numaListIndex].size()) {
            KVSTAR_ERROR("Bind core ids {} can not alloc per numa need alloc threads num {}.", bindCoreId, threadsPerNuma);
            return Status::InvalidParam();
        }

        // 3. 提取最终的 numaId 和 coreId
        const int targetNumaId = cpuNumaIds[numaListIndex];
        const int targetCoreId = bindCoreId[numaListIndex][coreListIndex];

        auto& queue = this->_queues.emplace_back(std::make_unique<RetrieveTaskQueue>()); // emplace_back 会返回一个指向刚刚被创建的那个新元素的引用
        auto status = queue->Setup(targetNumaId, targetCoreId, &this->_failureSet);
        if (status.Failure()) {
            KVSTAR_ERROR("Init and setup thread id {} in pool failed.", i);
            return status;
        }
        KVSTAR_DEBUG("Init and setup thread id {} in pool success.", i);
    }
    return Status::OK();
}

// 提交单个task无需dispatch
Status RetrieveTaskManager::SubmitSingleTask(RetrieveTask&& task, size_t &taskId)
{
    std::unique_lock<std::mutex> lk(this->_mutex);
    taskId = ++this->_taskIdSeed;
    KVSTAR_DEBUG("Retrieve task manager allocate id to task: {}.", taskId);
    auto [waiter_iter, success1] = this->_waiters.emplace(taskId, std::make_shared<RetrieveTaskWaiter>(taskId, 1));
    if (!success1) { return Status::OutOfMemory(); }

    // 2. 创建并存储 Result 容器
    auto resultPtr = std::make_shared<TaskResult>();
    auto [result_iter, success2] = this->_resultMap.emplace(taskId, resultPtr);
    if (!success2) {
        this->_waiters.erase(waiter_iter); // 如果失败，回滚
        return Status::OutOfMemory();
    }

    // 3. 将 waiter 关联到 task
    task.allocTaskId = taskId;
    task.waiter = waiter_iter->second;
    KVSTAR_DEBUG("Set task id to retrieve task waiter success.");

    // 4. 将 task 和 resultPtr 打包成 WorkItem 推入队列
    this->_queues[this->_lastTimeScheduledQueueIdx]->Push({std::move(task), resultPtr});

    KVSTAR_DEBUG("Push task and set task scheduled queue idx success, queue idx: {}.", this->_lastTimeScheduledQueueIdx);

    this->_lastTimeScheduledQueueIdx = (this->_lastTimeScheduledQueueIdx + 1) % this->_queues.size();

    return Status::OK();
}

Status RetrieveTaskManager::Wait(const size_t taskId) {
    std::shared_ptr<RetrieveTaskWaiter> waiter = nullptr;
    { // lock area
        std::unique_lock<std::mutex> lk(this->_mutex);
        auto iter = this->_waiters.find(taskId);
        if (iter == this->_waiters.end()) {
            return Status::NotFound();
        }
        waiter = iter->second;
        this->_waiters.erase(iter);
    }
    waiter->Wait();
    bool failure = this->_failureSet.Exist(taskId);
    this->_failureSet.Remove(taskId);
    if (failure) {
        KVSTAR_ERROR("Retrieve task({}) failed.", taskId);
    }
    return failure ? Status::Error() : Status::OK();
}

// GetResult 的实现
Status RetrieveTaskManager::GetResult(size_t taskId, std::shared_ptr<TaskResult>& result) {
    std::unique_lock<std::mutex> lk(this->_mutex);
    auto it = _resultMap.find(taskId);
    if (it == _resultMap.end()) {
        return Status::NotFound();
    }
    result = it->second;
    return Status::OK();
}


}