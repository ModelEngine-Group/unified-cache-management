#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_MANAGER_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_MANAGER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "retrieve_task_queue.h"
#include "task_result.h" // 引入结果定义

namespace KVStar {
class RetrieveTaskManager {
public:
    Status Setup(const size_t threadNum, const std::vector<int>& cpuNumaIds, const std::vector<std::vector<int>>& bindCoreId); // 重要, 线程池拉起的入口
    Status SubmitSingleTask(RetrieveTask&&task, size_t &taskId);

    // --- 获取结果数据的函数 ---
    Status GetResult(size_t taskId, std::shared_ptr<TaskResult>& result);

    Status Wait(const size_t taskId);
private:
    void Dispatch();

private:
    std::mutex _mutex;
    RetrieveTaskSet _failureSet;
    std::unordered_map<size_t, std::shared_ptr<RetrieveTaskWaiter>> _waiters;

    // --- 用于存储结果的 Map ---
    std::unordered_map<size_t, std::shared_ptr<TaskResult>> _resultMap;

    std::vector<std::unique_ptr<RetrieveTaskQueue>> _queues;
    size_t _lastTimeScheduledQueueIdx{0};
    size_t _taskIdSeed{0};

};

}



#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_MANAGER_H