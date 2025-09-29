#ifndef UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_QUEUE_H
#define UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_QUEUE_H

#include <condition_variable>
#include <future>
#include <list>
#include <mutex>
#include <thread>
#include "status/status.h"

#include "retrieve_task.h"
#include "retrieve_task_set.h"
#include "task_result.h"

namespace KVStar {
// 定义一个工作项，将任务和结果容器绑定
struct WorkItem {
    RetrieveTask task;
    std::shared_ptr<TaskResult> result;
};

class RetrieveTaskQueue {
public:
    ~RetrieveTaskQueue();
    Status Setup(const int numaId, const int bindCoreId, RetrieveTaskSet* failureSet); // failureSet from manager, for all queue
    void Push(WorkItem&& item);

private:
    void Worker(const int numaId, const int bindCoreId, std::promise<Status>& started);

private:
    std::list<WorkItem> _taskQ; // 任务队列
    std::mutex _mutex;
    std::condition_variable _cv;
    std::thread _worker;
    bool _running{false};
    RetrieveTaskSet* _failureSet;


};
}



#endif //UCM_SPARSE_KVSTAR_RETRIEVE_RETRIEVE_TASK_QUEUE_H