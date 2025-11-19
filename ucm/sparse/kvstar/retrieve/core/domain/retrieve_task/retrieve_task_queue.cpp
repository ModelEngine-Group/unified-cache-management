#ifdef NUMA_ENABLED
#include <numaif.h>
#endif
#include "retrieve_task_queue.h"
#include "retrieve_task_runner.h"

namespace KVStar {
RetrieveTaskQueue::~RetrieveTaskQueue() {
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

void RetrieveTaskQueue::Worker(const int numaId, const int bindCoreId, std::promise<Status>& started) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(bindCoreId, &cpuset);
    pthread_t thread = pthread_self();
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        perror("pthread_setaffinity_np");
        started.set_value(Status::OsApiError());
        return;
    }

#ifdef NUMA_ENABLED
    unsigned long nodemask = 1UL << numaId;
    rc = set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
    if (rc != 0) {
        perror("set_mempolicy");
        started.set_value(Status::OsApiError());
        return;
    }
#else
    KVSTAR_DEBUG("NUMA support is disabled.");
#endif

    KVSTAR_DEBUG("Bind current thread {} to numa {} core {} and set memory affinity success.", thread, numaId, bindCoreId);
    RetrieveTaskRunner runner;

    started.set_value(Status::OK());

    Status status = Status::OK();

    for(;;){
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_cv.wait(lk, [this] { return !this->_taskQ.empty() || !this->_running; });
        if (!this->_running) { return; }
        if (this->_taskQ.empty()) { continue; }

        auto workItem = std::move(this->_taskQ.front());
        this->_taskQ.pop_front();
        lk.unlock();

        workItem.result->status = TaskStatus::RUNNING;

        if (!_failureSet->Exist(workItem.task.allocTaskId)) {
            if ((status = runner.Run(workItem.task, *workItem.result)).Failure()) {
                KVSTAR_ERROR("Failed({}) to run retrieve task({}).", status, workItem.task.allocTaskId);
                this->_failureSet->Insert(workItem.task.allocTaskId);
                workItem.result->status = TaskStatus::FAILURE;
            } else {
                KVSTAR_DEBUG("Process current task success, task id: {}.", workItem.task.allocTaskId);
                workItem.result->status = TaskStatus::SUCCESS;
            }
        }

        workItem.task.waiter->Done();
    }

}


Status RetrieveTaskQueue::Setup(const int numaId, const int bindCoreId, RetrieveTaskSet* failureSet) {
    this->_failureSet = failureSet;
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_running = true;
    }
    std::promise<Status> started;
    auto fut = started.get_future();
    this->_worker = std::thread([&]{ this->Worker(numaId, bindCoreId, started); });
    return fut.get();
}

void RetrieveTaskQueue::Push(WorkItem&& item) {
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        this->_taskQ.push_back(std::move(item));
    }
    this->_cv.notify_one();
}


}