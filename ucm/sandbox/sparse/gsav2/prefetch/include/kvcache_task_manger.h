/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <sched.h>
#include <thread>
#include <unordered_map>
#include <vector>
// Ascend is the only supported backend.  In a UT build (KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC)
// kvcache_trans_ascend.h switches to the test stub so no Ascend SDK is required.
#include "../ascend/kvcache_trans_ascend.h"

// Task Status Enumeration
enum class TaskStatus { PENDING, RUNNING, COMPLETED, CANCELLED };

// Task structure
struct TaskInfo {
    int taskID;
    int layerID;
    torch::Tensor topkList;
    std::vector<int> topkLenList;
    std::vector<int> bsInIndex;
    std::vector<std::string> reqIdList;
    std::vector<torch::Tensor> indexerTopkList;

    TaskInfo(int iTaskID, int iLayerID, torch::Tensor& iTopkList, std::vector<int>& iTopkLenList,
             std::vector<int>& iBsInIndex, std::vector<std::string>& iReqIdList)
        : taskID(iTaskID),
          layerID(iLayerID),
          topkList(iTopkList),
          topkLenList(iTopkLenList),
          bsInIndex(iBsInIndex),
          reqIdList(iReqIdList)
    {
    }
    TaskInfo() {};
};

// Thread information structure
struct ThreadInfo {
    uint64_t threadID_;
    int deviceID_;
    void* stream_;
    void* context_;
    void* args_;

    ThreadInfo() : threadID_(-1), deviceID_(-1), stream_(nullptr), context_(nullptr), args_(nullptr)
    {
    }
    ThreadInfo(int threadID, int deviceID, void* stream, void* context, void* args)
        : threadID_(threadID), deviceID_(deviceID), stream_(stream), context_(context), args_(args)
    {
    }
};

struct TaskWrapper {
    std::shared_ptr<TaskInfo> inputTaskInfo;
    std::function<void(const TaskInfo&, ThreadInfo&)> inputTaskFunc;
    std::atomic<TaskStatus> status{TaskStatus::PENDING};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

    std::list<std::shared_ptr<TaskWrapper>>::iterator queue_iterator;

    TaskWrapper(std::shared_ptr<TaskInfo> info,
                std::function<void(const TaskInfo&, ThreadInfo&)> func)
        : inputTaskInfo(info), inputTaskFunc(func), start_time(std::chrono::steady_clock::now())
    {
    }

    TaskWrapper(TaskInfo& info, std::function<void(const TaskInfo&, ThreadInfo&)> func)
        : inputTaskInfo(std::make_shared<TaskInfo>(info)),
          inputTaskFunc(func),
          start_time(std::chrono::steady_clock::now())
    {
    }

    TaskWrapper(const TaskInfo& info, std::function<void(const TaskInfo&, ThreadInfo&)> func)
        : inputTaskInfo(std::make_shared<TaskInfo>(info)),
          inputTaskFunc(func),
          start_time(std::chrono::steady_clock::now())
    {
    }
};

class TaskManager {
public:
    size_t mMaxThreadNum = 16;

    TaskManager(int deviceID, int dpRank, int tpSize, void* args, size_t threadNum, int mStreamNum,
                std::string type, std::vector<int> cpuToBind);
    ~TaskManager();

    // Submitting a Task (Non-Blocking)
    void SubmitTask(TaskInfo& info,
                    std::function<void(const TaskInfo&, ThreadInfo&)> inputTaskFunc);

private:
    void ExecuteTask(std::shared_ptr<TaskWrapper>& task, ThreadInfo& local_thread_info);
    // Task queue
    mutable std::mutex queue_mutex_;
    std::list<std::shared_ptr<TaskWrapper>> task_queue_;
    std::condition_variable queue_cv_;

    // Thread pool
    std::vector<std::thread> worker_threads_;
    std::vector<ThreadInfo> threadInfos;

    // Control variable
    std::atomic<bool> should_stop_{false};
    std::atomic<int> next_task_id_{1};

    std::atomic<int> running_tasks_count_{0};

    void workerThread(size_t thread_index);
    std::hash<std::thread::id> hasher;
};

#endif  // TASK_MANAGER_H