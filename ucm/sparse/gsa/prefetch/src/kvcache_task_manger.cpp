/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#include "kvcache_pre.h"

namespace {

// Bind worker threads to specified CPU cores and print binding info.
void SetupCpuAffinity(const std::vector<int>& cpuToBind, const std::string& type,
    int deviceID, size_t cpuNum, size_t cpuOffset, size_t resetCpuNum,
    size_t cpuNumMax)
{
    if (cpuToBind.size() >= cpuNumMax) {
        if (type == "indexer") {
            cpuOffset = cpuOffset + cpuNum;
            cpuNum = resetCpuNum;
        }
    } else {
        cpuNum = cpuToBind.size();
        cpuOffset = 0;
    }
    if (cpuToBind.size() <= 0) {
        return;
    }
    std::vector<ssize_t> cores(cpuNum);
    std::cout << "[GSA] cpu binds of" << type << " for rank " << deviceID << ": [";
    for (size_t tid = 0; tid < cpuNum; tid++) {
        cores[tid] = cpuToBind[tid + cpuOffset];
        std::cout << cores[tid] << ", ";
    }
    std::cout << "]" << std::endl;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (const auto core : cores) { CPU_SET(core, &mask); }
    auto ret = sched_setaffinity(0, sizeof(mask), &mask);
    if (ret != 0) { std::cout << "[GSA] ERROR sched_setaffinity" << std::endl; }
}

} // namespace

TaskManager::TaskManager(int deviceID, int dpRank, int tpSize,
    void* args, size_t threadNum, int mStreamNum, std::string type,
    std::vector<int> cpuToBind)
{
    threadInfos.resize(threadNum);
    mMaxThreadNum = threadNum;
    for (size_t i = 0; i < threadNum; ++i) {
        worker_threads_.emplace_back([this, i, deviceID, dpRank, tpSize, args, mStreamNum, type, cpuToBind]() {
            void* stream;
            void* context;
            // Solution 1: Each rank corresponds to a different node and different IDs.
            size_t cpuNum = 20;
            size_t cpuOffset = 0;
            size_t resetCpuNum = 4;
            size_t cpuNumMax = 24;
            SetupCpuAffinity(cpuToBind, type, deviceID, cpuNum, cpuOffset, resetCpuNum, cpuNumMax);
            ObtainStream(&stream, &context, deviceID, mStreamNum);
            ThreadInfo local_thread_info(
                static_cast<uint64_t>(hasher(std::this_thread::get_id())),
                deviceID, stream, context, args
            );
            std::cout << "zambin thread id: " << static_cast<uint64_t>(hasher(std::this_thread::get_id()))
                << " args: " << args << " mStreamNum: " << mStreamNum << std::endl;
            // Store thread information into a global array
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                threadInfos[i] = local_thread_info;
            }
            while (!should_stop_.load(std::memory_order_acquire)) {
                std::shared_ptr<TaskWrapper> task_to_execute;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_cv_.wait(lock, [this] {
                        return !task_queue_.empty() || should_stop_.load(std::memory_order_acquire);
                    });
                    if (should_stop_.load(std::memory_order_acquire) && task_queue_.empty()) {
                        break;
                    }
                    if (!task_queue_.empty()) {
                        auto front_it = task_queue_.begin();
                        task_to_execute = *front_it;
                        task_queue_.erase(front_it);
                        // Update the task status
                        task_to_execute->status.store(TaskStatus::RUNNING, std::memory_order_release);
                        // add running_tasks_count_
                        running_tasks_count_.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (task_to_execute) {
                    ExecuteTask(task_to_execute, local_thread_info);
                }
            }
            DestoryStream(local_thread_info.stream_, local_thread_info.context_,
                local_thread_info.deviceID_, mStreamNum);
        });
    }
}

void TaskManager::ExecuteTask(std::shared_ptr<TaskWrapper>& task,
    ThreadInfo& local_thread_info)
{
    try {
        task->inputTaskFunc(*task->inputTaskInfo, local_thread_info);
    } catch (...) {
        std::cout << "[GSA] Task: " << task->inputTaskInfo->taskID << " is wrong!!" << std::endl;
    }
    // Update the task status
    task->status.store(TaskStatus::COMPLETED, std::memory_order_release);
    running_tasks_count_.fetch_sub(1, std::memory_order_relaxed);
    task->end_time = std::chrono::steady_clock::now();
}

TaskManager::~TaskManager()
{
    should_stop_.store(true, std::memory_order_release);

    queue_cv_.notify_all();

    for (auto& t : worker_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void TaskManager::SubmitTask(TaskInfo& info,
    std::function<void(const TaskInfo&, ThreadInfo&)> inputTaskFunc)
{
    auto task_wrapper = std::make_shared<TaskWrapper>(info, inputTaskFunc);

    // Add a task to the queue (non-blocking operation)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push_back(task_wrapper);
        task_wrapper->queue_iterator = --task_queue_.end();
    }
    queue_cv_.notify_one();
}