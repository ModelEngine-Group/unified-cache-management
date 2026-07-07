#ifndef SUBMIT_EXECUTOR_GDR_H
#define SUBMIT_EXECUTOR_GDR_H

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "error_handle_gdr.h"

class SubmitExecutor {
    struct Worker {
        std::mutex mutex;
        std::condition_variable ready;
        std::condition_variable finished;
        std::function<void()> task;
        std::exception_ptr error;
        bool hasTask = false;
        bool done = true;
        bool stop = false;
        std::thread thread;
    };

public:
    static SubmitExecutor& Instance()
    {
        static SubmitExecutor executor;
        return executor;
    }

    void Initialize(size_t workerCount)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ASSERT(!initialized_);
        ASSERT(workerCount > 0);

        workers_.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index) {
            auto worker = std::make_unique<Worker>();
            Worker* workerPtr = worker.get();
            worker->thread = std::thread([workerPtr]() { WorkerLoop(workerPtr); });
            workers_.emplace_back(std::move(worker));
        }
        initialized_ = true;
    }

    template <class Task>
    void Run(size_t workerCount, Task&& task)
    {
        std::vector<Worker*> selectedWorkers;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ASSERT(initialized_);
            ASSERT(workerCount <= workers_.size());
            selectedWorkers.reserve(workerCount);
            for (size_t index = 0; index < workerCount; ++index) {
                selectedWorkers.push_back(workers_[index].get());
            }
        }

        for (size_t index = 0; index < workerCount; ++index) {
            auto* worker = selectedWorkers[index];
            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                ASSERT(!worker->hasTask);
                worker->task = [&task, index]() { task(index); };
                worker->error = nullptr;
                worker->done = false;
                worker->hasTask = true;
            }
            worker->ready.notify_one();
        }

        std::exception_ptr error;
        for (auto* worker : selectedWorkers) {
            std::unique_lock<std::mutex> lock(worker->mutex);
            worker->finished.wait(lock, [worker]() { return worker->done; });
            if (error == nullptr && worker->error != nullptr) { error = worker->error; }
        }

        if (error != nullptr) { std::rethrow_exception(error); }
    }

    void Shutdown() noexcept
    {
        std::vector<std::unique_ptr<Worker>> workers;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) { return; }

            for (const auto& worker : workers_) {
                {
                    std::lock_guard<std::mutex> workerLock(worker->mutex);
                    worker->stop = true;
                }
                worker->ready.notify_one();
            }

            workers.swap(workers_);
            initialized_ = false;
        }

        for (auto& worker : workers) {
            if (worker->thread.joinable()) { worker->thread.join(); }
        }
    }

private:
    static void WorkerLoop(Worker* worker)
    {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(worker->mutex);
                worker->ready.wait(lock, [worker]() { return worker->hasTask || worker->stop; });
                if (worker->stop && !worker->hasTask) { return; }
                task = std::move(worker->task);
                worker->hasTask = false;
            }

            std::exception_ptr error;
            try {
                task();
            } catch (...) {
                error = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(worker->mutex);
                worker->error = error;
                worker->done = true;
            }
            worker->finished.notify_one();
        }
    }

    SubmitExecutor() = default;

    std::mutex mutex_;
    std::vector<std::unique_ptr<Worker>> workers_;
    bool initialized_ = false;
};

#endif  // SUBMIT_EXECUTOR_GDR_H
