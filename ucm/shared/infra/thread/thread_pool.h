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
#ifndef UNIFIEDCACHE_INFRA_THREAD_POOL_H
#define UNIFIEDCACHE_INFRA_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace UC {

template <class Task, class WorkerArgs = void*>
class ThreadPool {
    using WorkerInitFn = std::function<bool(WorkerArgs&)>;
    using WorkerFn = std::function<void(Task&, const WorkerArgs&)>;
    using WorkerTimeoutFn = std::function<void(Task&, const ssize_t)>;
    using WorkerExitFn = std::function<void(WorkerArgs&)>;

    class StopToken {
        std::shared_ptr<std::atomic<bool>> flag_ = std::make_shared<std::atomic<bool>>(false);

    public:
        void RequestStop() noexcept { this->flag_->store(true, std::memory_order_relaxed); }
        bool StopRequested() const noexcept { return this->flag_->load(std::memory_order_relaxed); }
    };

    enum class WorkerState { Idle, Running, TimeoutRequested, Exited };

    struct Worker {
        ssize_t tid;
        std::thread th;
        StopToken stop;
        std::weak_ptr<Task> current;
        std::atomic<std::chrono::steady_clock::time_point> tp{};
        std::atomic<WorkerState> state{WorkerState::Idle};
    };

public:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(this->taskMtx_);
            this->stop_.store(true, std::memory_order_relaxed);
            this->cv_.notify_all();
        }
        for (auto& worker : this->workers_) { worker->stop.RequestStop(); }
        if (this->monitor_.joinable()) { this->monitor_.join(); }
        for (auto& worker : this->workers_) {
            if (worker->th.joinable()) { worker->th.join(); }
        }
    }
    ThreadPool& SetWorkerFn(WorkerFn&& fn)
    {
        this->fn_ = std::move(fn);
        return *this;
    }
    ThreadPool& SetWorkerInitFn(WorkerInitFn&& fn)
    {
        this->initFn_ = std::move(fn);
        return *this;
    }
    ThreadPool& SetWorkerExitFn(WorkerExitFn&& fn)
    {
        this->exitFn_ = std::move(fn);
        return *this;
    }
    ThreadPool& SetWorkerTimeoutFn(WorkerTimeoutFn&& fn, const size_t timeoutMs,
                                   const size_t intervalMs = 1000)
    {
        this->timeoutFn_ = std::move(fn);
        this->timeoutMs_ = timeoutMs;
        this->intervalMs_ = intervalMs;
        return *this;
    }
    ThreadPool& SetNWorker(const size_t nWorker)
    {
        this->nWorker_ = nWorker;
        return *this;
    }
    size_t NWorker() const { return this->nWorker_; }
    bool Run()
    {
        if (this->nWorker_ == 0) { return false; }
        if (this->fn_ == nullptr) { return false; }
        this->workers_.reserve(this->nWorker_);
        for (size_t i = 0; i < this->nWorker_; i++) {
            if (!this->AddOneWorker()) { return false; }
        }
        if (this->timeoutMs_ > 0) {
            this->monitor_ = std::thread([this] { this->MonitorLoop(); });
        }
        return true;
    }
    void Push(std::list<Task>& tasks) noexcept
    {
        std::unique_lock<std::mutex> lock(this->taskMtx_);
        this->taskQ_.splice(this->taskQ_.end(), tasks);
        this->cv_.notify_all();
    }
    void Push(Task&& task) noexcept
    {
        std::unique_lock<std::mutex> lock(this->taskMtx_);
        this->taskQ_.push_back(std::move(task));
        this->cv_.notify_one();
    }

private:
    bool AddOneWorker()
    {
        try {
            auto worker = std::make_shared<Worker>();
            std::promise<bool> prom;
            auto fut = prom.get_future();
            worker->th = std::thread([this, worker, &prom] { this->WorkerLoop(prom, worker); });
            auto success = fut.get();
            if (!success) { return false; }
            this->workers_.push_back(worker);
            return true;
        } catch (...) {
            return false;
        }
    }
    void WorkerLoop(std::promise<bool>& prom, std::shared_ptr<Worker> worker)
    {
        worker->tid = syscall(SYS_gettid);
        WorkerArgs args = nullptr;
        auto success = true;
        if (this->initFn_) { success = this->initFn_(args); }
        prom.set_value(success);
        while (success) {
            std::shared_ptr<Task> task = nullptr;
            {
                std::unique_lock<std::mutex> lock(this->taskMtx_);
                this->cv_.wait(lock, [this, worker] {
                    return this->stop_.load(std::memory_order_relaxed) ||
                           worker->stop.StopRequested() || !this->taskQ_.empty();
                });
                if (this->stop_.load(std::memory_order_relaxed) || worker->stop.StopRequested()) {
                    break;
                }
                if (this->taskQ_.empty()) { continue; }
                task = std::make_shared<Task>(std::move(this->taskQ_.front()));
                this->taskQ_.pop_front();
            }
            worker->current = task;
            worker->tp.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
            worker->state.store(WorkerState::Running, std::memory_order_relaxed);
            this->fn_(*task, args);
            worker->current.reset();
            worker->tp.store({}, std::memory_order_relaxed);
            if (worker->stop.StopRequested()) { break; }
            worker->state.store(WorkerState::Idle, std::memory_order_relaxed);
        }
        worker->state.store(WorkerState::Exited, std::memory_order_relaxed);
        if (this->exitFn_) { this->exitFn_(args); }
    }

    void MonitorLoop()
    {
        const auto interval = std::chrono::milliseconds(this->intervalMs_);
        while (!this->stop_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(interval);
            this->MonitorTimeouts();
            size_t nWorker = this->CleanupExitedWorkers();
            for (size_t i = nWorker; i < this->nWorker_; i++) { (void)this->AddOneWorker(); }
        }
    }

    void MonitorTimeouts()
    {
        using namespace std::chrono;
        const auto timeout = milliseconds(this->timeoutMs_);
        for (auto& worker : this->workers_) {
            auto tp = worker->tp.load(std::memory_order_relaxed);
            auto task = worker->current.lock();
            auto now = steady_clock::now();
            if (!task || tp == steady_clock::time_point{}) { continue; }
            auto state = worker->state.load(std::memory_order_relaxed);
            if (state != WorkerState::Running) { continue; }
            if (now - tp <= timeout) { continue; }
            worker->state.store(WorkerState::TimeoutRequested, std::memory_order_relaxed);
            if (this->timeoutFn_) { this->timeoutFn_(*task, worker->tid); }
            worker->stop.RequestStop();
        }
    }

    size_t CleanupExitedWorkers()
    {
        for (auto it = this->workers_.begin(); it != this->workers_.end();) {
            auto state = (*it)->state.load(std::memory_order_relaxed);
            if (state == WorkerState::Exited) {
                if ((*it)->th.joinable()) { (*it)->th.join(); }
                it = this->workers_.erase(it);
                continue;
            }
            ++it;
        }
        return this->workers_.size();
    }

private:
    WorkerInitFn initFn_{nullptr};
    WorkerFn fn_{nullptr};
    WorkerTimeoutFn timeoutFn_{nullptr};
    WorkerExitFn exitFn_{nullptr};
    size_t timeoutMs_{0};
    size_t intervalMs_{0};
    size_t nWorker_{0};
    std::atomic<bool> stop_{false};
    std::vector<std::shared_ptr<Worker>> workers_;
    std::thread monitor_;
    std::mutex taskMtx_;
    std::list<Task> taskQ_;
    std::condition_variable cv_;
};

}  // namespace UC

#endif
