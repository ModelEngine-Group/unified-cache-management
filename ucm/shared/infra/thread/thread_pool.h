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

    struct Worker {
        ssize_t tid;
        std::thread th;
        StopToken stop;
        std::weak_ptr<Task> current;
        std::atomic<std::chrono::steady_clock::time_point> tp{};
    };

public:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(this->taskMtx_);
            this->stop_ = true;
            this->cv_.notify_all();
        }
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
                    return this->stop_ || worker->stop.StopRequested() || !this->taskQ_.empty();
                });
                if (this->stop_ || worker->stop.StopRequested()) { break; }
                if (this->taskQ_.empty()) { continue; }
                task = std::make_shared<Task>(std::move(this->taskQ_.front()));
                this->taskQ_.pop_front();
            }
            worker->current = task;
            worker->tp.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
            this->fn_(*task, args);
            if (worker->stop.StopRequested()) { break; }
            worker->current.reset();
            worker->tp.store({}, std::memory_order_relaxed);
        }
        if (this->exitFn_) { this->exitFn_(args); }
    }

    void MonitorLoop()
    {
        const auto interval = std::chrono::milliseconds(this->intervalMs_);
        while (true) {
            {
                std::unique_lock<std::mutex> lock(this->taskMtx_);
                this->cv_.wait_for(lock, interval, [this] { return this->stop_; });
                if (this->stop_) { break; }
            }
            size_t nWorker = this->Monitor();
            for (size_t i = nWorker; i < this->nWorker_; i++) { (void)this->AddOneWorker(); }
        }
    }

    size_t Monitor()
    {
        using namespace std::chrono;
        const auto timeout = milliseconds(this->timeoutMs_);
        size_t nWorker = 0;
        for (auto it = this->workers_.begin(); it != this->workers_.end();) {
            auto tp = (*it)->tp.load(std::memory_order_relaxed);
            auto task = (*it)->current.lock();
            auto now = steady_clock::now();
            if (task && tp != steady_clock::time_point{} && now - tp > timeout) {
                if (this->timeoutFn_) { this->timeoutFn_(*task, (*it)->tid); }
                (*it)->stop.RequestStop();
                if ((*it)->th.joinable()) { (*it)->th.detach(); }
                it = this->workers_.erase(it);
            } else {
                it++;
            }
            nWorker++;
        }
        return nWorker;
    }

private:
    WorkerInitFn initFn_{nullptr};
    WorkerFn fn_{nullptr};
    WorkerTimeoutFn timeoutFn_{nullptr};
    WorkerExitFn exitFn_{nullptr};
    size_t timeoutMs_{0};
    size_t intervalMs_{0};
    size_t nWorker_{0};
    bool stop_{false};
    std::vector<std::shared_ptr<Worker>> workers_;
    std::thread monitor_;
    std::mutex taskMtx_;
    std::list<Task> taskQ_;
    std::condition_variable cv_;
};

}  // namespace UC

#endif
