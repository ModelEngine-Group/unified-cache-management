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
#ifndef UNIFIEDCACHE_THREAD_POOL_H
#define UNIFIEDCACHE_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <future>
#include <list>
#include <mutex>
#include <thread>

namespace UC {

template <class Task, class WorkerArgs = void*>
class ThreadPool {
    using WorkerInitFn = std::function<bool(WorkerArgs&)>;
    using WorkerFn = std::function<void(Task&, const WorkerArgs&)>;
    using WorkerExitFn = std::function<void(WorkerArgs&)>;

public:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lk(this->mtx_);
            this->stop_ = true;
            this->cv_.notify_all();
        }
        for (auto& w : this->workers_) {
            if (w.joinable()) { w.join(); }
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
    ThreadPool& SetNWorker(const size_t nWorker)
    {
        this->nWorker_ = nWorker;
        return *this;
    }
    bool Run()
    {
        if (this->nWorker_ == 0) { return false; }
        if (!this->fn_) { return false; }
        std::list<std::promise<bool>> start(this->nWorker_);
        std::list<std::future<bool>> fut;
        for (auto& s : start) {
            fut.push_back(s.get_future());
            this->workers_.emplace_back([&] { this->Worker(s); });
        }
        auto success = true;
        for (auto& f : fut) {
            if (!f.get()) { success = false; }
        }
        return success;
    }
    void Push(std::list<Task>& tasks) noexcept
    {
        std::unique_lock<std::mutex> lk(this->mtx_);
        this->taskQ_.splice(this->taskQ_.end(), tasks);
        this->cv_.notify_all();
    }
    void Push(Task&& task) noexcept
    {
        std::unique_lock<std::mutex> lk(this->mtx_);
        this->taskQ_.push_back(std::move(task));
        this->cv_.notify_one();
    }

private:
    void Worker(std::promise<bool>& started) noexcept
    {
        WorkerArgs args = nullptr;
        auto success = true;
        if (this->initFn_) { success = this->initFn_(args); }
        started.set_value(success);
        while (success) {
            std::unique_lock<std::mutex> lk(this->mtx_);
            this->cv_.wait(lk, [this] { return this->stop_ || !this->taskQ_.empty(); });
            if (this->stop_) { break; }
            if (this->taskQ_.empty()) { continue; }
            auto task = std::make_shared<Task>(std::move(this->taskQ_.front()));
            this->taskQ_.pop_front();
            lk.unlock();
            this->fn_(*task, args);
        }
        if (this->exitFn_) { this->exitFn_(args); }
    }

private:
    bool stop_{false};
    size_t nWorker_{1};
    std::list<std::thread> workers_;
    WorkerInitFn initFn_;
    WorkerFn fn_;
    WorkerExitFn exitFn_;
    std::list<Task> taskQ_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace UC

#endif
