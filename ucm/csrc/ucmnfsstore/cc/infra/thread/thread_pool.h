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

template <class Task>
class ThreadPool {
    using WorkerInitFn = std::function<bool(void)>;
    using WorkerFn = std::function<void(Task&)>;
    using WorkerExitFn = std::function<void(void)>;

public:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lk(this->_mtx);
            this->_stop = true;
            this->_cv.notify_all();
        }
        for (auto& w : this->_workers) {
            if (w.joinable()) { w.join(); }
        }
    }
    bool Setup(
        WorkerFn&& fn, WorkerInitFn&& initFn = [] { return true; }, WorkerExitFn&& exitFn = [] {},
        const size_t nWorker = 1) noexcept
    {
        this->_initFn = initFn;
        this->_fn = fn;
        this->_exitFn = exitFn;
        std::list<std::promise<bool>> start(nWorker);
        std::list<std::future<bool>> fut;
        for (auto& s : start) {
            fut.push_back(s.get_future());
            this->_workers.emplace_back([&] { this->Worker(s); });
        }
        auto success = true;
        for (auto& f : fut) {
            if (!f.get()) { success = false; }
        }
        return success;
    }
    void Push(std::list<Task>& tasks) noexcept
    {
        std::unique_lock<std::mutex> lk(this->_mtx);
        this->_taskQ.splice(this->_taskQ.end(), tasks);
        this->_cv.notify_all();
    }
    void Push(Task&& task) noexcept
    {
        std::unique_lock<std::mutex> lk(this->_mtx);
        this->_taskQ.push_back(task);
        this->_cv.notify_one();
    }

private:
    void Worker(std::promise<bool>& started) noexcept
    {
        auto success = this->_initFn();
        started.set_value(success);
        while (success) {
            Task task;
            std::unique_lock<std::mutex> lk(this->_mtx);
            this->_cv.wait(lk, [this] { return this->_stop || !this->_taskQ.empty(); });
            if (this->_stop) { break; }
            if (this->_taskQ.empty()) { continue; }
            task = std::move(this->_taskQ.front());
            this->_taskQ.pop_front();
            lk.unlock();
            this->_fn(task);
        }
        this->_exitFn();
    }

private:
    bool _stop{false};
    std::list<std::thread> _workers;
    WorkerInitFn _initFn;
    WorkerFn _fn;
    WorkerExitFn _exitFn;
    std::list<Task> _taskQ;
    std::mutex _mtx;
    std::condition_variable _cv;
};

} // namespace UC

#endif
