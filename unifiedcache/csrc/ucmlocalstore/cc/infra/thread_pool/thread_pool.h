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
#ifndef UCM_LOCAL_STORE_THREAD_POOL_H
#define UCM_LOCAL_STORE_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <cstdint>

namespace UCM {

class ThreadPool {
    using Task = std::packaged_task<void()>;

public:
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    static ThreadPool& Instance() {
        static ThreadPool ins{};
        return ins;
    }

    ~ThreadPool() {
        this->Stop();
    }

    template <class F, class... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<decltype(std::forward<F>(f)(std::forward<Args>(args)...))>
    {
        using RetType = decltype(std::forward<F>(f)(std::forward<Args>(args)...));
        if (this->_stop.load(std::memory_order_acquire)) {
            return std::future<RetType>{};
        }

        auto task = std::make_shared<std::packaged_task<RetType()>>(
                        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                    );

        std::future<RetType> ret = task->get_future();
        {
            std::lock_guard<std::mutex> mtx(this->_mtx);
            this->_tasks.emplace([task] { (*task)(); });
        }
        this->_cv.notify_one();
        return ret;
    }

private:
    ThreadPool() : _stop{false}
    {
        this->Start();
    }

    void Start() {
        uint64_t n = static_cast<uint64_t>(std::thread::hardware_concurrency());
        for (uint64_t i = 0; i < n; ++i) {
            this->_pool.emplace_back([this]() {
                while (!this->_stop.load(std::memory_order_acquire)) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> mtx(this->_mtx);
                        this->_cv.wait(
                            mtx,
                            [this] { return this->_stop.load(std::memory_order_acquire) || !this->_tasks.empty(); }
                        );
                        if (this->_tasks.empty()) {
                            return;
                        }

                        task = std::move(this->_tasks.front());
                        this->_tasks.pop();
                    }
                    task();
                    }
                }
            );
        }
    }

    void Stop() {
        this->_stop.store(true, std::memory_order_release);
        this->_cv.notify_all();
        for (auto& t : this->_pool) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    std::atomic_bool         _stop;
    std::mutex               _mtx;
    std::condition_variable  _cv;
    std::thread              _thread;
    std::queue<Task>         _tasks;
};

class ThreadPool {
    using Task = std::packaged_task<void()>;

public:
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    static ThreadPool& Instance() {
        static ThreadPool ins{};
        return ins;
    }

    ~ThreadPool() {
        this->Stop();
    }

    template <class F, class... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<decltype(std::forward<F>(f)(std::forward<Args>(args)...))>
    {
        using RetType = decltype(std::forward<F>(f)(std::forward<Args>(args)...));
        if (this->_stop.load(std::memory_order_acquire)) {
            return std::future<RetType>{};
        }

        auto task = std::make_shared<std::packaged_task<RetType()>>(
                        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                    );

        std::future<RetType> ret = task->get_future();
        {
            std::lock_guard<std::mutex> mtx(this->_mtx);
            this->_tasks.emplace([task] { (*task)(); });
        }
        this->_cv.notify_one();
        return ret;
    }

private:
    ThreadPool() : _stop{false}
    {
        this->Start();
    }

    void Start() {
        uint64_t n = static_cast<uint64_t>(std::thread::hardware_concurrency());
        for (uint64_t i = 0; i < n; ++i) {
            this->_pool.emplace_back([this]() {
                while (!this->_stop.load(std::memory_order_acquire)) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> mtx(this->_mtx);
                        this->_cv.wait(
                            mtx,
                            [this] { return this->_stop.load(std::memory_order_acquire) || !this->_tasks.empty(); }
                        );
                        if (this->_tasks.empty()) {
                            return;
                        }

                        task = std::move(this->_tasks.front());
                        this->_tasks.pop();
                    }
                    task();
                    }
                }
            );
        }
    }

    void Stop() {
        this->_stop.store(true, std::memory_order_release);
        this->_cv.notify_all();
        for (auto& t : this->_pool) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    std::mutex               _mtx;
    std::condition_variable  _cv;
    std::atomic_bool         _stop;
    std::queue<Task>         _tasks;
    std::vector<std::thread> _pool;
};

} // namespace UCM

#endif // UCM_LOCAL_STORE_THREAD_POOL_H