/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 */

#include "common/thread_pool.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace transport {
namespace {

constexpr auto kIdleTimeout = std::chrono::seconds(1);

}  // namespace

struct ThreadPool::State {
    std::mutex mutex;
    std::condition_variable ready;
    std::condition_variable stopped;
    std::deque<Task> tasks;
    std::size_t max_threads = 1;
    std::size_t worker_count = 0;
    std::size_t idle_count = 0;
    bool stopping = false;
};

ThreadPool::ThreadPool(std::size_t max_threads) : state_(std::make_shared<State>())
{
    const auto detected = static_cast<std::size_t>(std::thread::hardware_concurrency());
    state_->max_threads = max_threads == 0 ? std::max<std::size_t>(detected, 1) : max_threads;
}

ThreadPool::~ThreadPool() { Shutdown(); }

UC::Status ThreadPool::Submit(Task task)
{
    if (!task) { return UC::Status::InvalidParam(); }
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->stopping) { return UC::Status::Error(); }
    state_->tasks.push_back(std::move(task));
    if (state_->idle_count == 0 && state_->worker_count < state_->max_threads) {
        ++state_->worker_count;
        try {
            std::thread(&ThreadPool::RunWorker, state_).detach();
        } catch (...) {
            --state_->worker_count;
            state_->tasks.pop_back();
            return UC::Status::Error();
        }
    }
    state_->ready.notify_one();
    return UC::Status::OK();
}

void ThreadPool::RunWorker(const std::shared_ptr<State>& state)
{
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            ++state->idle_count;
            const auto ready = state->ready.wait_for(
                lock, kIdleTimeout, [&state] { return state->stopping || !state->tasks.empty(); });
            --state->idle_count;
            if ((!ready || state->stopping) && state->tasks.empty()) {
                --state->worker_count;
                state->stopped.notify_all();
                return;
            }
            task = std::move(state->tasks.front());
            state->tasks.pop_front();
        }
        try {
            task();
        } catch (...) {
        }
    }
}

void ThreadPool::Shutdown()
{
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->stopping = true;
    state_->ready.notify_all();
    state_->stopped.wait(lock, [this] { return state_->worker_count == 0; });
}

}  // namespace transport
