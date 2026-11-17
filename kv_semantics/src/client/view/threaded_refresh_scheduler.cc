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
 * */
#include "view/threaded_refresh_scheduler.h"
#include <future>
#include <stdexcept>
#include <utility>

namespace kv {

struct ThreadedRefreshScheduler::RefreshWaiter {
    std::promise<Status> promise;
    std::future<Status> future = promise.get_future();
};

ThreadedRefreshScheduler::~ThreadedRefreshScheduler() { Stop(); }

Status ThreadedRefreshScheduler::Start(std::chrono::milliseconds interval, RefreshCallback callback)
{
    if (interval <= std::chrono::milliseconds::zero()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "RefreshScheduler interval must be greater than zero");
    }
    if (callback == nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "RefreshScheduler callback must not be empty");
    }

    std::lock_guard<std::mutex> lock{mutex_};
    if (running_ || stopping_) {
        return Status::Error(StatusCode::RESOURCE_BUSY, "RefreshScheduler is already running");
    }

    callback_ = std::move(callback);
    interval_ = interval;
    nextDeadline_ = std::chrono::steady_clock::now() + interval_;
    refreshPending_ = false;
    pendingTrigger_ = TriggerSource::TIMER;
    pendingWaiters_.clear();
    refreshActive_ = false;
    activeTrigger_ = TriggerSource::TIMER;
    activeWaiters_.clear();
    stopRequested_ = false;
    running_ = true;

    try {
        worker_ = std::thread(&ThreadedRefreshScheduler::WorkerLoop, this);
    } catch (const std::exception& error) {
        running_ = false;
        callback_ = nullptr;
        interval_ = std::chrono::milliseconds::zero();
        return Status::Error(
            StatusCode::INTERNAL_ERROR,
            "failed to start RefreshScheduler worker: " + std::string(error.what()));
    }

    return Status::OK();
}

void ThreadedRefreshScheduler::Stop()
{
    std::thread workerToJoin;
    {
        std::unique_lock<std::mutex> lock{mutex_};
        if (!running_) { return; }
        if (stopping_) {
            stopCv_.wait(lock, [this] { return !running_; });
            return;
        }

        stopping_ = true;
        stopRequested_ = true;
        workerCv_.notify_all();
        workerToJoin = std::move(worker_);
    }

    if (workerToJoin.joinable()) { workerToJoin.join(); }

    std::lock_guard<std::mutex> lock{mutex_};
    CancelWaitersLocked(pendingWaiters_);
    CancelWaitersLocked(activeWaiters_);
    refreshPending_ = false;
    pendingTrigger_ = TriggerSource::TIMER;
    refreshActive_ = false;
    activeTrigger_ = TriggerSource::TIMER;
    callback_ = nullptr;
    interval_ = std::chrono::milliseconds::zero();
    running_ = false;
    stopping_ = false;
    stopRequested_ = false;
    workerCv_.notify_all();
    stopCv_.notify_all();
}

void ThreadedRefreshScheduler::TriggerByTimer()
{
    std::lock_guard<std::mutex> lock{mutex_};
    (void)RequestRefreshLocked(TriggerSource::TIMER, false);
}

void ThreadedRefreshScheduler::TriggerByIoFailure(const Status& status)
{
    if (status.ok()) { return; }

    std::lock_guard<std::mutex> lock{mutex_};
    (void)RequestRefreshLocked(TriggerSource::IO_FAILURE, false);
}

Status ThreadedRefreshScheduler::RefreshNow()
{
    std::shared_ptr<RefreshWaiter> waiter;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!running_ || stopRequested_) {
            return Status::Error(StatusCode::NOT_INITIALIZED, "RefreshScheduler is not running");
        }
        waiter = RequestRefreshLocked(TriggerSource::MANUAL, true);
    }

    if (waiter == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "RefreshScheduler failed to create a refresh waiter");
    }
    return waiter->future.get();
}

std::shared_ptr<ThreadedRefreshScheduler::RefreshWaiter>
ThreadedRefreshScheduler::RequestRefreshLocked(TriggerSource source, bool createWaiter)
{
    if (!running_ || stopRequested_) { return nullptr; }

    auto waiter = createWaiter ? std::make_shared<RefreshWaiter>() : nullptr;
    if (refreshPending_) {
        if (source == TriggerSource::MANUAL) { pendingTrigger_ = TriggerSource::MANUAL; }
        if (waiter != nullptr) { pendingWaiters_.push_back(waiter); }
        return waiter;
    }
    if (refreshActive_) {
        if (waiter != nullptr) { activeWaiters_.push_back(waiter); }
        return waiter;
    }

    pendingTrigger_ = source;
    refreshPending_ = true;
    if (waiter != nullptr) { pendingWaiters_.push_back(waiter); }
    workerCv_.notify_all();
    return waiter;
}

void ThreadedRefreshScheduler::WorkerLoop()
{
    std::unique_lock<std::mutex> lock{mutex_};
    while (true) {
        if (stopRequested_) { break; }

        const auto now = std::chrono::steady_clock::now();
        if (now >= nextDeadline_) {
            if (!refreshPending_ && !refreshActive_) {
                (void)RequestRefreshLocked(TriggerSource::TIMER, false);
            }
            // A due timer always establishes the next fixed-period deadline.
            // IO failures may wake the worker early without moving it.
            nextDeadline_ = now + interval_;
        }

        if (refreshPending_) {
            activeTrigger_ = pendingTrigger_;
            activeWaiters_ = std::move(pendingWaiters_);
            pendingWaiters_.clear();
            refreshPending_ = false;
            pendingTrigger_ = TriggerSource::TIMER;
            refreshActive_ = true;

            RefreshCallback callback = callback_;
            lock.unlock();
            Status result;
            try {
                result = callback();
            } catch (const std::exception& error) {
                result = Status::Error(
                    StatusCode::INTERNAL_ERROR,
                    "refresh callback threw an exception: " + std::string(error.what()));
            } catch (...) {
                result = Status::Error(StatusCode::INTERNAL_ERROR,
                                       "refresh callback threw an unknown exception");
            }
            lock.lock();

            if (activeTrigger_ == TriggerSource::MANUAL) {
                nextDeadline_ = std::chrono::steady_clock::now() + interval_;
            }
            CompleteActiveRefreshLocked(result);
            continue;
        }

        if (stopRequested_) { break; }
        workerCv_.wait_until(lock, nextDeadline_,
                             [this] { return stopRequested_ || refreshPending_; });
    }
}

void ThreadedRefreshScheduler::CompleteActiveRefreshLocked(const Status& result)
{
    for (auto& waiter : activeWaiters_) { waiter->promise.set_value(result); }
    activeWaiters_.clear();
    activeTrigger_ = TriggerSource::TIMER;
    refreshActive_ = false;
}

void ThreadedRefreshScheduler::CancelWaitersLocked(
    std::vector<std::shared_ptr<RefreshWaiter>>& waiters)
{
    const Status canceled = Status::Error(StatusCode::CANCELED, "RefreshScheduler was stopped");
    for (auto& waiter : waiters) { waiter->promise.set_value(canceled); }
    waiters.clear();
}

}  // namespace kv
