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
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "view/refresh_scheduler.h"

namespace kv {

// Resident-thread implementation of the RefreshScheduler contract. All trigger
// sources are merged into one pending refresh and callbacks run serially.
class ThreadedRefreshScheduler final : public RefreshScheduler {
public:
    ThreadedRefreshScheduler() = default;
    ~ThreadedRefreshScheduler() override;

    ThreadedRefreshScheduler(const ThreadedRefreshScheduler&) = delete;
    ThreadedRefreshScheduler& operator=(const ThreadedRefreshScheduler&) = delete;
    ThreadedRefreshScheduler(ThreadedRefreshScheduler&&) = delete;
    ThreadedRefreshScheduler& operator=(ThreadedRefreshScheduler&&) = delete;

    Status Start(std::chrono::milliseconds interval, RefreshCallback callback) override;
    void Stop() override;

    void TriggerByTimer() override;
    void TriggerByIoFailure(const Status& status) override;
    Status RefreshNow() override;

private:
    enum class TriggerSource {
        TIMER,
        IO_FAILURE,
        MANUAL,
    };

    struct RefreshWaiter;

    std::shared_ptr<RefreshWaiter> RequestRefreshLocked(TriggerSource source, bool createWaiter);
    void WorkerLoop();
    void CompleteActiveRefreshLocked(const Status& result);
    void CancelWaitersLocked(std::vector<std::shared_ptr<RefreshWaiter>>& waiters);

    mutable std::mutex mutex_;
    std::condition_variable workerCv_;
    std::condition_variable stopCv_;
    std::thread worker_;
    RefreshCallback callback_;
    std::chrono::milliseconds interval_{0};
    std::chrono::steady_clock::time_point nextDeadline_;

    bool running_{false};
    bool stopping_{false};
    bool stopRequested_{false};

    bool refreshPending_{false};
    TriggerSource pendingTrigger_{TriggerSource::TIMER};
    std::vector<std::shared_ptr<RefreshWaiter>> pendingWaiters_;

    bool refreshActive_{false};
    TriggerSource activeTrigger_{TriggerSource::TIMER};
    std::vector<std::shared_ptr<RefreshWaiter>> activeWaiters_;
};

}  // namespace kv
