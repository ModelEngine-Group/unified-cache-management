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
 */
#ifndef UNIFIEDCACHE_TRANS_HOST_COPY_EXECUTOR_H
#define UNIFIEDCACHE_TRANS_HOST_COPY_EXECUTOR_H

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <utility>
#include <vector>
#include "status/status.h"
#include "thread/thread_pool.h"

namespace UC::Trans {

// A multi-worker executor, not a FIFO stream. Jobs and their callbacks may complete out of
// submission order, while the segments within one job are always copied in the supplied order.
class HostCopyExecutor {
public:
    enum class Direction { GATHER, SCATTER };

    struct Segment {
        void* address{nullptr};
        size_t size{0};
    };

    struct Result {
        Status status{Status::OK()};
        size_t bytes{0};
        double durationMs{0.0};
    };

    using Prerequisite = std::function<Status()>;
    using Completion = std::function<void(const Result&)>;

    struct Job {
        Direction direction{Direction::GATHER};
        void* contiguous{nullptr};
        std::vector<Segment> segments;
        Prerequisite prerequisite;
        Completion completion;
    };

    class Reservation {
        friend class HostCopyExecutor;

    public:
        Reservation() = default;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        Reservation(Reservation&& other) noexcept;
        Reservation& operator=(Reservation&& other) noexcept;
        ~Reservation();

        Status Submit(std::list<Job>& jobs);

    private:
        Reservation(HostCopyExecutor* executor, size_t count);
        void Release();

        HostCopyExecutor* executor_{nullptr};
        size_t count_{0};
    };

    HostCopyExecutor() = default;
    HostCopyExecutor(const HostCopyExecutor&) = delete;
    HostCopyExecutor& operator=(const HostCopyExecutor&) = delete;
    ~HostCopyExecutor();

    Status Setup(size_t workerNumber, size_t queueDepth,
                 const std::vector<ssize_t>& cpuAffinityCores = {});
    Expected<Reservation> Reserve(size_t number);
    Status Submit(std::list<Job>& jobs);
    Status PostCompletion(Completion completion);
    void Synchronize();

    static Status Gather(const std::vector<Segment>& sources, void* destination,
                         size_t* copiedBytes = nullptr);
    static Status Scatter(const void* source, const std::vector<Segment>& destinations,
                          size_t* copiedBytes = nullptr);

private:
    struct CompletionEvent {
        Completion completion;
        Result result;
    };

    Status SubmitReserved(size_t count, std::list<Job>& jobs);
    void Release(size_t count);
    void CopyWorker(Job& job);
    void CompletionWorker(CompletionEvent& event);
    void Shutdown();

    size_t queueDepth_{0};
    size_t outstanding_{0};
    bool accepting_{false};
    std::mutex stateMutex_;
    std::condition_variable idleCv_;
    ThreadPool<CompletionEvent> completionPool_;
    ThreadPool<Job> copyPool_;
};

}  // namespace UC::Trans

#endif
