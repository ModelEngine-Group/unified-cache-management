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
#include "host_copy_executor.h"
#include <chrono>
#include <cstring>
#include <exception>
#include "logger/logger.h"

namespace UC::Trans {

HostCopyExecutor::Reservation::Reservation(HostCopyExecutor* executor, size_t count)
    : executor_(executor), count_(count)
{}

HostCopyExecutor::Reservation::Reservation(Reservation&& other) noexcept
    : executor_(other.executor_), count_(other.count_)
{
    other.executor_ = nullptr;
    other.count_ = 0;
}

HostCopyExecutor::Reservation& HostCopyExecutor::Reservation::operator=(
    Reservation&& other) noexcept
{
    if (this == &other) { return *this; }
    Release();
    executor_ = other.executor_;
    count_ = other.count_;
    other.executor_ = nullptr;
    other.count_ = 0;
    return *this;
}

HostCopyExecutor::Reservation::~Reservation() { Release(); }

Status HostCopyExecutor::Reservation::Submit(std::list<Job>& jobs)
{
    if (executor_ == nullptr) { return Status::InvalidParam("invalid H2H reservation"); }
    auto* executor = executor_;
    const auto count = count_;
    executor_ = nullptr;
    count_ = 0;
    return executor->SubmitReserved(count, jobs);
}

void HostCopyExecutor::Reservation::Release()
{
    if (executor_ != nullptr) { executor_->Release(count_); }
    executor_ = nullptr;
    count_ = 0;
}

HostCopyExecutor::~HostCopyExecutor() { Shutdown(); }

Status HostCopyExecutor::Setup(size_t workerNumber, size_t queueDepth,
                               const std::vector<ssize_t>& cpuAffinityCores)
{
    if (workerNumber == 0 || queueDepth == 0) {
        return Status::InvalidParam("invalid H2H worker number({}) or queue depth({})",
                                    workerNumber, queueDepth);
    }
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        if (accepting_ || queueDepth_ != 0) {
            return Status::InvalidParam("HostCopyExecutor has already been set up");
        }
        queueDepth_ = queueDepth;
    }
    auto completionStarted =
        completionPool_
            .SetNWorker(1)
            .SetWorkerFn([this](CompletionEvent& event, void* const&) { CompletionWorker(event); })
            .SetCpuAffinity(cpuAffinityCores)
            .Run();
    auto copyStarted =
        copyPool_
            .SetNWorker(workerNumber)
            .SetWorkerFn([this](Job& job, void* const&) { CopyWorker(job); })
            .SetCpuAffinity(cpuAffinityCores)
            .Run();
    if (!completionStarted || !copyStarted) {
        return Status::Error("failed to start host copy executor workers");
    }
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        accepting_ = true;
    }
    return Status::OK();
}

Expected<HostCopyExecutor::Reservation> HostCopyExecutor::Reserve(size_t number)
{
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!accepting_) { return Status::Error("HostCopyExecutor is not accepting jobs"); }
    if (number == 0) { return Status::InvalidParam("cannot reserve zero H2H jobs"); }
    if (number > queueDepth_ || outstanding_ > queueDepth_ - number) {
        return Status::Error("HostCopyExecutor queue full");
    }
    outstanding_ += number;
    return Reservation(this, number);
}

Status HostCopyExecutor::Submit(std::list<Job>& jobs)
{
    auto reservation = Reserve(jobs.size());
    if (!reservation) { return reservation.Error(); }
    return reservation.Value().Submit(jobs);
}

Status HostCopyExecutor::PostCompletion(Completion completion)
{
    {
        std::lock_guard<std::mutex> lock(stateMutex_);
        if (!accepting_) { return Status::Error("HostCopyExecutor is not accepting jobs"); }
        ++outstanding_;
    }
    completionPool_.Push(CompletionEvent{std::move(completion), Result{}});
    return Status::OK();
}

Status HostCopyExecutor::SubmitReserved(size_t count, std::list<Job>& jobs)
{
    if (jobs.size() != count) {
        Release(count);
        return Status::InvalidParam("H2H job count({}, expect {})", jobs.size(), count);
    }
    copyPool_.Push(jobs);
    return Status::OK();
}

void HostCopyExecutor::Release(size_t count)
{
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (count > outstanding_) {
        UC_ERROR("Invalid HostCopyExecutor release count({}, outstanding {}).", count,
                 outstanding_);
        outstanding_ = 0;
    } else {
        outstanding_ -= count;
    }
    if (outstanding_ == 0) { idleCv_.notify_all(); }
}

void HostCopyExecutor::CopyWorker(Job& job)
{
    Result result;
    try {
        if (job.prerequisite) { result.status = job.prerequisite(); }
        if (result.status.Success()) {
            const auto start = std::chrono::steady_clock::now();
            if (job.direction == Direction::GATHER) {
                result.status = Gather(job.segments, job.contiguous, &result.bytes);
            } else {
                result.status = Scatter(job.contiguous, job.segments, &result.bytes);
            }
            result.durationMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
        }
    } catch (const std::exception& e) {
        result.status = Status::Error(e.what());
    } catch (...) {
        result.status = Status::Error("unknown exception in host copy worker");
    }
    completionPool_.Push(CompletionEvent{std::move(job.completion), std::move(result)});
}

void HostCopyExecutor::CompletionWorker(CompletionEvent& event)
{
    try {
        if (event.completion) { event.completion(event.result); }
    } catch (const std::exception& e) {
        UC_ERROR("Host copy completion failed: {}.", e.what());
    } catch (...) {
        UC_ERROR("Host copy completion failed with unknown exception.");
    }
    Release(1);
}

Status HostCopyExecutor::Gather(const std::vector<Segment>& sources, void* destination,
                                size_t* copiedBytes)
{
    if (destination == nullptr) { return Status::InvalidParam("invalid null host destination"); }
    auto* dst = static_cast<std::byte*>(destination);
    size_t offset = 0;
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i].address == nullptr) {
            return Status::InvalidParam("invalid null host source({})", i);
        }
        std::memcpy(dst + offset, sources[i].address, sources[i].size);
        offset += sources[i].size;
    }
    if (copiedBytes != nullptr) { *copiedBytes = offset; }
    return Status::OK();
}

Status HostCopyExecutor::Scatter(const void* source,
                                 const std::vector<Segment>& destinations,
                                 size_t* copiedBytes)
{
    if (source == nullptr) { return Status::InvalidParam("invalid null host source"); }
    const auto* src = static_cast<const std::byte*>(source);
    size_t offset = 0;
    for (size_t i = 0; i < destinations.size(); ++i) {
        if (destinations[i].address == nullptr) {
            return Status::InvalidParam("invalid null host destination({})", i);
        }
        std::memcpy(destinations[i].address, src + offset, destinations[i].size);
        offset += destinations[i].size;
    }
    if (copiedBytes != nullptr) { *copiedBytes = offset; }
    return Status::OK();
}

void HostCopyExecutor::Synchronize()
{
    std::unique_lock<std::mutex> lock(stateMutex_);
    idleCv_.wait(lock, [this] { return outstanding_ == 0; });
}

void HostCopyExecutor::Shutdown()
{
    std::unique_lock<std::mutex> lock(stateMutex_);
    accepting_ = false;
    idleCv_.wait(lock, [this] { return outstanding_ == 0; });
}

}  // namespace UC::Trans
