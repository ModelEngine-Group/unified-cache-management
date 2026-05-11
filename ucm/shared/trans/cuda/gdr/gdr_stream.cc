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
#include "gdr_stream.h"

#include <cerrno>
#include <string>
#include <thread>
#include <utility>

#include <cuda_runtime.h>

#include "gdr_config.h"
#include "logger/logger.h"

namespace {

UC::Status MakeGdrStatus(const char* op, int rc)
{
    return UC::Status::OsApiError(fmt::format("{} failed({})", op, rc));
}

}  // namespace

namespace UC::Trans {

GdrStream::~GdrStream()
{
    if (!channel_) { return; }

    (void)Synchronized();
    ShutdownBackgroundThreads();
}

Status GdrStream::Setup()
{
    const auto ret = cudaGetDevice(&deviceId_);
    if (ret != cudaSuccess) { return Status{ret, cudaGetErrorString(ret)}; }
    auto nicName = GdrNicConfig::ResolveNicName(deviceId_);
    if (!nicName) { return nicName.Error(); }
    nicName_ = std::move(nicName).Value();

    try {
        channel_ = GdrCopyLib::Open(deviceId_, nicName_);
    } catch (const std::exception& e) {
        return Status::OsApiError(
            fmt::format("failed to open GDR channel on device({}) with nic({}): {}", deviceId_,
                        nicName_, e.what()));
    }

    try {
        schedulerThread_ = std::thread{&GdrStream::SchedulerLoop, this};
        completionThread_ = std::thread{&GdrStream::CompletionLoop, this};
    } catch (const std::exception& e) {
        ShutdownBackgroundThreads();
        channel_.reset();
        return Status::Error(
            fmt::format("failed to start GDR scheduler/completion thread: {}", e.what()));
    }

    std::unique_lock<std::mutex> lock{mutex_};
    cv_.wait(lock, [this] {
        return HasAsyncError() || (schedulerReady_ && completionThreadReady_);
    });
    if (HasAsyncError()) {
        auto status = AsyncError();
        lock.unlock();
        ShutdownBackgroundThreads();
        channel_.reset();
        return status;
    }

    UC_INFO("Enable GDR stream on device({}) with nic({}).", deviceId_, nicName_);
    return Status::OK();
}

Status GdrStream::DeviceToHost(void* device, void* host, size_t size)
{
    auto status = DeviceToHostAsync(device, host, size);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    auto status = DeviceToHostAsync(device, host, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    auto status = DeviceToHostAsync(device, host, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::DeviceToHostAsync(void* device, void* host, size_t size)
{
    return SubmitAsync(host, device, size, GdrMemcpyDeviceToHost);
}

Status GdrStream::DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto status = DeviceToHostAsync(device[i], host[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::DeviceToHostAsync(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto status = DeviceToHostAsync(device[i], pHost, size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::HostToDevice(void* host, void* device, size_t size)
{
    auto status = HostToDeviceAsync(host, device, size);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    auto status = HostToDeviceAsync(host, device, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    auto status = HostToDeviceAsync(host, device, size, number);
    if (status.Failure()) { return status; }
    return Synchronized();
}

Status GdrStream::HostToDeviceAsync(void* host, void* device, size_t size)
{
    return SubmitAsync(device, host, size, GdrMemcpyHostToDevice);
}

Status GdrStream::HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto status = HostToDeviceAsync(host[i], device[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::HostToDeviceAsync(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; ++i) {
        auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + size * i);
        auto status = HostToDeviceAsync(pHost, device[i], size);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status GdrStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Status::OK();
}

// Wait until all queued work is done, or return an async error.
Status GdrStream::Synchronized()
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    std::unique_lock<std::mutex> lock{mutex_};
    cv_.wait(lock, [this] { return HasAsyncError() || IsIdle(); });
    if (HasAsyncError()) { return AsyncError(); }
    if (auto status = TakeCompletedOperationError(); status.has_value()) { return *status; }
    return Status::OK();
}

// Queue a wait. The scheduler thread will wait on the CUDA event later.
Status GdrStream::WaitEvent(void* event)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    bool shouldNotify = false;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (HasAsyncError()) { return AsyncError(); }
        if (!event) { return Status::OK(); }

        shouldNotify = operationsQueue_.empty();
        operationsQueue_.push_back(
            Operation{OperationType::Wait, nextOperationId_++, static_cast<cudaEvent_t>(event),
                      nullptr, nullptr, 0, GdrMemcpyHostToDevice});
    }
    if (shouldNotify) { cv_.notify_all(); }
    return Status::OK();
}

// Queue one copy. The scheduler thread will send it later.
Status GdrStream::SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    bool shouldNotify = false;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (HasAsyncError()) { return AsyncError(); }

        shouldNotify = operationsQueue_.empty();
        operationsQueue_.push_back(
            Operation{OperationType::Copy, nextOperationId_++, nullptr, dst, src, size, kind});
    }
    if (shouldNotify) { cv_.notify_all(); }
    return Status::OK();
}

// Run waits and copies from the queue in stream order.
void GdrStream::SchedulerLoop()
{
    // start up
    auto startupStatus = Status::OK();
    const auto ret = cudaSetDevice(deviceId_);
    if (ret != cudaSuccess) { startupStatus = Status{ret, cudaGetErrorString(ret)}; }

    {
        std::lock_guard<std::mutex> lock{mutex_};
        schedulerReady_ = true;
        if (startupStatus.Failure()) { StopWithAsyncError("cudaSetDevice", startupStatus); }
        cv_.notify_all();
    }
    if (startupStatus.Failure()) { return; }

    for (;;) {
        // wait until operation queue has something. 
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(
                lock,
                [this] { return stopRequested_ || HasAsyncError() || !operationsQueue_.empty(); });
        }
        
        // do until operation queue is empty.
        for (;;) {
            Operation op{OperationType::Copy, 0, nullptr, nullptr, nullptr, 0,
                         GdrMemcpyHostToDevice};
            size_t inflightBefore = 0;
            {
                std::lock_guard<std::mutex> lock{mutex_};
                if (HasAsyncError() || stopRequested_) { return; }
                if (operationsQueue_.empty()) { break; }

                op = operationsQueue_.front();
                inflightBefore = inflightRequestOperations_.size();
            }

            if (op.type == OperationType::Wait) {
                {
                    std::lock_guard<std::mutex> lock{mutex_};
                    schedulerWaitingOnEvent_ = true;
                }
                const auto waitRet = cudaEventSynchronize(op.event);
                {
                    std::lock_guard<std::mutex> lock{mutex_};
                    schedulerWaitingOnEvent_ = false;
                    if (waitRet != cudaSuccess) {
                        StopWithAsyncError(
                            "cudaEventSynchronize", Status{waitRet, cudaGetErrorString(waitRet)});
                        return;
                    }
                    if (stopRequested_) { continue; }
                    if (!operationsQueue_.empty()
                        && operationsQueue_.front().operationId == op.operationId) {
                        operationsQueue_.pop_front();
                    }
                    MarkOperationCompleted(op.operationId);
                    cv_.notify_all();
                }
            } else {    // copy submit
                const auto submitResult = SubmitCopyOperationFromQueue(op);
                if (submitResult == SubmitResult::Submitted) {
                    continue; 
                } else if (submitResult == SubmitResult::Error) { 
                    return; 
                } else {    // SubmitResult::Waiting
                    std::unique_lock<std::mutex> lock{mutex_};
                    cv_.wait(lock, [this, inflightBefore] {
                        return stopRequested_ || HasAsyncError()
                            || inflightRequestOperations_.size() < inflightBefore;
                    });
                }
            }
        }
    }
}

GdrStream::SubmitResult GdrStream::SubmitCopyOperationFromQueue(const Operation& op)
{
    uint64_t reqId = 0;
    std::lock_guard<std::mutex> lock{mutex_};
    if (HasAsyncError() || stopRequested_) { return SubmitResult::Error; }

    // Keep the submit and reqId bookkeeping atomic from the stream's point of view.
    // Fast completions can otherwise be polled before reqId is visible here.
    const auto rc = channel_->GdrMemcpyAsync(op.dst, op.src, op.size, op.kind, &reqId);
    if (rc == 0) {
        if (!operationsQueue_.empty()
            && operationsQueue_.front().operationId == op.operationId) {
            operationsQueue_.pop_front();
        }
        if (reqId != 0) {
            inflightRequestOperations_[reqId] = op.operationId;
        } else {
            MarkOperationCompleted(op.operationId);
        }
        if (reqId == 0) { cv_.notify_all(); }
        return SubmitResult::Submitted;
    }
    if (rc == -EAGAIN) { return SubmitResult::Waiting; }

    const auto status = MakeGdrStatus("GdrMemcpyAsync", rc);
    UC_ERROR("GDR copy operation {} failed at GdrMemcpyAsync: {}", op.operationId, status);
    if (!operationsQueue_.empty() && operationsQueue_.front().operationId == op.operationId) {
        operationsQueue_.pop_front();
    }
    MarkOperationFailed(op.operationId, status);
    cv_.notify_all();
    return SubmitResult::Submitted;
}

// Poll completions for sent GDR copies and mark them done.
void GdrStream::CompletionLoop()
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        completionThreadReady_ = true;
        cv_.notify_all();
    }

    for (;;) {
        const auto waitRc = channel_->WaitForCompletionEvent();
        {
            std::lock_guard<std::mutex> lock{mutex_};
            if (waitRc == -ECANCELED) {
                if (HasAsyncError() || stopRequested_) { return; }
            } else if (waitRc != 0) {
                StopWithAsyncError("WaitForCompletionEvent",
                                   MakeGdrStatus("WaitForCompletionEvent", waitRc));
                return;
            } else if (HasAsyncError() || stopRequested_) {
                return;
            }
        }
        if (waitRc == -ECANCELED) { continue; }

        for (;;) {
            uint64_t reqId = 0;
            const auto pollResult = channel_->PollCompletion(&reqId);
            if (pollResult == GdrCompletionPollResult::Completed) {
                std::lock_guard<std::mutex> lock{mutex_};
                if (HasAsyncError()) { return; }
                const auto it = inflightRequestOperations_.find(reqId);
                if (it == inflightRequestOperations_.end()) {
                    const auto status = Status::OsApiError(
                        fmt::format("unexpected GDR completion reqId({})", reqId));
                    StopWithAsyncError("PollCompletion reqId mismatch", status);
                    return;
                }
                const uint64_t operationId = it->second;
                inflightRequestOperations_.erase(it);
                MarkOperationCompleted(operationId);
                cv_.notify_all();
                continue;
            }

            if (pollResult == GdrCompletionPollResult::Empty) { break; }

            if (pollResult == GdrCompletionPollResult::UnknownRequest) {
                std::lock_guard<std::mutex> lock{mutex_};
                const auto status =
                    Status::OsApiError(fmt::format("unexpected GDR completion reqId({})", reqId));
                StopWithAsyncError("PollCompletion", status);
                return;
            }

            std::lock_guard<std::mutex> lock{mutex_};
            StopWithAsyncError("PollCompletion", Status::OsApiError("PollCompletion failed"));
            return;
        }
    }
}

void GdrStream::ShutdownBackgroundThreads()
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stopRequested_ = true;
        cv_.notify_all();
    }
    if (channel_) { channel_->InterruptCompletionWait(); }
    if (schedulerThread_.joinable()) { schedulerThread_.join(); }
    if (completionThread_.joinable()) { completionThread_.join(); }
}

void GdrStream::MarkOperationCompleted(uint64_t operationId)
{
    if (operationId <= lastCompletedOperationId_) { return; }

    if (operationId == lastCompletedOperationId_ + 1) {
        ++lastCompletedOperationId_;
        while (true) {
            const auto nextCompleted = completedOperations_.find(lastCompletedOperationId_ + 1);
            if (nextCompleted == completedOperations_.end()) { return; }
            completedOperations_.erase(nextCompleted);
            ++lastCompletedOperationId_;
        }
    }

    completedOperations_.insert(operationId);
    while (true) {
        const auto nextCompleted = completedOperations_.find(lastCompletedOperationId_ + 1);
        if (nextCompleted == completedOperations_.end()) { return; }
        completedOperations_.erase(nextCompleted);
        ++lastCompletedOperationId_;
    }
}

void GdrStream::MarkOperationFailed(uint64_t operationId, Status status)
{
    failedOperations_.emplace(operationId, std::move(status));
    MarkOperationCompleted(operationId);
}

void GdrStream::StopWithAsyncError(const char* source, Status status)
{
    if (!asyncError_.has_value()) {
        UC_ERROR("Async GDR stream error at {}: {}", source, status);
        asyncError_.emplace(std::move(status));
    }

    operationsQueue_.clear();
    inflightRequestOperations_.clear();
    completedOperations_.clear();
    failedOperations_.clear();
    schedulerWaitingOnEvent_ = false;
    stopRequested_ = true;
    if (channel_) { channel_->InterruptCompletionWait(); }
    cv_.notify_all();
}

bool GdrStream::HasAsyncError() const
{
    return asyncError_.has_value();
}

bool GdrStream::IsIdle() const
{
    return operationsQueue_.empty() && inflightRequestOperations_.empty()
        && !schedulerWaitingOnEvent_;
}

Status GdrStream::AsyncError() const
{
    return *asyncError_;
}

std::optional<Status> GdrStream::TakeCompletedOperationError()
{
    if (failedOperations_.empty()) { return std::nullopt; }
    auto it = failedOperations_.begin();
    if (it->first > lastCompletedOperationId_) { return std::nullopt; }

    auto status = it->second;
    failedOperations_.erase(it);
    return status;
}

}  // namespace UC::Trans
