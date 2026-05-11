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

#include <algorithm>
#include <cerrno>
#include <chrono>
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

const char* SchedulerStateName(uint32_t state)
{
    switch (state) {
        case 0:
            return "starting";
        case 1:
            return "waiting";
        case 2:
            return "draining";
        case 3:
            return "wait_event";
        case 4:
            return "submitting";
        case 5:
            return "backpressure";
        case 6:
            return "exiting";
        default:
            return "unknown";
    }
}

const char* CompletionStateName(uint32_t state)
{
    switch (state) {
        case 0:
            return "starting";
        case 1:
            return "request_notify";
        case 2:
            return "polling";
        case 3:
            return "waiting_event";
        case 4:
            return "exiting";
        default:
            return "unknown";
    }
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

    stopRequested_.store(false, std::memory_order_release);
    schedulerReady_.store(false, std::memory_order_release);
    completionThreadReady_.store(false, std::memory_order_release);

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
        return HasAsyncError()
            || (schedulerReady_.load(std::memory_order_acquire)
                && completionThreadReady_.load(std::memory_order_acquire));
    });
    if (HasAsyncError()) {
        auto status = AsyncErrorLocked();
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

Status GdrStream::Synchronized()
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    const auto nextId = nextOperationId_.load(std::memory_order_acquire);
    const uint64_t targetOperationId = nextId == 0 ? 0 : nextId - 1;

    std::unique_lock<std::mutex> lock{mutex_};
    auto done = [this, targetOperationId] {
        return HasAsyncError()
            || lastCompletedOperationId_.load(std::memory_order_acquire) >= targetOperationId;
    };
    while (!done()) {
        if (!cv_.wait_for(lock, std::chrono::seconds(2), done)) {
            DumpDebugState("Synchronized waiting", targetOperationId);
        }
    }
    if (HasAsyncError()) { return AsyncErrorLocked(); }
    if (auto status = TakeCompletedOperationError(); status.has_value()) { return *status; }
    return Status::OK();
}

Status GdrStream::WaitEvent(void* event)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }
    if (HasAsyncError()) {
        std::lock_guard<std::mutex> lock{mutex_};
        return AsyncErrorLocked();
    }
    if (!event) { return Status::OK(); }

    const auto operationId = nextOperationId_.load(std::memory_order_relaxed);
    Operation op{OperationType::Wait, operationId, static_cast<cudaEvent_t>(event), nullptr,
                 nullptr, 0, GdrMemcpyHostToDevice};
    bool shouldNotify = false;
    auto status = PushOperation(op, &shouldNotify);
    if (status.Failure()) { return status; }
    nextOperationId_.store(operationId + 1, std::memory_order_release);
    if (shouldNotify) { cv_.notify_all(); }
    return Status::OK();
}

Status GdrStream::SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }
    if (HasAsyncError()) {
        std::lock_guard<std::mutex> lock{mutex_};
        return AsyncErrorLocked();
    }

    const auto operationId = nextOperationId_.load(std::memory_order_relaxed);
    Operation op{OperationType::Copy, operationId, nullptr, dst, src, size, kind};
    bool shouldNotify = false;
    auto status = PushOperation(op, &shouldNotify);
    if (status.Failure()) { return status; }
    nextOperationId_.store(operationId + 1, std::memory_order_release);
    if (shouldNotify) { cv_.notify_all(); }
    return Status::OK();
}

Status GdrStream::PushOperation(const Operation& op, bool* shouldNotify)
{
    for (size_t i = 0; i < kRingPushSpinCount; ++i) {
        if (stopRequested_.load(std::memory_order_acquire)) {
            return Status::Error("GDR stream is stopping");
        }
        if (HasAsyncError()) {
            std::lock_guard<std::mutex> lock{mutex_};
            return AsyncErrorLocked();
        }

        const auto tail = operationRingTail_.load(std::memory_order_relaxed);
        const auto head = operationRingHead_.load(std::memory_order_acquire);
        if (tail - head < kOperationRingCapacity) {
            if (tail == head) {
                std::lock_guard<std::mutex> lock{mutex_};
                const auto lockedTail = operationRingTail_.load(std::memory_order_relaxed);
                const auto lockedHead = operationRingHead_.load(std::memory_order_acquire);
                if (lockedTail - lockedHead >= kOperationRingCapacity) { continue; }
                operationRing_[lockedTail & kOperationRingMask] = op;
                operationRingTail_.store(lockedTail + 1, std::memory_order_release);
                if (shouldNotify) { *shouldNotify = lockedTail == lockedHead; }
                return Status::OK();
            }
            operationRing_[tail & kOperationRingMask] = op;
            operationRingTail_.store(tail + 1, std::memory_order_release);
            if (shouldNotify) { *shouldNotify = false; }
            return Status::OK();
        }
        std::this_thread::yield();
    }
    return Status::Error("GDR stream operation ring full");
}

size_t GdrStream::PopOperationBatch(Operation* out, size_t maxCount)
{
    const auto head = operationRingHead_.load(std::memory_order_relaxed);
    const auto tail = operationRingTail_.load(std::memory_order_acquire);
    const auto count = std::min<uint64_t>(maxCount, tail - head);
    for (uint64_t i = 0; i < count; ++i) {
        out[i] = operationRing_[(head + i) & kOperationRingMask];
    }
    if (count != 0) { operationRingHead_.store(head + count, std::memory_order_release); }
    return static_cast<size_t>(count);
}

bool GdrStream::OperationRingEmpty() const
{
    return operationRingHead_.load(std::memory_order_acquire)
        == operationRingTail_.load(std::memory_order_acquire);
}

void GdrStream::ResetOperationRing()
{
    const auto tail = operationRingTail_.load(std::memory_order_acquire);
    operationRingHead_.store(tail, std::memory_order_release);
}

void GdrStream::SchedulerLoop()
{
    auto startupStatus = Status::OK();
    const auto ret = cudaSetDevice(deviceId_);
    if (ret != cudaSuccess) { startupStatus = Status{ret, cudaGetErrorString(ret)}; }

    schedulerReady_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (startupStatus.Failure()) {
        StopWithAsyncError("cudaSetDevice", startupStatus);
        schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
        return;
    }

    Operation batch[kSchedulerBatchSize];
    for (;;) {
        {
            std::unique_lock<std::mutex> lock{mutex_};
            schedulerState_.store(kSchedulerStateWaiting, std::memory_order_release);
            cv_.wait(lock, [this] {
                return stopRequested_.load(std::memory_order_acquire) || HasAsyncError()
                    || !OperationRingEmpty();
            });
        }
        if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) {
            schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
            return;
        }

        for (;;) {
            schedulerState_.store(kSchedulerStateDraining, std::memory_order_release);
            const auto count = PopOperationBatch(batch, kSchedulerBatchSize);
            if (count == 0) { break; }

            for (size_t i = 0; i < count; ++i) {
                const auto& op = batch[i];
                lastSchedulerOperationId_.store(op.operationId, std::memory_order_release);
                if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) {
                    schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
                    return;
                }

                if (op.type == OperationType::Wait) {
                    schedulerState_.store(kSchedulerStateWaitEvent, std::memory_order_release);
                    schedulerWaitingOnEvent_.store(true, std::memory_order_release);
                    const auto waitRet = cudaEventSynchronize(op.event);
                    schedulerWaitingOnEvent_.store(false, std::memory_order_release);
                    if (waitRet != cudaSuccess) {
                        StopWithAsyncError("cudaEventSynchronize",
                                           Status{waitRet, cudaGetErrorString(waitRet)});
                        schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
                        return;
                    }
                    MarkOperationCompleted(op.operationId);
                    completionSignal_.fetch_add(1, std::memory_order_acq_rel);
                    cv_.notify_all();
                    continue;
                }

                for (;;) {
                    schedulerState_.store(kSchedulerStateSubmitting, std::memory_order_release);
                    const auto submitResult = SubmitCopyOperationFromQueue(op);
                    if (submitResult == SubmitResult::Submitted) { break; }
                    if (submitResult == SubmitResult::Error) {
                        schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
                        return;
                    }

                    schedulerState_.store(kSchedulerStateBackpressure, std::memory_order_release);
                    const auto signalBefore =
                        completionSignal_.load(std::memory_order_acquire);
                    std::unique_lock<std::mutex> lock{mutex_};
                    cv_.wait(lock, [this, signalBefore] {
                        return stopRequested_.load(std::memory_order_acquire) || HasAsyncError()
                            || completionSignal_.load(std::memory_order_acquire) != signalBefore;
                    });
                    if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) {
                        schedulerState_.store(kSchedulerStateExiting, std::memory_order_release);
                        return;
                    }
                }
            }
        }
    }
}

GdrStream::SubmitResult GdrStream::SubmitCopyOperationFromQueue(const Operation& op)
{
    if (HasAsyncError() || stopRequested_.load(std::memory_order_acquire)) {
        return SubmitResult::Error;
    }

    if (op.size == 0) {
        MarkOperationCompleted(op.operationId);
        completionSignal_.fetch_add(1, std::memory_order_acq_rel);
        cv_.notify_all();
        return SubmitResult::Submitted;
    }

    const auto rc =
        channel_->GdrMemcpyAsyncWithReqId(op.dst, op.src, op.size, op.kind, op.operationId);
    if (rc == 0) {
        submittedCopies_.fetch_add(1, std::memory_order_acq_rel);
        lastSubmittedOperationId_.store(op.operationId, std::memory_order_release);
        return SubmitResult::Submitted;
    }
    if (rc == -EAGAIN) {
        eagainCount_.fetch_add(1, std::memory_order_acq_rel);
        return SubmitResult::Waiting;
    }

    const auto status = MakeGdrStatus("GdrMemcpyAsyncWithReqId", rc);
    UC_ERROR("GDR copy operation {} failed at GdrMemcpyAsyncWithReqId: {}", op.operationId,
             status);
    submitFailureCount_.fetch_add(1, std::memory_order_acq_rel);
    MarkOperationFailed(op.operationId, status);
    completionSignal_.fetch_add(1, std::memory_order_acq_rel);
    cv_.notify_all();
    return SubmitResult::Submitted;
}

void GdrStream::CompletionLoop()
{
    completionThreadReady_.store(true, std::memory_order_release);
    cv_.notify_all();

    for (;;) {
        completionState_.store(kCompletionStateRequestNotify, std::memory_order_release);
        const auto notifyRc = channel_->RequestCompletionNotification();
        if (notifyRc != 0) {
            StopWithAsyncError("RequestCompletionNotification",
                               MakeGdrStatus("RequestCompletionNotification", notifyRc));
            completionState_.store(kCompletionStateExiting, std::memory_order_release);
            return;
        }

        completionState_.store(kCompletionStatePolling, std::memory_order_release);
        for (;;) {
            uint64_t reqId = 0;
            const auto pollResult = channel_->PollCompletion(&reqId);
            if (pollResult == GdrCompletionPollResult::Completed) {
                if (HasAsyncError()) {
                    completionState_.store(kCompletionStateExiting, std::memory_order_release);
                    return;
                }
                MarkOperationCompleted(reqId);
                completedCopies_.fetch_add(1, std::memory_order_acq_rel);
                lastCompletionReqId_.store(reqId, std::memory_order_release);
                completionSignal_.fetch_add(1, std::memory_order_acq_rel);
                cv_.notify_all();
                continue;
            }

            if (pollResult == GdrCompletionPollResult::Empty) {
                completionEmptyPolls_.fetch_add(1, std::memory_order_acq_rel);
                break;
            }

            if (pollResult == GdrCompletionPollResult::UnknownRequest) {
                const auto status =
                    Status::OsApiError(fmt::format("unexpected GDR completion reqId({})", reqId));
                StopWithAsyncError("PollCompletion", status);
                completionState_.store(kCompletionStateExiting, std::memory_order_release);
                return;
            }

            StopWithAsyncError("PollCompletion", Status::OsApiError("PollCompletion failed"));
            completionState_.store(kCompletionStateExiting, std::memory_order_release);
            return;
        }

        if (HasAsyncError() || stopRequested_.load(std::memory_order_acquire)) {
            completionState_.store(kCompletionStateExiting, std::memory_order_release);
            return;
        }

        completionState_.store(kCompletionStateWaitingEvent, std::memory_order_release);
        completionWaits_.fetch_add(1, std::memory_order_acq_rel);
        const auto waitRc = channel_->WaitForCompletionEvent();
        if (waitRc == -ECANCELED) {
            completionWakeups_.fetch_add(1, std::memory_order_acq_rel);
            if (HasAsyncError() || stopRequested_.load(std::memory_order_acquire)) {
                completionState_.store(kCompletionStateExiting, std::memory_order_release);
                return;
            }
            continue;
        }
        if (waitRc != 0) {
            StopWithAsyncError("WaitForCompletionEvent",
                               MakeGdrStatus("WaitForCompletionEvent", waitRc));
            completionState_.store(kCompletionStateExiting, std::memory_order_release);
            return;
        }
        completionWakeups_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void GdrStream::ShutdownBackgroundThreads()
{
    stopRequested_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (channel_) { channel_->InterruptCompletionWait(); }
    if (schedulerThread_.joinable()) { schedulerThread_.join(); }
    if (completionThread_.joinable()) { completionThread_.join(); }
}

bool GdrStream::MarkOperationCompleted(uint64_t operationId)
{
    if (operationId == 0) { return false; }

    const auto completedBefore = lastCompletedOperationId_.load(std::memory_order_acquire);
    if (operationId <= completedBefore) { return false; }

    completionSlots_[operationId & kCompletionRingMask].operationId.store(
        operationId, std::memory_order_release);
    AdvanceCompletedOperations();
    return lastCompletedOperationId_.load(std::memory_order_acquire) != completedBefore;
}

void GdrStream::AdvanceCompletedOperations()
{
    for (;;) {
        auto current = lastCompletedOperationId_.load(std::memory_order_acquire);
        const auto next = current + 1;
        auto& slot = completionSlots_[next & kCompletionRingMask];
        if (slot.operationId.load(std::memory_order_acquire) != next) { return; }
        if (lastCompletedOperationId_.compare_exchange_weak(
                current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            slot.operationId.store(0, std::memory_order_release);
        }
    }
}

void GdrStream::MarkOperationFailed(uint64_t operationId, Status status)
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        failedOperations_.emplace(operationId, std::move(status));
    }
    MarkOperationCompleted(operationId);
}

void GdrStream::StopWithAsyncError(const char* source, Status status)
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!asyncError_.has_value()) {
            UC_ERROR("Async GDR stream error at {}: {}", source, status);
            asyncError_.emplace(std::move(status));
            asyncErrorSet_.store(true, std::memory_order_release);
        }
        failedOperations_.clear();
    }

    ResetOperationRing();
    schedulerWaitingOnEvent_.store(false, std::memory_order_release);
    stopRequested_.store(true, std::memory_order_release);
    completionSignal_.fetch_add(1, std::memory_order_acq_rel);
    if (channel_) { channel_->InterruptCompletionWait(); }
    cv_.notify_all();
}

bool GdrStream::HasAsyncError() const
{
    return asyncErrorSet_.load(std::memory_order_acquire);
}

bool GdrStream::IsIdle() const
{
    const auto nextId = nextOperationId_.load(std::memory_order_acquire);
    const uint64_t targetOperationId = nextId == 0 ? 0 : nextId - 1;
    return lastCompletedOperationId_.load(std::memory_order_acquire) >= targetOperationId
        && !schedulerWaitingOnEvent_.load(std::memory_order_acquire);
}

void GdrStream::DumpDebugState(const char* source, uint64_t targetOperationId) const
{
    const auto head = operationRingHead_.load(std::memory_order_acquire);
    const auto tail = operationRingTail_.load(std::memory_order_acquire);
    const auto schedulerState = schedulerState_.load(std::memory_order_acquire);
    const auto completionState = completionState_.load(std::memory_order_acquire);
    UC_WARN(
        "{}: target({}) next({}) lastCompleted({}) ringHead({}) ringTail({}) ringSize({}) "
        "scheduler({}) schedulerOp({}) completion({}) submitted({}) completed({}) "
        "lastSubmitted({}) lastCompletion({}) eagain({}) submitFail({}) completionSignal({}) "
        "emptyPolls({}) waits({}) wakeups({}) waitEvent({}) stop({}) asyncError({}).",
        source, targetOperationId, nextOperationId_.load(std::memory_order_acquire),
        lastCompletedOperationId_.load(std::memory_order_acquire), head, tail, tail - head,
        SchedulerStateName(schedulerState),
        lastSchedulerOperationId_.load(std::memory_order_acquire),
        CompletionStateName(completionState),
        submittedCopies_.load(std::memory_order_acquire),
        completedCopies_.load(std::memory_order_acquire),
        lastSubmittedOperationId_.load(std::memory_order_acquire),
        lastCompletionReqId_.load(std::memory_order_acquire),
        eagainCount_.load(std::memory_order_acquire),
        submitFailureCount_.load(std::memory_order_acquire),
        completionSignal_.load(std::memory_order_acquire),
        completionEmptyPolls_.load(std::memory_order_acquire),
        completionWaits_.load(std::memory_order_acquire),
        completionWakeups_.load(std::memory_order_acquire),
        schedulerWaitingOnEvent_.load(std::memory_order_acquire),
        stopRequested_.load(std::memory_order_acquire), HasAsyncError());
}

Status GdrStream::AsyncErrorLocked() const
{
    if (asyncError_.has_value()) { return *asyncError_; }
    return Status::Error("GDR stream async error");
}

std::optional<Status> GdrStream::TakeCompletedOperationError()
{
    if (failedOperations_.empty()) { return std::nullopt; }
    auto it = failedOperations_.begin();
    if (it->first > lastCompletedOperationId_.load(std::memory_order_acquire)) {
        return std::nullopt;
    }

    auto status = it->second;
    failedOperations_.erase(it);
    return status;
}

}  // namespace UC::Trans
