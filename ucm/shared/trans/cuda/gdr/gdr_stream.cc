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
#include <cuda_runtime.h>
#include <string>
#include <thread>
#include <utility>
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

    stopRequested_.store(false, std::memory_order_release);

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

    while (!HasAsyncError() &&
           lastCompletedOperationId_.load(std::memory_order_acquire) < targetOperationId) {
        std::this_thread::yield();
    }

    std::lock_guard<std::mutex> lock{mutex_};
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
    Operation op{
        OperationType::Wait,  operationId, static_cast<cudaEvent_t>(event), nullptr, nullptr, 0,
        GdrMemcpyHostToDevice};
    auto status = PushOperation(op);
    if (status.Failure()) { return status; }
    nextOperationId_.store(operationId + 1, std::memory_order_release);
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
    auto status = PushOperation(op);
    if (status.Failure()) { return status; }
    nextOperationId_.store(operationId + 1, std::memory_order_release);
    return Status::OK();
}

Status GdrStream::PushOperation(const Operation& op)
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
            operationRing_[tail & kOperationRingMask] = op;
            operationRingTail_.store(tail + 1, std::memory_order_release);
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

void GdrStream::ResetOperationRing()
{
    const auto tail = operationRingTail_.load(std::memory_order_acquire);
    operationRingHead_.store(tail, std::memory_order_release);
}

void GdrStream::SchedulerLoop()
{
    const auto ret = cudaSetDevice(deviceId_);
    if (ret != cudaSuccess) {
        StopWithAsyncError("cudaSetDevice", Status{ret, cudaGetErrorString(ret)});
        return;
    }

    Operation batch[kSchedulerBatchSize];
    constexpr size_t kSchedulerIdleSpinLimit = 16;
    constexpr auto kSchedulerIdleSleep = std::chrono::microseconds(100);
    size_t idleSpinCount = 0;
    // Poll the SPSC ring with short idle backoff to keep submission lock-free.
    for (;;) {
        if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) { return; }

        const auto count = PopOperationBatch(batch, kSchedulerBatchSize);
        if (count == 0) {
            if (++idleSpinCount < kSchedulerIdleSpinLimit) {
                std::this_thread::yield();
            } else {
                if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) { return; }
                std::this_thread::sleep_for(kSchedulerIdleSleep);
                idleSpinCount = 0;
            }
            continue;
        }
        idleSpinCount = 0;

        for (size_t i = 0; i < count; ++i) {
            const auto& op = batch[i];
            if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) { return; }

            if (op.type == OperationType::Wait) {
                const auto waitRet = cudaEventSynchronize(op.event);
                if (waitRet != cudaSuccess) {
                    StopWithAsyncError("cudaEventSynchronize",
                                       Status{waitRet, cudaGetErrorString(waitRet)});
                    return;
                }
                MarkOperationCompleted(op.operationId);
                continue;
            }

            for (;;) {
                const auto submitResult = SubmitCopyOperationFromQueue(op);
                if (submitResult == SubmitResult::Submitted) { break; }
                if (submitResult == SubmitResult::Error) { return; }
                if (stopRequested_.load(std::memory_order_acquire) || HasAsyncError()) { return; }
                std::this_thread::yield();
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
        return SubmitResult::Submitted;
    }

    const auto rc =
        channel_->GdrMemcpyAsyncWithReqId(op.dst, op.src, op.size, op.kind, op.operationId);
    if (rc == 0) { return SubmitResult::Submitted; }
    if (rc == -EAGAIN) { return SubmitResult::Waiting; }

    const auto status = MakeGdrStatus("GdrMemcpyAsyncWithReqId", rc);
    UC_ERROR("GDR copy operation {} failed at GdrMemcpyAsyncWithReqId: {}", op.operationId, status);
    MarkOperationFailed(op.operationId, status);
    return SubmitResult::Submitted;
}

void GdrStream::CompletionLoop()
{
    for (;;) {
        const auto notifyRc = channel_->RequestCompletionNotification();
        if (notifyRc != 0) {
            StopWithAsyncError("RequestCompletionNotification",
                               MakeGdrStatus("RequestCompletionNotification", notifyRc));
            return;
        }

        for (;;) {
            uint64_t reqId = 0;
            const auto pollResult = channel_->PollCompletion(&reqId);
            if (pollResult == GdrCompletionPollResult::Completed) {
                if (HasAsyncError()) { return; }
                MarkOperationCompleted(reqId);
                continue;
            }

            if (pollResult == GdrCompletionPollResult::Empty) { break; }

            if (pollResult == GdrCompletionPollResult::UnknownRequest) {
                const auto status =
                    Status::OsApiError(fmt::format("unexpected GDR completion reqId({})", reqId));
                StopWithAsyncError("PollCompletion", status);
                return;
            }

            StopWithAsyncError("PollCompletion", Status::OsApiError("PollCompletion failed"));
            return;
        }

        if (HasAsyncError() || stopRequested_.load(std::memory_order_acquire)) { return; }

        const auto waitRc = channel_->WaitForCompletionEvent();
        if (waitRc == -ECANCELED) {
            if (HasAsyncError() || stopRequested_.load(std::memory_order_acquire)) { return; }
            continue;
        }
        if (waitRc != 0) {
            StopWithAsyncError("WaitForCompletionEvent",
                               MakeGdrStatus("WaitForCompletionEvent", waitRc));
            return;
        }
    }
}

void GdrStream::ShutdownBackgroundThreads()
{
    stopRequested_.store(true, std::memory_order_release);
    if (channel_) { channel_->InterruptCompletionWait(); }
    if (schedulerThread_.joinable()) { schedulerThread_.join(); }
    if (completionThread_.joinable()) { completionThread_.join(); }
}

void GdrStream::MarkOperationCompleted(uint64_t operationId)
{
    if (operationId == 0) { return; }

    const auto completedBefore = lastCompletedOperationId_.load(std::memory_order_acquire);
    if (operationId <= completedBefore) { return; }

    completionSlots_[operationId & kCompletionRingMask].operationId.store(
        operationId, std::memory_order_release);
    AdvanceCompletedOperations();
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
    stopRequested_.store(true, std::memory_order_release);
    if (channel_) { channel_->InterruptCompletionWait(); }
}

bool GdrStream::HasAsyncError() const { return asyncErrorSet_.load(std::memory_order_acquire); }

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
