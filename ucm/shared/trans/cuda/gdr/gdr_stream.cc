#include "gdr_stream.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "logger/logger.h"

namespace {

constexpr auto kCompletionPollInterval = std::chrono::milliseconds(1);

std::string ParseStringEnv(const char* name, const char* defaultValue)
{
    const auto* value = std::getenv(name);
    if (!value || value[0] == '\0') { return defaultValue; }
    return value;
}

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
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stopRequested_ = true;
        cv_.notify_all();
    }
    if (schedulerThread_.joinable()) { schedulerThread_.join(); }
    if (completionThread_.joinable()) { completionThread_.join(); }
}

Status GdrStream::Setup()
{
    nicName_ = ParseStringEnv("UCM_GDR_NIC_NAME", "mlx5_0");

    const auto ret = cudaGetDevice(&deviceId_);
    if (ret != cudaSuccess) { return Status{ret, cudaGetErrorString(ret)}; }

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
        {
            std::lock_guard<std::mutex> lock{mutex_};
            stopRequested_ = true;
            cv_.notify_all();
        }
        if (schedulerThread_.joinable()) { schedulerThread_.join(); }
        if (completionThread_.joinable()) { completionThread_.join(); }
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
        if (schedulerThread_.joinable()) { schedulerThread_.join(); }
        if (completionThread_.joinable()) { completionThread_.join(); }
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

    std::lock_guard<std::mutex> lock{mutex_};
    if (HasAsyncError()) { return AsyncError(); }
    if (!event) { return Status::OK(); }

    operationsQueue_.push_back(
        Operation{OperationType::Wait, nextOperationId_++, static_cast<cudaEvent_t>(event),
                  nullptr, nullptr, 0, GdrMemcpyHostToDevice});
    cv_.notify_all();
    return Status::OK();
}

// Queue one copy. The scheduler thread will send it later.
Status GdrStream::SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind)
{
    if (!channel_) { return Status::Error("GDR channel is not ready"); }

    std::lock_guard<std::mutex> lock{mutex_};
    if (HasAsyncError()) { return AsyncError(); }

    operationsQueue_.push_back(
        Operation{OperationType::Copy, nextOperationId_++, nullptr, dst, src, size, kind});
    cv_.notify_all();
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

    // loop
    for (;;) {
        Operation op{OperationType::Copy, 0, nullptr, nullptr, nullptr, 0,
                     GdrMemcpyHostToDevice};
        bool hasOp = false;
        size_t inflightBefore = 0;
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(
                lock,
                [this] { return stopRequested_ || HasAsyncError() || !operationsQueue_.empty(); });

            if (HasAsyncError()) { return; }
            if (stopRequested_ && operationsQueue_.empty()) { return; }
            if (!operationsQueue_.empty()) {
                op = operationsQueue_.front();
                hasOp = true;
                inflightBefore = inflightRequestOperations_.size();
            }
        }

        if (!hasOp) { continue; }

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
                if (!operationsQueue_.empty()
                    && operationsQueue_.front().operationId == op.operationId) {
                    operationsQueue_.pop_front();
                }
                MarkOperationCompleted(op.operationId);
                cv_.notify_all();
            }
            continue;
        }

        const auto submitResult = SubmitCopyOperationFromQueue(op);
        if (submitResult == SubmitResult::Submitted) { continue; }
        if (submitResult == SubmitResult::Error) { return; }

        std::unique_lock<std::mutex> lock{mutex_};
        cv_.wait_for(lock, kCompletionPollInterval, [this, inflightBefore] {
            return stopRequested_ || HasAsyncError()
                || inflightRequestOperations_.size() < inflightBefore;
        });
        if (HasAsyncError()) { return; }
    }
}

GdrStream::SubmitResult GdrStream::SubmitCopyOperationFromQueue(const Operation& op)
{
    uint64_t reqId = 0;
    const auto rc = channel_->GdrMemcpyAsync(op.dst, op.src, op.size, op.kind, &reqId);
    std::lock_guard<std::mutex> lock{mutex_};
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
        cv_.notify_all();
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
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] {
                return stopRequested_ || HasAsyncError()
                    || !inflightRequestOperations_.empty();
            });

            if (HasAsyncError()) { return; }
            if (stopRequested_ && inflightRequestOperations_.empty()) { return; }
        }

        uint64_t reqId = 0;
        const auto rc = channel_->PollCompletion(&reqId);
        if (rc == 0) {
            std::lock_guard<std::mutex> lock{mutex_};
            const auto it = inflightRequestOperations_.find(reqId);
            if (it == inflightRequestOperations_.end()) {
                const auto status =
                    Status::OsApiError(fmt::format("unexpected GDR completion reqId({})", reqId));
                StopWithAsyncError("PollCompletion reqId mismatch", status);
                return;
            }
            const uint64_t operationId = it->second;
            inflightRequestOperations_.erase(it);
            MarkOperationCompleted(operationId);
            cv_.notify_all();
            continue;
        }
        if (rc == -EAGAIN) {
            std::this_thread::sleep_for(kCompletionPollInterval);
            continue;
        }

        std::lock_guard<std::mutex> lock{mutex_};
        StopWithAsyncError("PollCompletion", MakeGdrStatus("PollCompletion", rc));
        return;
    }
}

void GdrStream::MarkOperationCompleted(uint64_t operationId)
{
    if (operationId <= lastCompletedOperationId_) { return; }

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
