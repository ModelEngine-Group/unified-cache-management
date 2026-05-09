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
#ifndef UNIFIEDCACHE_TRANS_GDR_STREAM_H
#define UNIFIEDCACHE_TRANS_GDR_STREAM_H

#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <cuda_runtime_api.h>
#include "gdr_copy.h"
#include "trans/stream.h"

namespace UC::Trans {

class GdrStream : public Stream {
public:
    ~GdrStream() override;

    Status Setup() override;

    Status DeviceToHost(void* device, void* host, size_t size) override;
    Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHost(void* device[], void* host, size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device, void* host, size_t size) override;
    Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) override;

    Status HostToDevice(void* host, void* device, size_t size) override;
    Status HostToDevice(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDevice(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device, size_t size) override;
    Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) override;

    Status AppendCallback(std::function<void(bool)> cb) override;
    Status Synchronized() override;
    Status WaitEvent(void* event) override;

private:
    // Says if a queued operation is a wait or a copy.
    enum class OperationType {
        Wait,  // Wait for a CUDA event before later work can run.
        Copy,  // Send one GDR copy.
    };

    // One item in the stream queue. The caller adds it, SchedulerLoop() handles it in order.
    struct Operation {
        OperationType type;
        uint64_t operationId;      // Order number for this stream item.
        cudaEvent_t event{nullptr};  // Event to wait for when this is a wait.
        void* dst;                 // Copy destination when this is a copy.
        const void* src;           // Copy source when this is a copy.
        size_t size;               // Bytes to copy when this is a copy.
        GdrCopyKind kind;          // Copy direction when this is a copy.
    };

    enum class SubmitResult {
        Submitted,  // This copy was accepted, so the scheduler can move on.
        Waiting,   // The send queue is full, so the scheduler should wait for a completion.
        Error,     // Submit failed, so the stream must stop with an error.
    };

    Status SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind);
    void SchedulerLoop();
    void CompletionLoop();
    void ShutdownBackgroundThreads();
    SubmitResult SubmitCopyOperationFromQueue(const Operation& op);
    void MarkOperationCompleted(uint64_t operationId);
    void MarkOperationFailed(uint64_t operationId, Status status);
    void StopWithAsyncError(const char* source, Status status);
    bool HasAsyncError() const;
    bool IsIdle() const;
    Status AsyncError() const;
    std::optional<Status> TakeCompletedOperationError();

private:
    std::shared_ptr<GdrCopyChannel> channel_{nullptr};
    std::thread schedulerThread_;  // Runs waits and sends copies in stream order.
    std::thread completionThread_;  // Polls the GDR completion queue for sent copies.
    
    std::mutex mutex_;
    std::condition_variable cv_;  // Wakes waiting threads when stream state changes.
    std::deque<Operation> operationsQueue_;  // Waits and copies queued by the caller.
    std::unordered_map<uint64_t, uint64_t> inflightRequestOperations_;  // Submitted GDR copies that are still running.
    std::unordered_set<uint64_t> completedOperations_;  // Finished operations waiting for earlier ones to finish too.
    std::map<uint64_t, Status> failedOperations_;  // Per-operation failures that do not stop the whole stream.
    std::optional<Status> asyncError_;  // First fatal async error that stops the whole stream.

    uint64_t nextOperationId_{1}; 
    uint64_t lastCompletedOperationId_{0};
    int32_t deviceId_{-1};
    std::string nicName_{"mlx5_0"};
    bool stopRequested_{false};  // Tells background threads to exit after current work ends or fails.
    bool schedulerReady_{false};  // Scheduler thread has started.
    bool completionThreadReady_{false};  // Completion thread has started.
    bool schedulerWaitingOnEvent_{false};  // Scheduler thread is blocked in cudaEventSynchronize.
};

}  // namespace UC::Trans

#endif
