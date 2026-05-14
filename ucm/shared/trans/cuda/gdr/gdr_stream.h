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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
    static constexpr size_t kOperationRingCapacity = 8192;
    static constexpr size_t kOperationRingMask = kOperationRingCapacity - 1;
    static constexpr size_t kCompletionRingCapacity = 8192;
    static constexpr size_t kCompletionRingMask = kCompletionRingCapacity - 1;
    static constexpr size_t kSchedulerBatchSize = 128;
    static constexpr size_t kRingPushSpinCount = 10000;
    static_assert((kOperationRingCapacity & kOperationRingMask) == 0,
                  "operation ring capacity must be a power of two");
    static_assert((kCompletionRingCapacity & kCompletionRingMask) == 0,
                  "completion ring capacity must be a power of two");

    // Says if a queued operation is a wait or a copy.
    enum class OperationType {
        Wait,  // Wait for a CUDA event before later work can run.
        Copy,  // Send one GDR copy.
    };

    // One item in the stream queue. The caller adds it, SchedulerLoop() handles it in order.
    struct Operation {
        OperationType type;
        uint64_t operationId;        // Order number for this stream item.
        cudaEvent_t event{nullptr};  // Event to wait for when this is a wait.
        void* dst;                   // Copy destination when this is a copy.
        const void* src;             // Copy source when this is a copy.
        size_t size;                 // Bytes to copy when this is a copy.
        GdrCopyKind kind;            // Copy direction when this is a copy.
    };

    enum class SubmitResult {
        Submitted,  // This copy was accepted, so the scheduler can move on.
        Waiting,    // The send queue is full, so the scheduler should wait for a completion.
        Error,      // Submit failed, so the stream must stop with an error.
    };

    struct CompletionSlot {
        std::atomic<uint64_t> operationId{0};
    };

    Status SubmitAsync(void* dst, const void* src, size_t size, GdrCopyKind kind);
    Status PushOperation(const Operation& op);
    size_t PopOperationBatch(Operation* out, size_t maxCount);
    void ResetOperationRing();
    void SchedulerLoop();
    void CompletionLoop();
    void ShutdownBackgroundThreads();
    SubmitResult SubmitCopyOperationFromQueue(const Operation& op);
    void MarkOperationCompleted(uint64_t operationId);
    void AdvanceCompletedOperations();
    void MarkOperationFailed(uint64_t operationId, Status status);
    void StopWithAsyncError(const char* source, Status status);
    bool HasAsyncError() const;
    Status AsyncErrorLocked() const;
    std::optional<Status> TakeCompletedOperationError();

private:
    std::shared_ptr<GdrCopyChannel> channel_{nullptr};
    std::thread schedulerThread_;   // Runs waits and sends copies in stream order.
    std::thread completionThread_;  // Polls the GDR completion queue for sent copies.

    std::mutex mutex_;
    std::unique_ptr<Operation[]> operationRing_{
        std::make_unique<Operation[]>(kOperationRingCapacity)};
    std::atomic<uint64_t> operationRingHead_{0};
    std::atomic<uint64_t> operationRingTail_{0};
    std::unique_ptr<CompletionSlot[]> completionSlots_{
        std::make_unique<CompletionSlot[]>(kCompletionRingCapacity)};
    std::map<uint64_t, Status> failedOperations_;  // Non-fatal per-operation failures.
    std::optional<Status> asyncError_;  // First fatal async error that stops the whole stream.
    std::atomic<bool> asyncErrorSet_{false};

    std::atomic<uint64_t> nextOperationId_{1};
    std::atomic<uint64_t> lastCompletedOperationId_{0};
    int32_t deviceId_{-1};
    std::string nicName_{"mlx5_0"};
    std::atomic<bool> stopRequested_{
        false};  // Tells background threads to exit after current work ends or fails.
};

}  // namespace UC::Trans

#endif
