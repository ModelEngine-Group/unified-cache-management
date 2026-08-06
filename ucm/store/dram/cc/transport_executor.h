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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_EXECUTOR_H
#define UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "bounded_queue.h"
#include "messages.h"
#include "status/status.h"

namespace UC::Dram {

using MemoryHandle = std::uint64_t;

enum class MemoryRegionType : std::uint8_t {
    HOST = 0,
    DEVICE,
};

class ITransportBackend {
public:
    virtual ~ITransportBackend() = default;
    virtual Expected<MemoryHandle> RegisterMemory(void* address, std::size_t length,
                                                  MemoryRegionType type) = 0;
    virtual Status UnregisterMemory(MemoryHandle handle) = 0;
    virtual TransmitCompleted Transmit(const ::UC::Dram::Transmit& command) noexcept = 0;
    virtual Status Connect(const ::UC::Dram::Connect& command) noexcept = 0;
    // Success synchronously proves that the old epoch can no longer access client memory.
    virtual Status Fence(const ::UC::Dram::FenceEpoch& command) noexcept = 0;
    virtual void Stop() = 0;
};

class TransportExecutor final {
public:
    struct Options {
        std::size_t workerCount{1};
        std::size_t nodeCount{0};
        std::size_t maxInflightRequestsPerNode{0};
        std::shared_ptr<ITransportBackend> backend;
        NodeEventPublisher publishEvent;
    };

    explicit TransportExecutor(Options options);
    ~TransportExecutor();

    TransportExecutor(const TransportExecutor&) = delete;
    TransportExecutor& operator=(const TransportExecutor&) = delete;

    Status Start();
    void Shutdown();

    // Consumes on success.
    Status TryPost(TransportCommand& command);

private:
    struct Worker {
        explicit Worker(std::size_t capacity) : queue(capacity) {}

        std::mutex mutex;
        std::condition_variable wake;
        BoundedQueue<TransportCommand> queue;
        std::thread thread;
    };

    void Execute(TransportCommand command) noexcept;
    void Run(Worker& worker) noexcept;

    Options options_;
    std::size_t commandQueueCapacity_{0};
    std::size_t fenceQueueCapacity_{0};
    std::vector<std::unique_ptr<Worker>> workers_;
    std::mutex admissionMutex_;
    std::size_t queuedCommands_{0};
    std::size_t queuedFences_{0};
    std::atomic<bool> acceptingCommands_{false};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_TRANSPORT_EXECUTOR_H
