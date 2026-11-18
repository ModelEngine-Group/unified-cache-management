/**
 * MIT License
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "kv_types.h"

namespace kv {

class KvClient {
public:
    // Destroys the client and releases its resources.
    virtual ~KvClient() = default;

    // Initializes routing, transports, and workers from the supplied configuration.
    virtual Status Init(const KvClientConfig& config) = 0;
    // Stops accepting requests, drains pending tasks, and releases client resources.
    virtual Status Shutdown() = 0;

    // Submits an asynchronous existence query and returns a task ID for retrieving its result.
    virtual Status QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId,
                              const IoOptions& options = {}) = 0;
    // Submits asynchronous reads into the supplied buffers and returns a task ID.
    virtual Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                             const IoOptions& options = {}) = 0;
    // Submits asynchronous writes. The optional prerequisite runs once on the client worker
    // before dispatch; its error completes the task without IO. Exceptions become INTERNAL_ERROR.
    virtual Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                              const IoOptions& options = {}) = 0;
    // Submits asynchronous deletion of the supplied keys and returns a task ID.
    virtual Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId,
                               const IoOptions& options = {}) = 0;

    // Checks completion without consuming the task, returning true also for an unknown task ID.
    virtual bool Check(TaskId taskId) = 0;
    // Waits for the result and consumes the task even on timeout; zero uses the default timeout.
    virtual Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) = 0;

    // Registers memory regions and returns the handles and tokens needed for IO.
    virtual Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                                   std::vector<RegisteredMemory>& registeredRegions) = 0;
    // Unregisters the supplied memory handles without waiting for tasks that reference them.
    // Handles referenced by queued or in-flight tasks must remain registered until those tasks
    // complete.
    virtual Status UnregisterRegions(const std::vector<MRHandle>& handles) = 0;
};

// Creates a client wired to the default transport and provider factories.
std::unique_ptr<KvClient> CreateKvClient();

}  // namespace kv
