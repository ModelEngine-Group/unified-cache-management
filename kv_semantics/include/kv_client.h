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
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "types.h"

namespace UC::ASU {

class AsuClient {
public:
    virtual ~AsuClient() = default;

    virtual Status Init(const AsuClientConfig& config) = 0;
    virtual Status Init(const std::string& configPath) = 0;
    virtual Status Shutdown() = 0;

    virtual Status QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId) = 0;
    virtual Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) = 0;
    virtual Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) = 0;
    virtual Status BatchLoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) = 0;
    virtual Status BatchStoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) = 0;
    virtual Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId) = 0;

    // Returns false only while the task is still in progress. Wait retrieves the final result.
    virtual bool Check(TaskId taskId) = 0;
    virtual Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) = 0;

    virtual Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                                   std::vector<RegisteredMemory>& registeredRegions) = 0;
    // Handles referenced by queued or in-flight tasks must remain registered until those tasks
    // complete.
    virtual Status UnregisterRegions(const std::vector<MRHandle>& handles) = 0;
};

// Creates a client wired to the default transport and provider factories.
std::unique_ptr<AsuClient> CreateAsuClient();

}  // namespace UC::ASU