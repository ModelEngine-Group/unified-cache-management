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
#pragma once

#include "asu_transport/asu_transport.h"

namespace UC::ASU {

class AsuTransportImpl final : public AsuTransport {
public:
    AsuTransportImpl() = default;
    ~AsuTransportImpl() override;

    Status Init(const TransportConfig& config) override;
    Status Shutdown() override;

    Status CheckHealth() override;

    Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                 QueryResult& result) override;
    Status QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                      TaskId& task_id) override;
    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) override;
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) override;
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id) override;

    Status Cancel(TaskId task_id) override;
    Status Check(TaskId task_id, TaskResult& result) override;
    Status Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result) override;

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisterResult>& results) override;

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                 std::vector<RegisterResult>& results) override;

    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    TransportConfig config_;
};

}  // namespace UC::ASU
