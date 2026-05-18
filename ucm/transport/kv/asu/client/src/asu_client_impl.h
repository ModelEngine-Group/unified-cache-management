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

#include <memory>
#include <unordered_map>
#include "asu_client/asu_client.h"

namespace UC::ASU {

class AsuClientImpl final : public AsuClient {
public:
    explicit AsuClientImpl(TransportFactory factory);
    ~AsuClientImpl() override;

    Status Init(const AsuClientConfig& config) override;
    Status Shutdown() override;

    Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                 QueryResult& result) override;

    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) override;
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) override;
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id) override;

    Status Check(TaskId task_id, TaskResult& result) override;
    Status Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result) override;

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisterResult>& results) override;
    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    struct Router {
        // TODO: 实现DHT和路由
    };
    struct ViewSnapshot {
        std::shared_ptr<Router> router;
        std::unordered_map<AsuId, std::shared_ptr<AsuTransport>> transports;
    };
    TransportFactory transport_factory_;
};

}  // namespace UC::ASU
