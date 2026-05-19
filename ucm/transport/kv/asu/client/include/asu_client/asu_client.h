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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "asu_transport/asu_transport.h"

namespace UC::ASU {

struct AsuClientConfig {
    std::string client_id;
    std::vector<std::string> view_service_addrs;

    std::vector<TransportConfig> transport_configs;

    std::uint64_t default_wait_timeout_ms{100};
    std::unordered_map<std::string, std::string> attrs;
};

class AsuClient {
public:
    virtual ~AsuClient() = default;

    virtual Status Init(const AsuClientConfig& config) = 0;
    virtual Status Shutdown() = 0;

    virtual Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                         QueryResult& result) = 0;

    virtual Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) = 0;
    virtual Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) = 0;
    virtual Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id) = 0;

    virtual Status Check(TaskId task_id, TaskResult& result) = 0;
    virtual Status Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result) = 0;

    virtual Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                                   std::vector<RegisterResult>& results) = 0;
    virtual Status UnregisterRegions(const std::vector<MRHandle>& handles) = 0;
};

using TransportFactory = std::function<std::unique_ptr<AsuTransport>()>;

std::unique_ptr<AsuClient> CreateAsuClient(TransportFactory factory = CreateAsuTransport);

}  // namespace UC::ASU
