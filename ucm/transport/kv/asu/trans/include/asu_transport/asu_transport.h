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
#include "asu_transport/types.h"

namespace UC::ASU {

struct AsuEndpoint {
    std::string ip;
    std::uint16_t port{0};
    Protocol protocol{Protocol::ROCE};
    std::int32_t numa_node{-1};
    std::int32_t device_id{-1};
    std::string hca_name;
    std::uint8_t hca_port{1};
    std::unordered_map<std::string, std::string> attrs;
};

struct TransportConfig {
    // TODO: 拆分Config，按逻辑模块细化
    std::string asu_name;
    AsuId asu_id{0};
    std::vector<AsuEndpoint> endpoints;

    std::uint32_t query_qp_num{1};
    std::uint32_t load_qp_num{4};
    std::uint32_t store_qp_num{2};

    std::uint32_t max_inflight_tasks{1024};
    std::uint64_t max_inflight_bytes{1ULL << 30};

    std::uint32_t max_query_inflight{256};
    std::uint32_t max_load_inflight{512};
    std::uint32_t max_store_inflight{256};

    std::uint64_t query_timeout_ms{5};
    std::uint64_t load_timeout_ms{100};
    std::uint64_t store_timeout_ms{100};

    bool enable_device_direct{true};
    bool enable_host_fallback{false};
    bool preconnect{true};
    bool bind_cq_poller{true};

    std::unordered_map<std::string, std::string> attrs;
};

class AsuTransport {
public:
    virtual ~AsuTransport() = default;

    virtual Status Init(const TransportConfig& config) = 0;
    virtual Status Shutdown() = 0;
    virtual Status CheckHealth() = 0;

    virtual Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                         QueryResult& result) = 0;
    virtual Status QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                              TaskId& task_id) = 0;

    virtual Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) = 0;
    virtual Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id) = 0;
    virtual Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id) = 0;

    // Best-effort cancellation, does not interrupt underlying UB/RoCE IO
    virtual Status Cancel(TaskId task_id) = 0;
    virtual Status Check(TaskId task_id, TaskResult& result) = 0;
    virtual Status Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result) = 0;

    virtual Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                                   std::vector<RegisterResult>& results) = 0;

    virtual Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                         std::vector<RegisterResult>& results) = 0;

    virtual Status UnregisterRegions(const std::vector<MRHandle>& handles) = 0;
};

std::unique_ptr<AsuTransport> CreateAsuTransport();

}  // namespace UC::ASU
