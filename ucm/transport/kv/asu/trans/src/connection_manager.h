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

#include <acl/acl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
#include "kv_protocol.h"

namespace UC::ASU {

using ConnectionHandle = void*;

enum class RoutingPolicy : std::uint8_t {
    ROUND_ROBIN,
    LEAST_LOADED,
};

class ConnectionChannel;
class ConnectionGroup;
struct ScatterGatherEntry;

class ConnectionManager {
public:
    ConnectionManager();
    ~ConnectionManager();

    using CreateConnectionFunc =
        std::function<std::vector<ConnectionHandle>(const AsuEndpoint&, std::uint32_t)>;
    using DeleteConnectionsFunc =
        std::function<std::vector<Status>(const std::vector<ConnectionHandle>&)>;

    void SetConnectionOps(CreateConnectionFunc create_fn, DeleteConnectionsFunc delete_fn);
    void PrepareForInit();

    Status AddGroup(const AsuEndpoint& endpoint, std::uint32_t qp_num);
    Status Shutdown();

    ConnectionChannel* SelectConnection();
    void SetRoutingPolicy(RoutingPolicy policy);
    void ReportFailure(ConnectionChannel* channel);

    void StartRecoverLoop();
    void StopRecoverLoop();

    std::int64_t TotalInflightCount();

private:
    std::vector<std::unique_ptr<ConnectionGroup>> groups_;
    std::shared_mutex structureMu_;  // shared_mutex allows concurrent reads

    // Flat cache for fast channel selection, rebuilt on structure changes.
    std::vector<ConnectionChannel*> channelCache_;
    std::mutex channelCacheMu_;
    std::atomic<bool> cacheDirty_{false};

    std::atomic<bool> shuttingDown_{false};

    std::atomic<std::uint32_t> rrIndex_{0};
    RoutingPolicy routingPolicy_{RoutingPolicy::ROUND_ROBIN};
    static constexpr std::uint32_t kMaxInflightPerChannel = 256;
    static constexpr std::uint32_t kFailureThreshold = 2;
    static constexpr std::uint64_t kDrainTimeoutMs = 30000;
    static constexpr std::uint64_t kRecoverIntervalMs = 100;

    std::thread recoverWorker_;
    std::atomic<bool> stopRecover_{false};

    std::shared_mutex drainMu_;
    std::vector<ConnectionChannel*> drainList_;

    CreateConnectionFunc createFn_;
    DeleteConnectionsFunc deleteFn_;

    void RecoverLoop();
    void RebuildChannelCache();
    ConnectionChannel* SelectByRoundRobin();
    ConnectionChannel* SelectByLeastLoaded();
};

struct SendIoBatch {
    ConnectionHandle connectionHandle{nullptr};
    const ScatterGatherEntry* sendSge{nullptr};
    aclrtStream stream{nullptr};
};

std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, std::uint32_t kernelCount,
                         std::uint32_t quietCount);

}  // namespace UC::ASU
