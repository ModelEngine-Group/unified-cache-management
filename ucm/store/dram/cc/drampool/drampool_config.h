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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/transport.h"
#include "eviction_policy.h"
#include "status/status.h"

namespace UC::DramPool {

inline constexpr std::uint32_t kDefaultPollerPendingDepth = 64;
inline constexpr std::uint64_t kBytesPerGiB = 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kBytesPerMiB = 1024ULL * 1024ULL;
inline constexpr std::size_t kFlagBufferSlotAlignment = 64;
inline constexpr std::uint64_t kMillisecondsPerMinute = 60'000;
inline constexpr std::uint64_t kDefaultTtlMinutes = 120;
inline constexpr std::uint64_t kDefaultDumpTtlMs = kDefaultTtlMinutes * kMillisecondsPerMinute;
inline constexpr char kDefaultDramPoolRuntimeConfigPath[] = "examples/drampool.yaml";

struct DramPoolConfig {
    std::string runtimeConfigPath{kDefaultDramPoolRuntimeConfigPath};
    // Northbound KV control endpoint supplied by --addr.
    transport::Endpoint addr{};
    std::vector<std::string> nics{};
    // --pool-size-gb follows the project convention: the unit is GiB.
    std::uint64_t poolSizeGb{0};
    std::vector<std::uint64_t> poolBlockSizes{};
    std::vector<std::uint32_t> poolBlockProportions{};
    // Resolved by ParseCommandLine from poolSizeGb, block sizes, and proportions.
    std::vector<std::uint32_t> poolSlotCounts{};
    std::uint64_t defaultDumpTtlMs{kDefaultDumpTtlMs};

    // Transport internals are loaded from the runtime YAML file.
    std::vector<std::int32_t> transportDeviceIds{0};
    // Cluster-wide routing from the request channel address to the transport identity.
    std::unordered_map<std::string, transport::ManagerID> twoSidedToOneSided{};

    // Bounded handoff from RequestReceiveLoop to TaskWorker.
    std::uint32_t requestQueueDepth{65536};
    // Bounded handoff from TaskWorker to CompletionPoller.
    std::uint32_t completionQueueDepth{65536};
    std::uint32_t requestReceiverIdleWaitUs{100};

    std::uint32_t pollerPendingDepth{kDefaultPollerPendingDepth};

    // Local staging pool used only for response/flag RDMA writes.
    std::uint64_t flagBufferCapacityMb{64};
    std::uint64_t flagBufferSlotSizeBytes{64};
    // Resolved by ParseYamlConfig from capacity and slot layout.
    std::uint32_t flagBufferSlotCount{0};

    bool gcEnabled{true};
    std::uint32_t gcIntervalMs{1000};

    // MetadataManager internals are loaded from the runtime YAML file.
    EvictionPolicyType metadataPeriodicEvictionPolicy{EvictionPolicyType::TTL};
    EvictionPolicyType metadataDeepEvictionPolicy{EvictionPolicyType::POSITION};
    std::uint64_t metadataLeaseTimeMs{5000};
    double metadataDefaultEvictRatio{0.0};
    std::uint64_t metadataEvictPeriodMs{365ULL * 24ULL * 60ULL * 60ULL * 1000ULL};

    std::uint32_t opTimeoutMs{5000};

    std::string logLevel{"info"};
    std::string logDir{"./logs"};
    std::uint32_t logMaxFiles{10};
    std::uint32_t logMaxSizeMb{5};
};

// One DramPool daemon runs in each process; startup sets this once before Server::Init().
inline DramPoolConfig g_config{};

std::string BuildUsage(const char* program);
Status ParseCommandLine(int argc, char** argv, DramPoolConfig& config);
Status ParseDramPoolEndpoint(const std::string& name, const std::string& value,
                             transport::Endpoint& endpoint);
Status ParseYamlConfig(const std::string& path, DramPoolConfig& config);

}  // namespace UC::DramPool
