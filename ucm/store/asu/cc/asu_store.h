// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "kv_types.h"

namespace UC::AsuStore {

struct Config {
    // Configuration key: enable_metrics.
    // Controls this store's native ASU metrics and KV backend installation.
    // Does not filter KV samples or disable a process-wide KV backend.
    bool enableMetrics{true};
    std::string mode{"client"};
    std::string configPath;
    std::string clientId{"ucm-asu-store"};
    std::string uniqueId;
    std::vector<std::string> viewServiceAddrs;
    std::vector<ssize_t> asuIds;
    std::vector<std::string> asuIps;
    std::vector<ssize_t> asuPorts;
    std::string localIp;
    std::string asuNamePrefix{"asu"};
    std::vector<std::uint32_t> kvNsIds;
    std::uint64_t waitTimeoutMs{100};
    std::uint64_t queryTimeoutMs{500};
    std::uint64_t maxErrorCount{2};
    std::uint64_t clientMaxInflightTasks{1024};
    bool useBatchIoForSingleKey{true};
    std::uint64_t transportMaxInflightTasks{1024};
    std::uint64_t completionPollSpinLimit{16};
    std::uint64_t maxInflightBytes{1ULL << 30};
    std::vector<std::size_t> tensorSizes;
    std::vector<std::uintptr_t> gpuKvBufferAddrs;
    std::vector<std::size_t> gpuKvBufferSizes;
    std::size_t shardSize{0};
    std::size_t blockSize{0};
    std::int32_t deviceId{-1};
    std::string tensorLayout;
    kv::TransProviderType transProviderType{kv::TransProviderType::AICPU};
    std::string aicpuHcommProtocol;
    std::vector<std::string> aicpuLocalAddrs;
    std::optional<std::uint64_t> aicpuSendTimeoutMs;
    // HCCP KV-over-RoCE options, read from asu_hccp_kv_roce_lib_path and
    // asu_hccp_kv_roce_max_connections when the backend is hccp_kv_over_roce.
    std::string hccpKvRoceLibPath;
    std::optional<std::uint64_t> hccpKvRoceMaxConnections;
    // Numeric overrides for HccpKvRoceInitConfig, keyed by the ABI field name
    // in snake_case (for example "mem_size"). The provider fills every field
    // that is absent here from the defaults documented in the ABI header.
    std::unordered_map<std::string, std::int64_t> hccpKvRoceInit;
    std::string fakeBackendPath;
    std::uint64_t fakeBackendLatencyMs{1};
    std::uint64_t fakeBackendWorkerThreads{4};
    bool fakeBackendCompleteImmediately{false};
    ssize_t sharedProviderMode{0};
    bool sc{true};
    std::unordered_map<std::string, std::string> clientAttrs;
};

}  // namespace UC::AsuStore
