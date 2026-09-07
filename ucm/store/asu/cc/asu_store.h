#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "asu_transport/types.h"

namespace UC::AsuStore {

struct Config {
    std::string mode{"client"};
    std::string role;
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
    UC::ASU::TransProviderType transProviderType{UC::ASU::TransProviderType::AICPU};
    std::string fakeBackendPath;
    std::uint64_t fakeBackendLatencyMs{1};
    std::uint64_t fakeBackendWorkerThreads{4};
    ssize_t sharedProviderMode{0};
    bool sc{true};
    std::unordered_map<std::string, std::string> clientAttrs;
};

}  // namespace UC::AsuStore
