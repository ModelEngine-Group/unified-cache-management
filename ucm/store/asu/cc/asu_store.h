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
    std::string asuLocalIp;
    std::string asuNamePrefix{"asu"};
    std::vector<std::uint32_t> kvNsIds;
    std::vector<std::uint16_t> asuPorts;
    std::uint64_t defaultWaitTimeoutMs{100};
    std::uint64_t queryTimeoutMs{5};
    std::uint64_t loadTimeoutMs{100};
    std::uint64_t storeTimeoutMs{100};
    std::uint64_t maxInflightTasks{1024};
    std::uint64_t maxInflightBytes{1ULL << 30};
    std::vector<std::size_t> tensorSizes;
    std::size_t shardSize{0};
    std::size_t blockSize{0};
    std::int32_t deviceId{-1};
    std::string memoryType;
    std::string tensorLayout;
    UC::ASU::TransProviderType transProviderType{UC::ASU::TransProviderType::AICPU};
    std::string fakeBackendPath;
    std::uint64_t fakeBackendLatencyMs{1};
    std::unordered_map<std::string, std::string> clientAttrs;
};

class AsuBackend {
public:
    virtual ~AsuBackend() = default;
    virtual UC::ASU::Status Init(const Config& config) = 0;
    virtual UC::ASU::Status Init(const std::string& configPath) = 0;
    virtual UC::ASU::Status Shutdown() = 0;
    virtual UC::ASU::Status Query(const std::vector<UC::ASU::CacheKey>& keys,
                                  const UC::ASU::QueryOptions& options,
                                  UC::ASU::QueryResult& result) = 0;
    virtual UC::ASU::Status LoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                                      UC::ASU::TaskId& taskId) = 0;
    virtual UC::ASU::Status StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                                       UC::ASU::TaskId& taskId) = 0;
    virtual UC::ASU::Status DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                                        UC::ASU::TaskId& taskId) = 0;
    virtual UC::ASU::Status Check(UC::ASU::TaskId taskId, UC::ASU::TaskResult& result) = 0;
    virtual UC::ASU::Status Wait(UC::ASU::TaskId taskId, std::uint64_t timeoutMs,
                                 UC::ASU::TaskResult& result) = 0;
    virtual UC::ASU::Status RegisterRegions(const std::vector<UC::ASU::MemoryRegion>& regions,
                                            std::vector<UC::ASU::RegisterResult>& results) = 0;
};

}  // namespace UC::AsuStore
