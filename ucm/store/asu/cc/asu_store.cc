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
#include "asu_store.h"
#include <algorithm>
#include <any>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "asu_client/asu_client.h"
#include "logger/logger.h"
#include "trans/event.h"
#include "ucmstore_v1.h"

namespace UC::AsuStore {

enum class TensorLayout { MLA, GQA, HMA };

namespace {

using AsuStatus = UC::ASU::Status;
using AsuStatusCode = UC::ASU::StatusCode;

TensorLayout ParseTensorLayout(const std::string& layout)
{
    if (layout == "gqa") { return TensorLayout::GQA; }
    if (layout == "hma") { return TensorLayout::HMA; }
    return TensorLayout::MLA;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t HashAsuKey(const Detail::BlockId& block)
{
    static Detail::BlockIdHasher hasher;
    return static_cast<std::uint64_t>(hasher(block));
}

UC::ASU::CacheKey MakeAsuKey(const Detail::BlockId& block)
{
    const auto hash = HashAsuKey(block);
    UC::ASU::CacheKey key{};
    std::memcpy(key.data(), &hash, key.size());
    return key;
}

Status ConvertStatus(const AsuStatus& status)
{
    if (status.ok()) { return Status::OK(); }

    const auto& message = status.message;
    switch (status.code) {
        case AsuStatusCode::INVALID_ARGUMENT: return Status::InvalidParam(message);
        case AsuStatusCode::NOT_FOUND:
        case AsuStatusCode::TASK_NOT_FOUND: return Status::NotFound();
        case AsuStatusCode::TIMEOUT: return Status::Timeout();
        case AsuStatusCode::BUFFER_NOT_SUPPORTED:
        case AsuStatusCode::UNSUPPORTED: return Status::Unsupported();
        case AsuStatusCode::RESOURCE_BUSY:
        case AsuStatusCode::IN_PROGRESS: return Status::Retry();
        default: return Status::Error(message);
    }
}

void LogAsuStatus(const char* operation, const AsuStatus& status)
{
    if (status.ok()) { return; }

    UC_ERROR("ASU {} failed: code={}, message={}.", operation, static_cast<int>(status.code),
             status.message);
}

Status WaitPrerequisiteEvent(std::uintptr_t eventHandle)
{
    return Trans::Event{eventHandle}.Synchronize();
}

const char* TransProviderBackendName(UC::ASU::TransProviderType providerType)
{
    switch (providerType) {
        case UC::ASU::TransProviderType::FAKE: return "fake";
        case UC::ASU::TransProviderType::AIV: return "aiv";
        case UC::ASU::TransProviderType::AICPU: return "aicpu";
        case UC::ASU::TransProviderType::UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

UC::ASU::TransProviderType ParseTransProviderBackend(std::string backend)
{
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    if (backend == "FAKE") { return UC::ASU::TransProviderType::FAKE; }
    if (backend == "AIV") { return UC::ASU::TransProviderType::AIV; }
    if (backend == "AICPU") { return UC::ASU::TransProviderType::AICPU; }
    return UC::ASU::TransProviderType::UNSUPPORTED;
}

bool TryGetStringLike(const Detail::Dictionary& inConfig, const std::string& key,
                      std::string& value)
{
    if (!inConfig.Contains(key)) { return false; }
    try {
        inConfig.Get(key, value);
        return true;
    } catch (const std::bad_any_cast&) {
    }
    try {
        bool boolValue = false;
        inConfig.Get(key, boolValue);
        value = boolValue ? "true" : "false";
        return true;
    } catch (const std::bad_any_cast&) {
    }
    try {
        ssize_t numberValue = 0;
        inConfig.GetNumber(key, numberValue);
        value = std::to_string(numberValue);
        return true;
    } catch (const std::bad_any_cast&) {
    }
    return false;
}

void ReadClientAttr(const Detail::Dictionary& inConfig, const std::string& yamlKey,
                    const std::string& attrKey, Config& config)
{
    std::string value;
    if (TryGetStringLike(inConfig, yamlKey, value)) { config.clientAttrs[attrKey] = value; }
}

}  // namespace

UC::ASU::TransportConfig BuildTransportConfig(const Config& config, std::size_t index)
{
    UC::ASU::TransportConfig transportConfig;
    transportConfig.asuId = static_cast<UC::ASU::AsuId>(config.asuIds[index]);
    transportConfig.asuName = config.asuNamePrefix + "-" + std::to_string(config.asuIds[index]);
    transportConfig.deviceId = config.deviceId;
    transportConfig.timeoutMs = config.waitTimeoutMs;
    transportConfig.maxErrorCount = static_cast<std::uint32_t>(config.maxErrorCount);
    transportConfig.maxInflightTasks = static_cast<std::uint32_t>(config.transportMaxInflightTasks);
    transportConfig.completionPollSpinLimit =
        static_cast<std::size_t>(config.completionPollSpinLimit);
    transportConfig.maxInflightBytes = config.maxInflightBytes;
    transportConfig.providerType = config.transProviderType;

    // Set for all backends including fake
    const auto kvNsIndex = config.uniqueId.find("_fawa_wa") == std::string::npos ? 0 : 1;
    transportConfig.attrs["kv_ns_id"] = std::to_string(config.kvNsIds[kvNsIndex]);
    if (!config.localIp.empty()) { transportConfig.attrs["localIp"] = config.localIp; }
    transportConfig.attrs["sc"] = config.sc ? "true" : "false";

    if (!config.asuIps.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = config.asuIps[index];
        endpoint.port = static_cast<std::uint16_t>(config.asuPorts[index]);
        transportConfig.endpoints.emplace_back(std::move(endpoint));
    }
    if (config.transProviderType == UC::ASU::TransProviderType::FAKE) {
        const auto fakeDeviceId = config.deviceId >= 0 ? config.deviceId : 0;
        transportConfig.deviceId = fakeDeviceId;
        transportConfig.attrs.try_emplace("kernel_count", "1");
        transportConfig.attrs.try_emplace("quiet_count", "1");
        transportConfig.attrs.try_emplace("dtype", "0");
        transportConfig.attrs.try_emplace("dspec", "0");
        transportConfig.attrs.try_emplace("lr", "false");
        transportConfig.attrs["fake_backend.path"] = config.fakeBackendPath;
        transportConfig.attrs["fake_backend.latency_ms"] =
            std::to_string(config.fakeBackendLatencyMs);
        transportConfig.attrs["fake_backend.worker_threads"] =
            std::to_string(config.fakeBackendWorkerThreads);
        transportConfig.attrs["fake_backend.device_id"] = std::to_string(fakeDeviceId);
        if (transportConfig.endpoints.empty()) {
            UC::ASU::AsuEndpoint endpoint;
            endpoint.ip = "fake_backend";
            endpoint.port = 19001;
            endpoint.protocol = UC::ASU::Protocol::TCP;
            transportConfig.endpoints.emplace_back(std::move(endpoint));
        }
    }
    return transportConfig;
}

UC::ASU::AsuClientConfig BuildAsuClientConfig(const Config& config)
{
    UC::ASU::AsuClientConfig asuConfig;
    asuConfig.clientId = config.clientId;
    asuConfig.viewServiceAddrs = config.viewServiceAddrs;
    asuConfig.maxInflightTasks = static_cast<std::uint32_t>(config.clientMaxInflightTasks);
    asuConfig.defaultWaitTimeoutMs = config.waitTimeoutMs;
    asuConfig.timeoutMs = config.waitTimeoutMs;
    asuConfig.sharedProviderMode =
        static_cast<UC::ASU::SharedProviderMode>(config.sharedProviderMode);
    asuConfig.attrs = config.clientAttrs;
    asuConfig.transportConfigs.reserve(config.asuIds.size());
    for (std::size_t i = 0; i < config.asuIds.size(); ++i) {
        asuConfig.transportConfigs.emplace_back(BuildTransportConfig(config, i));
    }
    return asuConfig;
}

class AsuStore final : public StoreV1 {
public:
#ifdef ASU_BUILD_TESTS
    using ClientFactory = std::function<std::unique_ptr<UC::ASU::AsuClient>(const Config&)>;

    void SetClientFactory(ClientFactory factory) { clientFactory_ = std::move(factory); }
#endif

    ~AsuStore() override
    {
        if (client_) {
            auto status = client_->Shutdown();
            if (!status.ok()) { UC_ERROR("Failed to shutdown ASU client: {}.", status.message); }
        }
    }

    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        NormalizeAsuShardConfig(config);
        auto status = CheckConfig(config);
        if (status.Failure()) { return status; }

        tensorLayout_ = ParseTensorLayout(config.tensorLayout);
        config_ = std::move(config);
        client_ = CreateClient(config_);

        auto asuStatus = config_.configPath.empty() ? client_->Init(BuildAsuClientConfig(config_))
                                                    : client_->Init(config_.configPath);
        if (!asuStatus.ok()) {
            UC_ERROR("Failed to init ASU client: {}.", asuStatus.message);
            client_.reset();
            return ConvertStatus(asuStatus);
        }

        if (config_.role == "worker" && !config_.gpuKvBufferAddrs.empty()) {
            std::vector<KVCacheRegistration> registrations;
            registrations.reserve(config_.gpuKvBufferAddrs.size());
            for (std::size_t index = 0; index < config_.gpuKvBufferAddrs.size(); ++index) {
                registrations.push_back(
                    {config_.gpuKvBufferAddrs[index], config_.gpuKvBufferSizes[index]});
            }
            status = RegisterKVCaches(registrations.data(), registrations.size());
            if (status.Failure()) {
                LogAsuStatus("shutdown after registration failure", client_->Shutdown());
                client_.reset();
                return status;
            }
        }

        ShowConfig(config_);
        return Status::OK();
    }

    std::string Readme() const override { return "AsuStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        return QueryBlocks(blocks, num);
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return static_cast<ssize_t>(-1); }

        auto keys = BuildBlockKeys(blocks, num);
        UC::ASU::QueryResult queryResult;
        auto status = Query(keys, queryResult);
        if (status.code == AsuStatusCode::TIMEOUT) { return static_cast<ssize_t>(-1); }
        if (!status.ok()) {
            LogAsuStatus("prefix query", status);
            return ConvertStatus(status);
        }
        if (queryResult.prefixHitKeys > keys.size()) {
            return Status::Error("ASU prefix hit keys out of range");
        }

        if (queryResult.prefixHitKeys == 0) { return static_cast<ssize_t>(-1); }
        return static_cast<ssize_t>(queryResult.prefixHitKeys - 1);
    }

    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        (void)blocks;
        (void)num;
    }

private:
    Status RegisterKVCaches(const UC::KVCacheRegistration* registrations, std::size_t count)
    {
        if (!client_) { return Status::Error("ASU client is not initialized"); }

        std::vector<UC::ASU::MemoryRegion> regions;
        regions.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (registrations[index].addr == 0 || registrations[index].size == 0) { continue; }
            UC::ASU::MemoryRegion region;
            region.memoryType = UC::ASU::MemoryType::DEVICE;
            region.addr = static_cast<std::uint64_t>(registrations[index].addr);
            region.size = static_cast<std::uint64_t>(registrations[index].size);
            region.deviceId = config_.deviceId;
            regions.emplace_back(region);
        }
        if (regions.empty()) { return Status::OK(); }

        std::vector<UC::ASU::RegisteredMemory> registeredRegions;
        auto status = client_->RegisterRegions(regions, registeredRegions);
        if (!status.ok()) {
            LogAsuStatus("register persistent regions", status);
            return ConvertStatus(status);
        }
        if (registeredRegions.size() != regions.size()) {
            return Status::Error("ASU registered region count mismatch");
        }
        std::vector<RegisteredPersistentRegion> registered;
        registered.reserve(regions.size());
        for (std::size_t index = 0; index < regions.size(); ++index) {
            registered.emplace_back(RegisteredPersistentRegion{registeredRegions[index].region,
                                                               registeredRegions[index].handle});
        }

        {
            std::lock_guard<std::mutex> lock(persistentRegionsMu_);
            persistentRegions_ = std::move(registered);
        }
        UC_INFO("ASU registered {} persistent KV cache region(s).", regions.size());
        return Status::OK();
    }

public:
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        return Submit(std::move(task), &UC::ASU::AsuClient::BatchLoadAsync);
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        auto status = WaitPrerequisiteEvent(task.prerequisiteHandle);
        if (status.Failure()) {
            UC_ERROR("ASU wait prerequisite event failed: status={}.", status);
            return status;
        }
        return Submit(std::move(task), &UC::ASU::AsuClient::BatchStoreAsync);
    }

    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        return client_->Check(static_cast<UC::ASU::TaskId>(taskId));
    }

    Status Wait(Detail::TaskHandle taskId) override
    {
        UC::ASU::TaskResult result;
        auto status =
            client_->Wait(static_cast<UC::ASU::TaskId>(taskId), config_.waitTimeoutMs, result);
        if (!status.ok()) {
            LogAsuStatus("wait task", status);
            return ConvertStatus(status);
        }
        LogAsuStatus("task result wait", result.status);
        return ConvertStatus(result.status);
    }

private:
    using SubmitFunc = AsuStatus (UC::ASU::AsuClient::*)(const std::vector<UC::ASU::KVBuffer>&,
                                                         UC::ASU::TaskId&);

    Config ParseConfig(const Detail::Dictionary& inConfig)
    {
        Config config;
        inConfig.Get("asu_mode", config.mode);
        inConfig.Get("role", config.role);
        inConfig.Get("asu_config_path", config.configPath);
        inConfig.Get("asu_client_id", config.clientId);
        inConfig.Get("unique_id", config.uniqueId);
        inConfig.Get("asu_view_service_addrs", config.viewServiceAddrs);
        inConfig.GetNumbers("asu_ids", config.asuIds);
        inConfig.Get("asu_ips", config.asuIps);
        inConfig.GetNumbers("asu_ports", config.asuPorts);
        inConfig.Get("asu_local_ip", config.localIp);
        inConfig.Get("asu_name_prefix", config.asuNamePrefix);
        inConfig.GetNumbers("kv_ns_ids", config.kvNsIds);
        inConfig.GetNumber("asu_wait_timeout_ms", config.waitTimeoutMs);
        inConfig.GetNumber("asu_query_timeout_ms", config.queryTimeoutMs);
        inConfig.GetNumber("asu_max_error_count", config.maxErrorCount);
        inConfig.GetNumber("asu_client_max_inflight_tasks", config.clientMaxInflightTasks);
        inConfig.GetNumber("asu_transport_max_inflight_tasks", config.transportMaxInflightTasks);
        inConfig.GetNumber("asu_completion_poll_spin_limit", config.completionPollSpinLimit);
        inConfig.GetNumber("asu_max_inflight_bytes", config.maxInflightBytes);
        inConfig.GetNumbers("gpu_kv_buffer_addrs", config.gpuKvBufferAddrs);
        inConfig.GetNumbers("gpu_kv_buffer_sizes", config.gpuKvBufferSizes);
        inConfig.GetNumber("shard_size", config.shardSize);
        inConfig.GetNumber("block_size", config.blockSize);
        inConfig.GetNumber("device_id", config.deviceId);
        inConfig.Get("tensor_layout", config.tensorLayout);
        std::string providerBackend;
        if (TryGetStringLike(inConfig, "asu_trans_provider_backend", providerBackend)) {
            config.transProviderType = ParseTransProviderBackend(providerBackend);
        }
        inConfig.Get("asu_fake_backend_path", config.fakeBackendPath);
        inConfig.GetNumber("asu_fake_backend_latency_ms", config.fakeBackendLatencyMs);
        inConfig.GetNumber("asu_fake_backend_worker_threads", config.fakeBackendWorkerThreads);
        inConfig.GetNumber("asu_shared_provider", config.sharedProviderMode);
        inConfig.Get("asu_sc", config.sc);
        ReadClientAttr(inConfig, "asu_router_type", "hash_table.type", config);
        ReadClientAttr(inConfig, "asu_ring_hash_virtual_node_count", "ring_hash.virtual_node_count",
                       config);
        ReadClientAttr(inConfig, "asu_maglev_table_size", "maglev.table_size", config);
        ReadClientAttr(inConfig, "asu_contiguous_block_affinity_block_count",
                       "contiguous_block_affinity.block_count", config);
        ReadClientAttr(inConfig, "asu_contiguous_block_affinity_full_spread_type",
                       "contiguous_block_affinity.full_spread_type", config);
        ReadClientAttr(inConfig, "asu_contiguous_block_affinity_dynamic_adjust_enabled",
                       "contiguous_block_affinity.dynamic_adjust_enabled", config);
        ReadClientAttr(inConfig, "asu_batch_topk_affinity_top_k", "batch_topk_affinity.top_k",
                       config);
        ReadClientAttr(inConfig, "asu_batch_topk_affinity_dynamic_adjust_enabled",
                       "batch_topk_affinity.dynamic_adjust_enabled", config);

        std::size_t tensorSize = 0;
        inConfig.GetNumber("tensor_size", tensorSize);
        if (tensorSize != 0) {
            if (config.shardSize != 0) {
                config.tensorSizes.assign(config.shardSize / tensorSize, tensorSize);
            }
        } else {
            inConfig.GetNumbers("tensor_size_list", config.tensorSizes);
        }
        return config;
    }

    void NormalizeAsuShardConfig(Config& config)
    {
        if (config.role == "scheduler" || config.tensorSizes.empty() || config.shardSize == 0 ||
            config.blockSize == 0) {
            return;
        }
        if (config.blockSize % config.shardSize != 0) { return; }

        const auto shardsPerBlock = config.blockSize / config.shardSize;
        std::size_t alignedShardSize = 0;
        for (auto& tensorSize : config.tensorSizes) {
            tensorSize = AlignUp(tensorSize, UC::ASU::kAsuAlignmentBytes);
            alignedShardSize += tensorSize;
        }
        config.shardSize = alignedShardSize;
        config.blockSize = alignedShardSize * shardsPerBlock;
    }

    Status CheckConfig(const Config& config)
    {
        if (config.gpuKvBufferAddrs.size() != config.gpuKvBufferSizes.size()) {
            return Status::InvalidParam("GPU KV cache address/size list lengths differ");
        }
        for (std::size_t index = 0; index < config.gpuKvBufferAddrs.size(); ++index) {
            if (config.gpuKvBufferAddrs[index] == 0 || config.gpuKvBufferSizes[index] == 0 ||
                config.gpuKvBufferAddrs[index] >
                    static_cast<std::uintptr_t>(std::numeric_limits<ssize_t>::max()) ||
                config.gpuKvBufferSizes[index] >
                    static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
                return Status::InvalidParam("invalid GPU KV cache range at index({})", index);
            }
        }
        if (config.sharedProviderMode != 0 && config.sharedProviderMode != 1) {
            return Status::InvalidParam("asu_shared_provider must be 0 or 1");
        }
        if (config.mode != "client" && config.mode != "transport") {
            return Status::InvalidParam("invalid asu_mode({})", config.mode);
        }
        if (config.configPath.empty() && config.asuIds.empty()) {
            return Status::InvalidParam("invalid asu_ids");
        }
        if (std::any_of(config.asuIds.begin(), config.asuIds.end(),
                        [](ssize_t asuId) { return asuId < 0; })) {
            return Status::InvalidParam("asu_ids must not contain negative values");
        }
        auto sortedAsuIds = config.asuIds;
        std::sort(sortedAsuIds.begin(), sortedAsuIds.end());
        if (std::adjacent_find(sortedAsuIds.begin(), sortedAsuIds.end()) != sortedAsuIds.end()) {
            return Status::InvalidParam("asu_ids must not contain duplicate values");
        }
        if (config.mode == "transport") {
            if (config.configPath.empty() && config.asuIds.size() != 1) {
                return Status::InvalidParam("transport mode requires exactly one asu_id");
            }
            if (!config.asuIps.empty() && config.asuIps.size() != 1) {
                return Status::InvalidParam("transport mode requires at most one asu_ip");
            }
        }
        if (!config.asuIps.empty() && config.asuIps.size() != config.asuIds.size()) {
            return Status::InvalidParam("asu_ips size must match asu_ids size");
        }
        if (!config.asuIps.empty() && config.asuPorts.size() != config.asuIps.size()) {
            return Status::InvalidParam("asu_ports size must match asu_ips size");
        }
        if (std::any_of(config.asuPorts.begin(), config.asuPorts.end(), [](ssize_t port) {
                return port <= 0 ||
                       static_cast<std::uint64_t>(port) > std::numeric_limits<std::uint16_t>::max();
            })) {
            return Status::InvalidParam("asu_ports values must be in range [1, 65535]");
        }
        if (config.configPath.empty()) {
            const auto expectedKvNsCount = config.uniqueId.find("_fawa_") == std::string::npos
                                               ? std::size_t{1}
                                               : std::size_t{2};
            if (config.kvNsIds.size() != expectedKvNsCount) {
                return Status::InvalidParam("kv_ns_ids must contain exactly {} value(s)",
                                            expectedKvNsCount);
            }
        }
        if (config.transProviderType == UC::ASU::TransProviderType::UNSUPPORTED) {
            return Status::Unsupported();
        }
        if (config.configPath.empty() &&
            config.transProviderType == UC::ASU::TransProviderType::AIV && config.deviceId < 0) {
            return Status::InvalidParam(
                "device_id is required when asu_trans_provider_backend is aiv");
        }
        if (config.transProviderType == UC::ASU::TransProviderType::FAKE &&
            !config.configPath.empty()) {
            return Status::InvalidParam(
                "asu_trans_provider_backend=fake does not support asu_config_path");
        }
        if (!config.tensorLayout.empty() && config.tensorLayout != "mla" &&
            config.tensorLayout != "gqa" && config.tensorLayout != "hma") {
            return Status::InvalidParam("invalid tensor_layout({})", config.tensorLayout);
        }
        if (!config.role.empty() && config.role != "scheduler" && config.role != "worker") {
            return Status::InvalidParam("invalid role({})", config.role);
        }
        if (config.maxErrorCount == 0 ||
            config.maxErrorCount > std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidParam("asu_max_error_count must be in uint32 range and nonzero");
        }
        if (config.waitTimeoutMs == 0) {
            return Status::InvalidParam("asu_wait_timeout_ms must be greater than zero");
        }
        if (config.queryTimeoutMs == 0) {
            return Status::InvalidParam("asu_query_timeout_ms must be greater than zero");
        }
        if (config.transProviderType == UC::ASU::TransProviderType::FAKE &&
            config.fakeBackendWorkerThreads == 0) {
            return Status::InvalidParam(
                "asu_fake_backend_worker_threads must be greater than zero");
        }
        if (config.clientMaxInflightTasks > std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidParam("asu_client_max_inflight_tasks exceeds uint32 range");
        }
        if (config.transportMaxInflightTasks > std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidParam("asu_transport_max_inflight_tasks exceeds uint32 range");
        }
        if (config.completionPollSpinLimit == 0 ||
            config.completionPollSpinLimit > std::numeric_limits<std::size_t>::max()) {
            return Status::InvalidParam(
                "asu_completion_poll_spin_limit must be a positive size_t value");
        }
        // Scheduler config check done
        if (config.role == "scheduler") { return Status::OK(); }

        if (config.tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (config.tensorLayout == "gqa" &&
            (config.tensorSizes.size() < 2 || config.tensorSizes.size() % 2 != 0)) {
            return Status::InvalidParam("invalid tensor size count({})", config.tensorSizes.size());
        }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0) { return Status::InvalidParam("invalid block size"); }
        if (config.blockSize > std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidParam("block size exceeds uint32 offset range");
        }
        const auto tensorSum =
            std::accumulate(config.tensorSizes.begin(), config.tensorSizes.end(), std::size_t{0});
        if (tensorSum == 0 || tensorSum > config.shardSize) {
            return Status::InvalidParam("invalid shard size({})", config.shardSize);
        }
        if (config.blockSize % config.shardSize != 0) {
            return Status::InvalidParam("invalid block size({})", config.blockSize);
        }
        const auto shardsPerBlock = config.blockSize / config.shardSize;
        if (shardsPerBlock > 1 && config.tensorLayout == "gqa" && config.tensorSizes.size() != 2) {
            return Status::InvalidParam("invalid layerwise gqa tensor size count({})",
                                        config.tensorSizes.size());
        }
        return Status::OK();
    }

    std::unique_ptr<UC::ASU::AsuClient> CreateClient(const Config& config)
    {
#ifdef ASU_BUILD_TESTS
        if (clientFactory_) { return clientFactory_(config); }
#endif
        return UC::ASU::CreateAsuClient();
    }

    std::size_t ShardsPerBlock() const { return config_.blockSize / config_.shardSize; }

    std::vector<std::size_t> BuildMlaTensorOffsets(std::size_t shardIndex) const
    {
        std::vector<std::size_t> offsets(config_.tensorSizes.size());
        auto offset = shardIndex * config_.shardSize;
        for (std::size_t index = 0; index < config_.tensorSizes.size(); ++index) {
            offsets[index] = offset;
            offset += config_.tensorSizes[index];
        }
        return offsets;
    }

    std::vector<std::size_t> BuildGqaTensorOffsets(std::size_t shardIndex) const
    {
        std::vector<std::size_t> offsets(config_.tensorSizes.size());
        const auto tensorCount = config_.tensorSizes.size();
        if (ShardsPerBlock() > 1) {  // layerwise case
            const auto numLayers = ShardsPerBlock();
            offsets[0] = shardIndex * config_.tensorSizes[0];
            offsets[1] = numLayers * config_.tensorSizes[0] + shardIndex * config_.tensorSizes[1];
            return offsets;
        }

        std::size_t keyBase = 0;  // non-layerwise case
        std::size_t keyPrefix = 0;
        std::size_t valuePrefix = 0;
        for (std::size_t index = 0; index < tensorCount; ++index) {
            if (index % 2 == 0) { keyBase += config_.tensorSizes[index]; }
        }
        for (std::size_t index = 0; index < tensorCount; ++index) {
            if (index % 2 == 0) {
                offsets[index] = keyPrefix;
                keyPrefix += config_.tensorSizes[index];
            } else {
                offsets[index] = keyBase + valuePrefix;
                valuePrefix += config_.tensorSizes[index];
            }
        }
        return offsets;
    }

    std::vector<std::size_t> BuildHmaTensorOffsets(std::size_t shardIndex) const
    {
        std::vector<std::size_t> offsets(config_.tensorSizes.size());
        auto offset = shardIndex * config_.shardSize;
        for (std::size_t index = 0; index < config_.tensorSizes.size(); ++index) {
            offsets[index] = offset;
            offset += config_.tensorSizes[index];
        }
        return offsets;
    }

    std::vector<std::size_t> BuildTensorOffsets(std::size_t shardIndex) const
    {
        switch (tensorLayout_) {
            case TensorLayout::MLA: return BuildMlaTensorOffsets(shardIndex);
            case TensorLayout::GQA: return BuildGqaTensorOffsets(shardIndex);
            case TensorLayout::HMA: return BuildHmaTensorOffsets(shardIndex);
        }
        throw std::logic_error("unhandled ASU tensor layout");
    }

    AsuStatus Query(const std::vector<UC::ASU::CacheKey>& keys, UC::ASU::QueryResult& result) const
    {
        UC::ASU::TaskId taskId = UC::ASU::kInvalidTaskId;
        auto status = client_->QueryAsync(keys, taskId);
        if (!status.ok()) { return status; }

        UC::ASU::TaskResult taskResult;
        status = client_->Wait(taskId, config_.queryTimeoutMs, taskResult);
        if (taskResult.queryResult.has_value()) {
            result = std::move(*taskResult.queryResult);
        } else if (status.ok()) {
            return AsuStatus::Error(UC::ASU::StatusCode::INTERNAL_ERROR,
                                    "client query result is missing");
        }
        return status;
    }

    Expected<std::vector<uint8_t>> QueryBlocks(const Detail::BlockId* blocks, std::size_t num) const
    {
        std::vector<uint8_t> result(num, false);
        if (num == 0) { return result; }

        auto keys = BuildBlockKeys(blocks, num);
        UC::ASU::QueryResult queryResult;
        auto status = Query(keys, queryResult);
        if (status.code == AsuStatusCode::TIMEOUT) { return result; }
        if (!status.ok()) {
            LogAsuStatus("query blocks", status);
            return ConvertStatus(status);
        }
        if (queryResult.exists.size() != keys.size()) {
            return Status::Error("ASU query result size mismatch");
        }

        for (std::size_t i = 0; i < num; ++i) { result[i] = queryResult.exists[i] != 0; }
        return result;
    }

    std::vector<UC::ASU::CacheKey> BuildBlockKeys(const Detail::BlockId* blocks,
                                                  std::size_t num) const
    {
        std::vector<UC::ASU::CacheKey> keys;
        keys.reserve(num);
        for (std::size_t blockIndex = 0; blockIndex < num; ++blockIndex) {
            keys.emplace_back(MakeAsuKey(blocks[blockIndex]));
        }
        return keys;
    }

    Expected<Detail::TaskHandle> Submit(Detail::TaskDesc task, SubmitFunc submit)
    {
        auto entries = BuildKvBuffers(task);
        if (!entries) { return entries.Error(); }

        UC::ASU::TaskId taskId = UC::ASU::kInvalidTaskId;
        auto status = ((*client_).*submit)(entries.Value(), taskId);
        if (!status.ok()) {
            LogAsuStatus("submit task", status);
            return ConvertStatus(status);
        }
        return static_cast<Detail::TaskHandle>(taskId);
    }

    Expected<std::vector<UC::ASU::KVBuffer>> BuildKvBuffers(const Detail::TaskDesc& task) const
    {
        std::vector<UC::ASU::KVBuffer> entries;
        entries.reserve(task.size() * config_.tensorSizes.size());

        for (const auto& shard : task) {
            if (shard.index >= ShardsPerBlock()) {
                return Status::InvalidParam("invalid shard index({})", shard.index);
            }
            if (shard.addrs.size() != config_.tensorSizes.size()) {
                return Status::InvalidParam("invalid tensor addr count({})", shard.addrs.size());
            }
            const auto tensorOffsets = BuildTensorOffsets(shard.index);
            for (std::size_t tensorIndex = 0; tensorIndex < shard.addrs.size(); ++tensorIndex) {
                UC::ASU::KVBuffer entry;
                entry.key = MakeAsuKey(shard.owner);
                entry.buffer.region.memoryType = UC::ASU::MemoryType::DEVICE;
                entry.buffer.region.addr =
                    reinterpret_cast<std::uint64_t>(shard.addrs[tensorIndex]);
                entry.buffer.region.size = config_.tensorSizes[tensorIndex];
                entry.buffer.region.deviceId = config_.deviceId;
                entry.buffer.handle = FindPersistentHandle(entry.buffer.region);
                if (entry.buffer.handle == UC::ASU::kInvalidMRHandle) {
                    return Status::Error("ASU KV buffer is outside registered persistent regions");
                }
                entry.offset = static_cast<std::uint32_t>(tensorOffsets[tensorIndex]);
                entries.emplace_back(std::move(entry));
            }
        }
        return entries;
    }

    void ShowConfig(const Config& config) const
    {
        UC_INFO("AsuStore.");
        UC_INFO("Set AsuStore::Mode to {}.", config.mode);
        UC_INFO("Set AsuStore::Role to {}.", config.role);
        UC_INFO("Set AsuStore::ConfigPath to {}.", config.configPath);
        UC_INFO("Set AsuStore::ClientId to {}.", config.clientId);
        UC_INFO("Set AsuStore::AsuIds to {}.", config.asuIds);
        UC_INFO("Set AsuStore::AsuIps to {}.", config.asuIps);
        UC_INFO("Set AsuStore::KvNsIds to {}.", config.kvNsIds);
        UC_INFO("Set AsuStore::ShardSize to {}.", config.shardSize);
        UC_INFO("Set AsuStore::BlockSize to {}.", config.blockSize);
        UC_INFO("Set AsuStore::TensorSizes to {}.", config.tensorSizes);
        UC_INFO("Set AsuStore::TensorLayout to {}.", config.tensorLayout);
        UC_INFO("Set AsuStore::DeviceId to {}.", config.deviceId);
        UC_INFO("Set AsuStore::TransProviderBackend to {}.",
                TransProviderBackendName(config.transProviderType));
        UC_INFO("Set AsuStore::FakeBackendPath to {}.", config.fakeBackendPath);
        UC_INFO("Set AsuStore::FakeBackendWorkerThreads to {}.", config.fakeBackendWorkerThreads);
    }

    UC::ASU::MRHandle FindPersistentHandle(const UC::ASU::MemoryRegion& region) const
    {
        std::lock_guard<std::mutex> lock(persistentRegionsMu_);
        for (const auto& persistent : persistentRegions_) {
            if (region.addr < persistent.region.addr || region.size > persistent.region.size) {
                continue;
            }
            const auto offset = region.addr - persistent.region.addr;
            if (offset <= persistent.region.size - region.size) { return persistent.handle; }
        }
        return UC::ASU::kInvalidMRHandle;
    }

    struct RegisteredPersistentRegion {
        UC::ASU::MemoryRegion region;
        UC::ASU::MRHandle handle{UC::ASU::kInvalidMRHandle};
    };

    Config config_;
    TensorLayout tensorLayout_{TensorLayout::MLA};
    std::unique_ptr<UC::ASU::AsuClient> client_;
    mutable std::mutex persistentRegionsMu_;
    std::vector<RegisteredPersistentRegion> persistentRegions_;
#ifdef ASU_BUILD_TESTS
    ClientFactory clientFactory_;
#endif
};

}  // namespace UC::AsuStore

extern "C" UC::StoreV1* MakeAsuStore() { return new UC::AsuStore::AsuStore(); }
