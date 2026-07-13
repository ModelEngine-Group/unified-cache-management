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
#include <acl/acl.h>
#include <algorithm>
#include <any>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "asu_client/asu_client.h"
#include "asu_transport/asu_transport.h"
#include "logger/logger.h"
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

AsuStatus WaitPrerequisiteEvent(std::uintptr_t eventHandle)
{
    if (eventHandle == 0) { return AsuStatus::OK(); }
    auto ret = aclrtSynchronizeEvent(reinterpret_cast<aclrtEvent>(eventHandle));
    if (ret == ACL_SUCCESS) { return AsuStatus::OK(); }
    return AsuStatus::Error(AsuStatusCode::INTERNAL_ERROR,
                            "aclrtSynchronizeEvent failed: " + std::to_string(ret));
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

UC::ASU::MemoryType ParseMemoryType(const std::string& memoryType)
{
    if (memoryType == "host") { return UC::ASU::MemoryType::HOST; }
    if (memoryType == "host_pinned") { return UC::ASU::MemoryType::HOST_PINNED; }
    if (memoryType == "ascend_device") { return UC::ASU::MemoryType::ASCEND_DEVICE; }
    return UC::ASU::MemoryType::ASCEND_DEVICE;
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
    transportConfig.queryTimeoutMs = config.queryTimeoutMs;
    transportConfig.loadTimeoutMs = config.loadTimeoutMs;
    transportConfig.storeTimeoutMs = config.storeTimeoutMs;
    transportConfig.maxInflightTasks = static_cast<std::uint32_t>(config.maxInflightTasks);
    transportConfig.maxInflightBytes = config.maxInflightBytes;
    transportConfig.providerType = config.transProviderType;

    // Set for all backends including fake
    const auto kvNsIndex = config.uniqueId.find("_fawa_wa") == std::string::npos ? 0 : 1;
    transportConfig.attrs["kv_ns_id"] = std::to_string(config.kvNsIds[kvNsIndex]);

    if (!config.asuIps.empty()) {
        UC::ASU::AsuEndpoint endpoint;
        endpoint.ip = config.asuIps[index];
        endpoint.port = config.asuPort;
        endpoint.deviceId = config.deviceId;
        transportConfig.endpoints.emplace_back(std::move(endpoint));
    }
    if (config.transProviderType == UC::ASU::TransProviderType::FAKE) {
        const auto fakeDeviceId = config.deviceId >= 0 ? config.deviceId : 0;
        transportConfig.attrs.try_emplace("kernel_count", "1");
        transportConfig.attrs.try_emplace("quiet_count", "1");
        transportConfig.attrs.try_emplace("dtype", "0");
        transportConfig.attrs.try_emplace("dspec", "0");
        transportConfig.attrs.try_emplace("lr", "false");
        transportConfig.attrs["sc"] = "true";
        transportConfig.attrs["fake_backend.path"] = config.fakeBackendPath;
        transportConfig.attrs["fake_backend.latency_ms"] =
            std::to_string(config.fakeBackendLatencyMs);
        transportConfig.attrs["fake_backend.device_id"] = std::to_string(fakeDeviceId);
        if (transportConfig.endpoints.empty()) {
            UC::ASU::AsuEndpoint endpoint;
            endpoint.ip = "fake_backend";
            endpoint.port = 19001;
            endpoint.protocol = UC::ASU::Protocol::TCP;
            endpoint.deviceId = fakeDeviceId;
            transportConfig.endpoints.emplace_back(std::move(endpoint));
        }
    }
    return transportConfig;
}

class ClientBackend final : public AsuBackend {
public:
    AsuStatus Init(const Config& config) override
    {
        client_ = UC::ASU::CreateAsuClient();
        UC::ASU::AsuClientConfig asuConfig;
        asuConfig.clientId = config.clientId;
        asuConfig.viewServiceAddrs = config.viewServiceAddrs;
        asuConfig.defaultWaitTimeoutMs = config.defaultWaitTimeoutMs;
        asuConfig.attrs = config.clientAttrs;
        asuConfig.transportConfigs.reserve(config.asuIds.size());
        for (std::size_t i = 0; i < config.asuIds.size(); ++i) {
            asuConfig.transportConfigs.emplace_back(BuildTransportConfig(config, i));
        }
        return client_->Init(asuConfig);
    }

    AsuStatus Init(const std::string& configPath) override
    {
        client_ = UC::ASU::CreateAsuClient();
        return client_->Init(configPath);
    }

    AsuStatus Shutdown() override { return client_ ? client_->Shutdown() : AsuStatus::OK(); }

    AsuStatus Query(const std::vector<UC::ASU::CacheKey>& keys,
                    const UC::ASU::QueryOptions& options, UC::ASU::QueryResult& result) override
    {
        return client_->Query(keys, options, result);
    }

    AsuStatus LoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                        UC::ASU::TaskId& taskId) override
    {
        return client_->LoadAsync(entries, taskId);
    }

    AsuStatus StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                         UC::ASU::TaskId& taskId) override
    {
        return client_->StoreAsync(entries, taskId);
    }

    AsuStatus DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                          UC::ASU::TaskId& taskId) override
    {
        return client_->DeleteAsync(keys, taskId);
    }

    AsuStatus Check(UC::ASU::TaskId taskId, UC::ASU::TaskResult& result) override
    {
        return client_->Check(taskId, result);
    }

    AsuStatus Wait(UC::ASU::TaskId taskId, std::uint64_t timeoutMs,
                   UC::ASU::TaskResult& result) override
    {
        return client_->Wait(taskId, timeoutMs, result);
    }

private:
    std::unique_ptr<UC::ASU::AsuClient> client_;
};

class TransportBackend final : public AsuBackend {
public:
    AsuStatus Init(const Config& config) override
    {
        transport_ = UC::ASU::CreateAsuTransport();
        if (!transport_) {
            return AsuStatus::Error(AsuStatusCode::INTERNAL_ERROR,
                                    "ASU transport factory returned null");
        }
        return transport_->Init(BuildTransportConfig(config, 0));
    }

    AsuStatus Init(const std::string& configPath) override
    {
        transport_ = UC::ASU::CreateAsuTransport();
        if (!transport_) {
            return AsuStatus::Error(AsuStatusCode::INTERNAL_ERROR,
                                    "ASU transport factory returned null");
        }
        return transport_->Init(configPath);
    }

    AsuStatus Shutdown() override { return transport_ ? transport_->Shutdown() : AsuStatus::OK(); }

    AsuStatus Query(const std::vector<UC::ASU::CacheKey>& keys,
                    const UC::ASU::QueryOptions& options, UC::ASU::QueryResult& result) override
    {
        return transport_->Query(keys, options, result);
    }

    AsuStatus LoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                        UC::ASU::TaskId& taskId) override
    {
        return transport_->LoadAsync(entries, taskId);
    }

    AsuStatus StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                         UC::ASU::TaskId& taskId) override
    {
        return transport_->StoreAsync(entries, taskId);
    }

    AsuStatus DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                          UC::ASU::TaskId& taskId) override
    {
        return transport_->DeleteAsync(keys, taskId);
    }

    AsuStatus Check(UC::ASU::TaskId taskId, UC::ASU::TaskResult& result) override
    {
        return transport_->Check(taskId, result);
    }

    AsuStatus Wait(UC::ASU::TaskId taskId, std::uint64_t timeoutMs,
                   UC::ASU::TaskResult& result) override
    {
        return transport_->Wait(taskId, timeoutMs, result);
    }

private:
    std::unique_ptr<UC::ASU::AsuTransport> transport_;
};

class AsuStore final : public StoreV1 {
public:
#ifdef ASU_BUILD_TESTS
    using BackendFactory = std::function<std::unique_ptr<AsuBackend>(const Config&)>;

    void SetBackendFactory(BackendFactory factory) { backendFactory_ = std::move(factory); }
#endif

    ~AsuStore() override
    {
        if (backend_) {
            auto status = backend_->Shutdown();
            if (!status.ok()) { UC_ERROR("Failed to shutdown ASU backend: {}.", status.message); }
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
        backend_ = CreateBackend(config_);

        auto asuStatus = config_.configPath.empty() ? backend_->Init(config_)
                                                    : backend_->Init(config_.configPath);
        if (!asuStatus.ok()) {
            UC_ERROR("Failed to init ASU backend: {}.", asuStatus.message);
            backend_.reset();
            return ConvertStatus(asuStatus);
        }

        ShowConfig(config_);
        return Status::OK();
    }

    std::string Readme() const override { return "AsuStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        UC::ASU::QueryOptions options;
        options.mode = UC::ASU::QueryMode::PER_KEY;
        options.timeoutMs = config_.queryTimeoutMs;
        return QueryBlocks(blocks, num, options);
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return static_cast<ssize_t>(-1); }

        UC::ASU::QueryOptions options;
        options.mode = UC::ASU::QueryMode::PREFIX;
        options.timeoutMs = config_.queryTimeoutMs;

        auto keys = BuildBlockKeys(blocks, num);
        UC::ASU::QueryResult queryResult;
        auto status = backend_->Query(keys, options, queryResult);
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

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        return Submit(std::move(task), &AsuBackend::LoadAsync);
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        auto status = WaitPrerequisiteEvent(task.prerequisiteHandle);
        if (!status.ok()) {
            LogAsuStatus("wait prerequisite event", status);
            return ConvertStatus(status);
        }
        return Submit(std::move(task), &AsuBackend::StoreAsync);
    }

    Expected<bool> Check(Detail::TaskHandle taskId) override
    {
        UC::ASU::TaskResult result;
        auto status = backend_->Check(static_cast<UC::ASU::TaskId>(taskId), result);
        if (!status.ok()) {
            LogAsuStatus("check task", status);
            return ConvertStatus(status);
        }
        if (!result.status.ok() && result.status.code != AsuStatusCode::IN_PROGRESS) {
            LogAsuStatus("task result check", result.status);
        }
        return result.status.code != AsuStatusCode::IN_PROGRESS;
    }

    Status Wait(Detail::TaskHandle taskId) override
    {
        UC::ASU::TaskResult result;
        auto status = backend_->Wait(static_cast<UC::ASU::TaskId>(taskId),
                                     config_.defaultWaitTimeoutMs, result);
        if (!status.ok()) {
            LogAsuStatus("wait task", status);
            return ConvertStatus(status);
        }
        LogAsuStatus("task result wait", result.status);
        return ConvertStatus(result.status);
    }

private:
    using SubmitFunc = AsuStatus (AsuBackend::*)(const std::vector<UC::ASU::KVBuffer>&,
                                                 UC::ASU::TaskId&);

    Config ParseConfig(const Detail::Dictionary& inConfig)
    {
        Config config;
        inConfig.Get("asu_mode", config.mode);
        inConfig.Get("asu_config_path", config.configPath);
        inConfig.Get("asu_client_id", config.clientId);
        inConfig.Get("unique_id", config.uniqueId);
        inConfig.Get("asu_view_service_addrs", config.viewServiceAddrs);
        inConfig.GetNumbers("asu_ids", config.asuIds);
        inConfig.Get("asu_ips", config.asuIps);
        inConfig.Get("asu_name_prefix", config.asuNamePrefix);
        inConfig.GetNumbers("kv_ns_ids", config.kvNsIds);
        ssize_t asuPort = 0;
        inConfig.GetNumber("asu_port", asuPort);
        config.asuPort = static_cast<std::uint16_t>(std::max<ssize_t>(0, asuPort));
        inConfig.GetNumber("asu_default_wait_timeout_ms", config.defaultWaitTimeoutMs);
        inConfig.GetNumber("asu_query_timeout_ms", config.queryTimeoutMs);
        inConfig.GetNumber("asu_load_timeout_ms", config.loadTimeoutMs);
        inConfig.GetNumber("asu_store_timeout_ms", config.storeTimeoutMs);
        inConfig.GetNumber("asu_max_inflight_tasks", config.maxInflightTasks);
        inConfig.GetNumber("asu_max_inflight_bytes", config.maxInflightBytes);
        inConfig.GetNumber("shard_size", config.shardSize);
        inConfig.GetNumber("block_size", config.blockSize);
        inConfig.GetNumber("device_id", config.deviceId);
        inConfig.Get("asu_memory_type", config.memoryType);
        inConfig.Get("asu_tensor_layout", config.tensorLayout);
        std::string providerBackend;
        if (TryGetStringLike(inConfig, "asu_trans_provider_backend", providerBackend)) {
            config.transProviderType = ParseTransProviderBackend(providerBackend);
        }
        inConfig.Get("asu_fake_backend_path", config.fakeBackendPath);
        inConfig.GetNumber("asu_fake_backend_latency_ms", config.fakeBackendLatencyMs);
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
        if (config.tensorSizes.empty() || config.shardSize == 0 || config.blockSize == 0) {
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
        if (config.mode != "client" && config.mode != "transport") {
            return Status::InvalidParam("invalid asu_mode({})", config.mode);
        }
        if (config.configPath.empty() && config.asuIds.empty()) {
            return Status::InvalidParam("invalid asu_ids");
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
        if (config.uniqueId.find("_fawa_") != std::string::npos &&
            config.kvNsIds.size() != 2) {
            return Status::InvalidParam("FAWA requires exactly two kv_ns_ids");
        }
        if (config.transProviderType == UC::ASU::TransProviderType::UNSUPPORTED) {
            return Status::Unsupported();
        }
        if (config.transProviderType == UC::ASU::TransProviderType::FAKE &&
            !config.configPath.empty()) {
            return Status::InvalidParam(
                "asu_trans_provider_backend=fake does not support asu_config_path");
        }
        if (!config.tensorLayout.empty() && config.tensorLayout != "mla" &&
            config.tensorLayout != "gqa" && config.tensorLayout != "hma") {
            return Status::InvalidParam("invalid asu_tensor_layout({})", config.tensorLayout);
        }
        if (config.tensorSizes.empty()) { return Status::InvalidParam("invalid tensor size"); }
        if (config.tensorLayout == "gqa" &&
            (config.tensorSizes.size() < 2 || config.tensorSizes.size() % 2 != 0)) {
            return Status::InvalidParam("invalid tensor size count({})", config.tensorSizes.size());
        }
        if (config.shardSize == 0) { return Status::InvalidParam("invalid shard size"); }
        if (config.blockSize == 0) { return Status::InvalidParam("invalid block size"); }
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
        if (shardsPerBlock > 1 && config.tensorLayout == "mla" && config.tensorSizes.size() != 1) {
            return Status::InvalidParam("invalid layerwise mla tensor size count({})",
                                        config.tensorSizes.size());
        }
        return Status::OK();
    }

    std::unique_ptr<AsuBackend> CreateBackend(const Config& config)
    {
#ifdef ASU_BUILD_TESTS
        if (backendFactory_) { return backendFactory_(config); }
#endif
        if (config.mode == "transport") { return std::make_unique<TransportBackend>(); }
        return std::make_unique<ClientBackend>();
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

    Expected<std::vector<uint8_t>> QueryBlocks(const Detail::BlockId* blocks, std::size_t num,
                                               const UC::ASU::QueryOptions& options) const
    {
        std::vector<uint8_t> result(num, false);
        if (num == 0) { return result; }

        auto keys = BuildBlockKeys(blocks, num);
        UC::ASU::QueryResult queryResult;
        auto status = backend_->Query(keys, options, queryResult);
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
        auto status = ((*backend_).*submit)(entries.Value(), taskId);
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
        const auto memoryType = config_.memoryType.empty()
                                    ? (config_.deviceId >= 0 ? UC::ASU::MemoryType::ASCEND_DEVICE
                                                             : UC::ASU::MemoryType::HOST)
                                    : ParseMemoryType(config_.memoryType);

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
                entry.buffer.region.memoryType = memoryType;
                entry.buffer.region.addr =
                    reinterpret_cast<std::uint64_t>(shard.addrs[tensorIndex]);
                entry.buffer.region.size = config_.tensorSizes[tensorIndex];
                entry.buffer.region.deviceId = config_.deviceId;
                entry.buffer.handle = UC::ASU::kInvalidMRHandle;
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
    }

    Config config_;
    TensorLayout tensorLayout_{TensorLayout::MLA};
    std::unique_ptr<AsuBackend> backend_;
#ifdef ASU_BUILD_TESTS
    BackendFactory backendFactory_;
#endif
};

}  // namespace UC::AsuStore

extern "C" UC::StoreV1* MakeAsuStore() { return new UC::AsuStore::AsuStore(); }
