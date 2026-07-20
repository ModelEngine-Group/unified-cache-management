/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include "client_service.h"
#include "global_config.h"
#include "logger/logger.h"
#include "trans_manager.h"
#include "transfer_engine.h"
#include "type/random_block_id.h"
#include "ucmstore_v1.h"

namespace UC::MooncakeStore {

namespace {

constexpr char kHexTable[] = "0123456789abcdef";

std::string BlockIdToKey(const Detail::BlockId& block)
{
    std::string out;
    out.resize(block.size() * 2);
    for (size_t i = 0; i < block.size(); ++i) {
        auto b = static_cast<uint8_t>(block[i]);
        out[i * 2] = kHexTable[b >> 4];
        out[i * 2 + 1] = kHexTable[b & 0x0F];
    }
    return out;
}

std::string HealthKey() { return "__ucm_health_" + BlockIdToKey(Detail::RandomBlockId()); }

Status MooncakeError(mooncake::ErrorCode error) { return Status::Error(mooncake::toString(error)); }

}  // namespace

class MooncakeStore : public StoreV1 {
    TransManager transMgr_;
    Config config_;
    bool transEnable_{false};

    std::shared_ptr<mooncake::Client> rpcClient_;

    std::mutex registerMtx_;
    std::unordered_map<void*, size_t> registered_;

    std::atomic<bool> closed_{false};

public:
    ~MooncakeStore() override { Close(); }

    void Close()
    {
        if (closed_.exchange(true, std::memory_order_acq_rel)) { return; }

        transMgr_.Close();

        auto client = transMgr_.GetRealClient();
        if (client) {
            std::lock_guard<std::mutex> lk(registerMtx_);
            for (auto& [addr, sz] : registered_) {
                auto rc = client->unregister_buffer(addr);
                if (rc != 0) {
                    UC_WARN("unregister_buffer failed: addr={}, size={}, rc={}", addr, sz, rc);
                }
            }
            registered_.clear();
        }
    }

    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = ParseConfig(inConfig);
        auto s = CheckConfig(config);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Config check failed: {}.", s);
            return s;
        }
        config_ = config;
        transEnable_ = config.deviceId >= 0;

        if (transEnable_) {
            s = transMgr_.Setup(config);
            if (s.Failure()) [[unlikely]] { return s; }
            s = RegisterKvBuffers(config);
            if (s.Failure()) [[unlikely]] { return s; }
        } else {
            s = SetupRpcClient(config);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        ShowConfig(config);
        return Status::OK();
    }

    std::string Readme() const override { return "MooncakeStore"; }

    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return std::vector<uint8_t>{}; }

        auto res = LookupOnPrefix(blocks, num);
        if (!res) [[unlikely]] { return res.Error(); }

        std::vector<uint8_t> results(num, 0);
        const auto index = res.Value();
        for (ssize_t i = 0; i <= index; ++i) { results[i] = 1; }
        return results;
    }

    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        if (num == 0) { return static_cast<ssize_t>(-1); }

        std::vector<std::string> keys;
        keys.reserve(num);
        for (size_t i = 0; i < num; ++i) { keys.push_back(BlockIdToKey(blocks[i]) + "_0"); }

        auto exists = RpcBatchIsExist(keys);

        ssize_t firstMiss = -1;
        for (size_t i = 0; i < num; ++i) {
            if (exists[i] != 1) {
                firstMiss = static_cast<ssize_t>(i);
                break;
            }
        }

        if (firstMiss == -1) { return static_cast<ssize_t>(num) - 1; }

        if (config_.storeBackend) {
            auto backendRes =
                config_.storeBackend->LookupOnPrefix(blocks + firstMiss, num - firstMiss);
            if (backendRes) {
                ssize_t backendHit = backendRes.Value();
                if (backendHit >= 0) { return firstMiss + backendHit; }
            }
        }

        return firstMiss - 1;
    }

    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        if (config_.storeBackend) { config_.storeBackend->Prefetch(blocks, num); }
    }

    Status CheckHealth() override
    {
        const auto key = HealthKey();
        constexpr std::array<char, 8> expected{'U', 'C', 'M', 'H', 'E', 'A', 'L', 'T'};
        std::array<char, expected.size()> actual{};

        if (transEnable_) {
            auto client = transMgr_.GetRealClient();
            if (!client) { return Status::Error("mooncake client is not available"); }
            std::vector<std::span<const char>> parts{
                {expected.data(), expected.size()}
            };
            auto rc = client->put_parts(key, std::move(parts));
            if (rc != 0) { return Status::Error("mooncake health put failed"); }
            auto buffer = client->get_buffer(key);
            auto cleanup = client->remove(key);
            if (!buffer || buffer->size() != expected.size() ||
                std::memcmp(buffer->ptr(), expected.data(), expected.size()) != 0) {
                return Status::Error("mooncake health data mismatch");
            }
            return cleanup == 0 ? Status::OK() : Status::Error("mooncake health remove failed");
        }

        std::vector<mooncake::Slice> source{
            {const_cast<char*>(expected.data()), expected.size()}
        };
        auto put = rpcClient_->Put(key, source, mooncake::ReplicateConfig{});
        if (!put.has_value()) { return MooncakeError(put.error()); }
        std::vector<mooncake::Slice> target{
            {actual.data(), actual.size()}
        };
        auto get = rpcClient_->Get(key, target);
        auto cleanup = rpcClient_->Remove(key);
        if (!get.has_value()) { return MooncakeError(get.error()); }
        if (actual != expected) { return Status::Error("mooncake health data mismatch"); }
        return cleanup.has_value() ? Status::OK() : MooncakeError(cleanup.error());
    }

    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enabled (scheduler mode)"); }
        TransTask transTask;
        transTask.type = TaskType::LOAD;
        transTask.brief = task.brief;
        BuildShards(task, transTask);
        return transMgr_.Submit(std::move(transTask));
    }

    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        if (!transEnable_) { return Status::Error("transfer is not enabled (scheduler mode)"); }
        TransTask transTask;
        transTask.type = TaskType::DUMP;
        transTask.brief = task.brief;
        transTask.prerequisiteHandle = task.prerequisiteHandle;
        BuildShards(task, transTask);
        return transMgr_.Submit(std::move(transTask));
    }

    Expected<bool> Check(Detail::TaskHandle taskId) override { return transMgr_.Check(taskId); }

    Status Wait(Detail::TaskHandle taskId) override { return transMgr_.Wait(taskId); }

private:
    Status RegisterKvBuffers(const Config& config)
    {
        auto client = transMgr_.GetRealClient();
        if (!client) { return Status::OK(); }
        if (config.gpuKvBufferAddrs.empty()) { return Status::OK(); }

        std::lock_guard<std::mutex> lk(registerMtx_);
        for (size_t i = 0; i < config.gpuKvBufferAddrs.size(); ++i) {
            void* addr = reinterpret_cast<void*>(config.gpuKvBufferAddrs[i]);
            size_t size = config.gpuKvBufferSizes[i];
            if (registered_.count(addr)) { continue; }

            int rc = client->register_buffer(addr, size);
            if (rc != 0) {
                UC_ERROR("register_buffer failed: addr={}, size={}, rc={}", addr, size, rc);
                return Status::Error("register_buffer failed");
            }
            registered_[addr] = size;
            UC_DEBUG("Registered buffer addr={}, size={}", addr, size);
        }
        return Status::OK();
    }

    Status SetupRpcClient(const Config& config)
    {
        auto dummyTE = std::make_shared<mooncake::TransferEngine>();
        auto clientOpt =
            mooncake::Client::Create(config.localHostname, config.metadataServer, config.protocol,
                                     std::nullopt, config.masterServerAddress, dummyTE);
        if (!clientOpt.has_value()) {
            UC_ERROR("RPC Client::Create failed");
            return Status::Error("RPC Client::Create failed");
        }
        rpcClient_ = clientOpt.value();
        UC_DEBUG("RPC Client setup ok (lookup only)");
        return Status::OK();
    }

    std::vector<int> RpcBatchIsExist(const std::vector<std::string>& keys)
    {
        if (keys.empty()) { return std::vector<int>(keys.size(), -1); }

        if (!rpcClient_) { return std::vector<int>(keys.size(), -1); }

        auto results = rpcClient_->BatchIsExist(keys);
        std::vector<int> out;
        out.reserve(results.size());
        for (auto& r : results) { out.push_back(!r.has_value() ? -1 : (r.value() ? 1 : 0)); }
        return out;
    }

    void BuildShards(const Detail::TaskDesc& desc, TransTask& out)
    {
        out.shards.reserve(desc.size());
        for (const auto& shard : desc) {
            std::string key = BlockIdToKey(shard.owner) + "_" + std::to_string(shard.index);

            if (shard.addrs.size() > config_.tensorSizeList.size()) {
                UC_WARN(
                    "BuildShards: key={} has {} addrs but tensorSizeList has only {}, truncating",
                    key, shard.addrs.size(), config_.tensorSizeList.size());
            }

            size_t count = std::min(shard.addrs.size(), config_.tensorSizeList.size());
            std::vector<void*> addrs(shard.addrs.begin(), shard.addrs.begin() + count);
            std::vector<size_t> sizes(config_.tensorSizeList.begin(),
                                      config_.tensorSizeList.begin() + count);

            out.shards.push_back(TransShard{std::move(key), shard.owner, shard.index,
                                            std::move(addrs), std::move(sizes)});
        }
    }

    Config ParseConfig(const Detail::Dictionary& inConfig)
    {
        Config config;
        inConfig.Get("local_hostname", config.localHostname);

        auto colonPos = config.localHostname.rfind(':');
        if (colonPos != std::string::npos) {
            UC_WARN("local_hostname contains port '{}', stripping — SDK will auto-assign port",
                    config.localHostname);
            config.localHostname = config.localHostname.substr(0, colonPos);
        }

        inConfig.Get("metadata_server", config.metadataServer);
        inConfig.GetNumber("global_segment_size", config.globalSegmentSize);
        inConfig.GetNumber("local_buffer_size", config.localBufferSize);
        size_t globalSegmentSizeGb = 0;
        size_t localBufferSizeGb = 0;
        inConfig.GetNumber("global_segment_size_gb", globalSegmentSizeGb);
        inConfig.GetNumber("local_buffer_size_gb", localBufferSizeGb);
        if (globalSegmentSizeGb != 0) { config.globalSegmentSize = globalSegmentSizeGb << 30; }
        if (localBufferSizeGb != 0) { config.localBufferSize = localBufferSizeGb << 30; }
        inConfig.Get("protocol", config.protocol);
        inConfig.Get("device_name", config.deviceName);
        inConfig.Get("master_server_address", config.masterServerAddress);
        inConfig.GetNumber("replica_num", config.replicaNum);
        inConfig.Get("with_soft_pin", config.withSoftPin);
        inConfig.GetNumber("device_id", config.deviceId);
        inConfig.GetNumbers("tensor_size_list", config.tensorSizeList);
        inConfig.GetNumber("dump_queue_depth", config.dumpQueueDepth);
        inConfig.GetNumber("load_queue_depth", config.loadQueueDepth);
        inConfig.GetNumber("timeout_ms", config.timeoutMs);
        inConfig.GetNumber("stream_number", config.streamNumber);
        inConfig.Get("cpu_affinity_cores", config.cpuAffinityCores);
        inConfig.Get("io_direct", config.ioDirect);
        inConfig.GetNumber("local_rank_size", config.localRankSize);
        inConfig.Get("unique_id", config.uniqueId);
        inConfig.Get("store_backend", config.storeBackend);
        inConfig.GetNumbers("gpu_kv_buffer_addrs", config.gpuKvBufferAddrs);
        inConfig.GetNumbers("gpu_kv_buffer_sizes", config.gpuKvBufferSizes);
        DeriveShareBufferNumber(inConfig, config);
        DeriveHostBufPoolSize(inConfig, config);
        return config;
    }

    void DeriveShareBufferNumber(const Detail::Dictionary& inConfig, Config& config)
    {
        size_t shareBufferCapacityGb = 0;
        inConfig.GetNumber("share_buffer_capacity_gb", shareBufferCapacityGb);
        if (shareBufferCapacityGb != 0) {
            config.shareBufferCapacity = shareBufferCapacityGb << 30;
        }

        const uint64_t shareUnit = std::accumulate(config.tensorSizeList.begin(),
                                                   config.tensorSizeList.end(), uint64_t{0});
        if (shareUnit > 0) {
            config.shareBufferNumber = static_cast<size_t>(config.shareBufferCapacity / shareUnit);
        }
    }

    void DeriveHostBufPoolSize(const Detail::Dictionary& inConfig, Config& config)
    {
        constexpr uint32_t kHostBufPerStream = 256;
        constexpr uint64_t kHostBufMaxPinnedBytes = 8ULL << 30;
        constexpr uint32_t kHostBufMinCount = 64;

        const uint64_t hostUnit = std::accumulate(config.tensorSizeList.begin(),
                                                  config.tensorSizeList.end(), uint64_t{0});
        if (hostUnit > 0) {
            uint64_t derived = static_cast<uint64_t>(config.streamNumber) * kHostBufPerStream;
            uint64_t capByMem = kHostBufMaxPinnedBytes / hostUnit;
            if (capByMem == 0) { capByMem = 1; }
            derived = std::min<uint64_t>(derived, capByMem);
            if (derived < kHostBufMinCount) {
                derived = std::min<uint64_t>(kHostBufMinCount, capByMem);
            }
            config.hostBufPoolSize = static_cast<uint32_t>(derived);
        }

        uint32_t explicitPool = 0;
        inConfig.GetNumber("host_buf_pool_size", explicitPool);
        if (explicitPool != 0) { config.hostBufPoolSize = explicitPool; }
    }

    Status CheckConfig(const Config& config)
    {
        if (config.localHostname.empty()) {
            return Status::InvalidParam("local_hostname is required");
        }
        if (config.protocol.empty()) { return Status::InvalidParam("protocol is required"); }
        if (config.replicaNum == 0) { return Status::InvalidParam("replica_num must be > 0"); }
        if (config.streamNumber < 1 || config.streamNumber > 32) {
            return Status::InvalidParam("invalid stream_number({})", config.streamNumber);
        }
        return Status::OK();
    }

    void ShowConfig(const Config& config)
    {
        constexpr const char* ns = "MooncakeStore";
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("{}-{}({}).", ns, UCM_COMMIT_ID, buildType);
        UC_INFO("{}::LocalHostname = {}", ns, config.localHostname);
        UC_INFO("{}::MetadataServer = {}", ns, config.metadataServer);
        UC_INFO("{}::MasterServerAddress = {}", ns, config.masterServerAddress);
        UC_INFO("{}::Protocol = {}", ns, config.protocol);
        UC_INFO("{}::DeviceName = {}", ns, config.deviceName);
        UC_INFO("{}::GlobalSegmentSize = {}", ns, config.globalSegmentSize);
        UC_INFO("{}::ReplicaNum = {}", ns, config.replicaNum);
        UC_INFO("{}::WithSoftPin = {}", ns, config.withSoftPin);
        UC_INFO("{}::DeviceId = {}", ns, config.deviceId);
        UC_INFO("{}::DumpQueueDepth = {}", ns, config.dumpQueueDepth);
        UC_INFO("{}::LoadQueueDepth = {}", ns, config.loadQueueDepth);
        UC_INFO("{}::HostBufPoolSize = {}", ns, config.hostBufPoolSize);
        UC_INFO("{}::TimeoutMs = {}", ns, config.timeoutMs);
        UC_INFO("{}::StreamNumber = {}", ns, config.streamNumber);
        UC_INFO("{}::CpuAffinityCores = {}", ns, config.cpuAffinityCores);
        UC_INFO("{}::IoDirect = {}", ns, config.ioDirect);
        UC_INFO("{}::LocalRankSize = {}", ns, config.localRankSize);
        UC_INFO("{}::UniqueId = {}", ns, config.uniqueId);
        UC_INFO("{}::ShareBufferCapacity = {}GB", ns, config.shareBufferCapacity >> 30);
        UC_INFO("{}::ShareBufferNumber = {}", ns, config.shareBufferNumber);
        UC_INFO("{}::LocalBufferSize = {}", ns, config.localBufferSize);
        UC_INFO("{}::StoreBackend = {}", ns, config.storeBackend ? "yes" : "none");
        UC_INFO("{}::TransEnable = {}", ns, transEnable_);
    }
};

}  // namespace UC::MooncakeStore

extern "C" UC::StoreV1* MakeMooncakeStore() { return new UC::MooncakeStore::MooncakeStore(); }
extern "C" void DestroyMooncakeStore(UC::StoreV1* p) { delete p; }
