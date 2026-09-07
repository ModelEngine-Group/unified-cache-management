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
#include "asu/cc/asu_store.cc"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "detail/types_helper.h"

namespace {

struct CacheKeyHasher {
    std::size_t operator()(const UC::ASU::CacheKey& key) const
    {
        return std::hash<std::string_view>{}(UC::ASU::CacheKeyView(key));
    }
};

UC::ASU::MRHandle MakeTestMrHandle(std::uintptr_t value)
{
    return reinterpret_cast<UC::ASU::MRHandle>(value);
}

struct FakeAsuClientState {
    std::vector<UC::AsuStore::Config> initConfigs;
    std::vector<UC::ASU::KVBuffer> lastLoadEntries;
    std::vector<UC::ASU::KVBuffer> lastStoreEntries;
    std::vector<UC::ASU::MemoryRegion> registeredRegions;
    std::size_t registrationCalls{0};
    std::size_t shutdownCalls{0};
    bool failRegistration{false};
    bool omitRegisteredHandle{false};
};

class FakeAsuClient final : public UC::ASU::AsuClient {
public:
    explicit FakeAsuClient(std::shared_ptr<FakeAsuClientState> state) : state_(std::move(state)) {}

    UC::ASU::Status Init(const UC::ASU::AsuClientConfig& config) override
    {
        (void)config;
        initialized_ = true;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Init(const std::string& configPath) override
    {
        (void)configPath;
        initialized_ = true;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Shutdown() override
    {
        ++state_->shutdownCalls;
        initialized_ = false;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status QueryAsync(const std::vector<UC::ASU::CacheKey>& keys,
                               UC::ASU::TaskId& taskId) override
    {
        if (!initialized_) { return NotInitialized(); }

        UC::ASU::TaskResult taskResult;
        taskResult.status = UC::ASU::Status::OK();
        taskResult.entryStatus.assign(keys.size(), UC::ASU::Status::OK());
        UC::ASU::QueryResult result;
        result.exists.clear();
        result.exists.reserve(keys.size());
        for (const auto& key : keys) { result.exists.emplace_back(storedKeys_.count(key) != 0); }
        result.prefixHitKeys = 0;
        for (auto exists : result.exists) {
            if (exists == 0) { break; }
            ++result.prefixHitKeys;
        }
        taskResult.queryResult = std::move(result);
        taskId = nextTaskId_++;
        taskResults_.emplace(taskId, std::move(taskResult));
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status LoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                              UC::ASU::TaskId& taskId) override
    {
        state_->lastLoadEntries = entries;
        return Submit(entries, taskId);
    }

    UC::ASU::Status StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                               UC::ASU::TaskId& taskId) override
    {
        state_->lastStoreEntries = entries;
        for (const auto& entry : entries) { storedKeys_.emplace(entry.key); }
        return Submit(entries, taskId);
    }

    UC::ASU::Status BatchLoadAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                                   UC::ASU::TaskId& taskId) override
    {
        return LoadAsync(entries, taskId);
    }

    UC::ASU::Status BatchStoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                                    UC::ASU::TaskId& taskId) override
    {
        return StoreAsync(entries, taskId);
    }

    UC::ASU::Status DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                                UC::ASU::TaskId& taskId) override
    {
        for (const auto& key : keys) { storedKeys_.erase(key); }
        return Submit(keys.size(), taskId);
    }

    bool Check(UC::ASU::TaskId taskId) override
    {
        if (!initialized_) { return true; }
        auto iter = taskResults_.find(taskId);
        return iter == taskResults_.end() ||
               iter->second.status.code != UC::ASU::StatusCode::IN_PROGRESS;
    }

    UC::ASU::Status Wait(UC::ASU::TaskId taskId, std::uint64_t timeoutMs,
                         UC::ASU::TaskResult& result) override
    {
        (void)timeoutMs;
        if (!initialized_) { return NotInitialized(); }

        auto iter = taskResults_.find(taskId);
        if (iter == taskResults_.end()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::TASK_NOT_FOUND,
                                          "fake task not found");
        }

        result = iter->second;
        taskResults_.erase(iter);
        return result.status;
    }

    UC::ASU::Status RegisterRegions(
        const std::vector<UC::ASU::MemoryRegion>& regions,
        std::vector<UC::ASU::RegisteredMemory>& registeredRegions) override
    {
        if (!initialized_) { return NotInitialized(); }
        ++state_->registrationCalls;
        if (state_->failRegistration) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::NOT_INITIALIZED,
                                          "injected registration failure");
        }
        registeredRegions.clear();
        registeredRegions.reserve(regions.size());
        state_->registeredRegions.insert(state_->registeredRegions.end(), regions.begin(),
                                         regions.end());
        for (std::size_t index = 0; index < regions.size(); ++index) {
            registeredRegions.emplace_back(
                UC::ASU::RegisteredMemory{regions[index], MakeTestMrHandle(nextMrHandle_++)});
        }
        if (state_->omitRegisteredHandle) { registeredRegions.clear(); }
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status UnregisterRegions(const std::vector<UC::ASU::MRHandle>& handles) override
    {
        (void)handles;
        return UC::ASU::Status::OK();
    }

private:
    UC::ASU::Status Submit(const std::vector<UC::ASU::KVBuffer>& entries, UC::ASU::TaskId& taskId)
    {
        return Submit(entries.size(), taskId);
    }

    UC::ASU::Status Submit(std::size_t entryCount, UC::ASU::TaskId& taskId)
    {
        if (!initialized_) { return NotInitialized(); }

        taskId = nextTaskId_++;
        UC::ASU::TaskResult result;
        result.status = UC::ASU::Status::OK();
        result.entryStatus.assign(entryCount, UC::ASU::Status::OK());
        taskResults_.emplace(taskId, std::move(result));
        return UC::ASU::Status::OK();
    }

    static UC::ASU::Status NotInitialized()
    {
        return UC::ASU::Status::Error(UC::ASU::StatusCode::NOT_INITIALIZED,
                                      "fake ASU client is not initialized");
    }

    std::shared_ptr<FakeAsuClientState> state_;
    bool initialized_{false};
    UC::ASU::TaskId nextTaskId_{1};
    std::uintptr_t nextMrHandle_{1};
    std::unordered_set<UC::ASU::CacheKey, CacheKeyHasher> storedKeys_;
    std::unordered_map<UC::ASU::TaskId, UC::ASU::TaskResult> taskResults_;
};

std::shared_ptr<FakeAsuClientState> UseFakeClient(UC::AsuStore::AsuStore& store)
{
    auto state = std::make_shared<FakeAsuClientState>();
    store.SetClientFactory([state](const UC::AsuStore::Config& config) {
        state->initConfigs.emplace_back(config);
        return std::make_unique<FakeAsuClient>(state);
    });
    return state;
}

UC::Detail::Dictionary MakeBaseConfig()
{
    UC::Detail::Dictionary config;
    config.Set("asu_client_id", std::string{"asu-store-test"});
    config.Set("asu_name_prefix", std::string{"asu-store-test"});
    config.Set("asu_ports", std::vector<ssize_t>{12345});
    config.SetNumber("device_id", -1);
    config.SetNumber("tensor_size", std::size_t{64});
    config.SetNumber("shard_size", std::size_t{64});
    config.SetNumber("block_size", std::size_t{64});
    config.SetNumber("asu_wait_timeout_ms", std::uint64_t{1000});
    config.SetNumber("asu_query_timeout_ms", std::uint64_t{500});
    config.SetNumber("asu_client_max_inflight_tasks", std::uint64_t{16});
    config.SetNumber("asu_transport_max_inflight_tasks", std::uint64_t{16});
    config.Set("kv_ns_ids", std::vector<ssize_t>{100});
    return config;
}

UC::Detail::TaskDesc MakeTask(const UC::Detail::BlockId& block, void* addr)
{
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{block, 0, {addr}});
    return task;
}

void RegisterPersistentRanges(UC::Detail::Dictionary& config,
                              const std::vector<std::pair<void*, std::size_t>>& ranges)
{
    ASSERT_FALSE(ranges.empty());
    std::vector<ssize_t> addrs;
    std::vector<ssize_t> sizes;
    for (const auto& [addr, size] : ranges) {
        addrs.push_back(static_cast<ssize_t>(reinterpret_cast<std::uintptr_t>(addr)));
        sizes.push_back(static_cast<ssize_t>(size));
    }
    config.Set("role", std::string{"worker"});
    config.Set("gpu_kv_buffer_addrs", addrs);
    config.Set("gpu_kv_buffer_sizes", sizes);
}

void ExpectLookupMiss(UC::StoreV1& store, const UC::Detail::BlockId& block)
{
    auto lookup = store.Lookup(&block, 1);
    ASSERT_TRUE(lookup.HasValue()) << lookup.Error().ToString();
    const std::vector<std::uint8_t> expected{0};
    ASSERT_EQ(lookup.Value(), expected);

    auto prefix = store.LookupOnPrefix(&block, 1);
    ASSERT_TRUE(prefix.HasValue()) << prefix.Error().ToString();
    ASSERT_EQ(prefix.Value(), -1);
}

void ExpectLoadDumpSmoke(UC::StoreV1& store, const UC::Detail::BlockId& block, void* buffer)
{
    auto dump = store.Dump(MakeTask(block, buffer));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_TRUE(store.Wait(dump.Value()).Success());

    auto load = store.Load(MakeTask(block, buffer));
    ASSERT_TRUE(load.HasValue()) << load.Error().ToString();
    ASSERT_TRUE(store.Wait(load.Value()).Success());
}

}  // namespace

TEST(UCAsuStoreTest, TransportModeRejectsMultipleAsus)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ports", std::vector<ssize_t>{12345, 12346});
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Failure());
}

TEST(UCAsuStoreTest, ParsesKvNamespaces)
{
    {
        UC::AsuStore::AsuStore store;
        auto state = UseFakeClient(store);
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        ASSERT_TRUE(store.Setup(config).Success());
        ASSERT_FALSE(state->initConfigs.empty());
        EXPECT_EQ(state->initConfigs.back().kvNsIds, std::vector<std::uint32_t>{100U});
        auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
        EXPECT_EQ(transportConfig.attrs.at("kv_ns_id"), "100");
    }

    const std::vector<std::pair<std::string, std::uint32_t>> cases{
        {"fa", 100U},
        {"wa", 101U}
    };
    for (const auto& [suffix, expected] : cases) {
        UC::AsuStore::AsuStore store;
        auto state = UseFakeClient(store);
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.Set("unique_id", std::string{"engine_fawa_"} + suffix);
        config.Set("kv_ns_ids", std::vector<ssize_t>{100, 101});
        ASSERT_TRUE(store.Setup(config).Success());
        ASSERT_FALSE(state->initConfigs.empty());
        EXPECT_EQ(state->initConfigs.back().kvNsIds, (std::vector<std::uint32_t>{100U, 101U}));
        auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
        EXPECT_EQ(transportConfig.attrs.at("kv_ns_id"), std::to_string(expected));
    }
}

TEST(UCAsuStoreTest, PropagatesWaitTimeout)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_wait_timeout_ms", std::uint64_t{321});

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().waitTimeoutMs, 321);

    auto asuClientConfig = UC::AsuStore::BuildAsuClientConfig(state->initConfigs.back());
    EXPECT_EQ(asuClientConfig.defaultWaitTimeoutMs, 321);
    EXPECT_EQ(asuClientConfig.timeoutMs, 321);

    auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
    EXPECT_EQ(transportConfig.timeoutMs, 321);
}

TEST(UCAsuStoreTest, RejectsZeroWaitTimeout)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_wait_timeout_ms", std::uint64_t{0});

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, ParsesQueryTimeout)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_query_timeout_ms", std::uint64_t{321});

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().queryTimeoutMs, 321);
}

TEST(UCAsuStoreTest, RejectsZeroQueryTimeout)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_query_timeout_ms", std::uint64_t{0});

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, PropagatesMaxErrorCountToTransport)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_max_error_count", std::uint64_t{7});

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().maxErrorCount, std::uint64_t{7});

    auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
    EXPECT_EQ(transportConfig.maxErrorCount, std::uint32_t{7});
}

TEST(UCAsuStoreTest, PropagatesSeparateMaxInflightTasks)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_client_max_inflight_tasks", std::uint64_t{17});
    config.SetNumber("asu_transport_max_inflight_tasks", std::uint64_t{23});
    config.SetNumber("asu_completion_poll_spin_limit", std::uint64_t{19});

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());

    const auto asuConfig = UC::AsuStore::BuildAsuClientConfig(state->initConfigs.back());
    EXPECT_EQ(asuConfig.maxInflightTasks, std::uint32_t{17});
    ASSERT_EQ(asuConfig.transportConfigs.size(), std::size_t{1});
    EXPECT_EQ(asuConfig.transportConfigs.front().maxInflightTasks, std::uint32_t{23});
    EXPECT_EQ(asuConfig.transportConfigs.front().completionPollSpinLimit, std::size_t{19});
}

TEST(UCAsuStoreTest, FakeProviderPreservesConfiguredSc)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("asu_trans_provider_backend", std::string{"fake"});
    config.Set("asu_sc", false);

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());

    const auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
    EXPECT_EQ(transportConfig.attrs.at("sc"), "false");
}

TEST(UCAsuStoreTest, PropagatesFakeBackendWorkerThreads)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("asu_trans_provider_backend", std::string{"fake"});
    config.SetNumber("asu_fake_backend_worker_threads", std::uint64_t{8});

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());

    const auto transportConfig = UC::AsuStore::BuildTransportConfig(state->initConfigs.back(), 0);
    EXPECT_EQ(transportConfig.attrs.at("fake_backend.worker_threads"), "8");
}

TEST(UCAsuStoreTest, RejectsMissingKvNamespaces)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("kv_ns_ids", std::vector<ssize_t>{});

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, RejectsUnexpectedKvNamespaceCount)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("kv_ns_ids", std::vector<ssize_t>{100, 101});

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, RejectsInvalidAsuIds)
{
    for (const auto& asuIds : std::vector<std::vector<ssize_t>>{
             {-1},
             {1001, 1001}
    }) {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", asuIds);

        EXPECT_TRUE(store.Setup(config).Failure());
    }
}

TEST(UCAsuStoreTest, RejectsInvalidAsuPorts)
{
    for (auto port : {ssize_t{-1}, ssize_t{0}, ssize_t{65536}}) {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
        config.Set("asu_ports", std::vector<ssize_t>{port});

        EXPECT_TRUE(store.Setup(config).Failure());
    }
}

TEST(UCAsuStoreTest, RejectsMismatchedAsuPorts)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ports", std::vector<ssize_t>{12345});

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, ParsesSharedProviderMode)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ports", std::vector<ssize_t>{12345, 12346});
    config.SetNumber("asu_shared_provider", 1);

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().sharedProviderMode, 1);
    const auto asuConfig = UC::AsuStore::BuildAsuClientConfig(state->initConfigs.back());
    EXPECT_EQ(asuConfig.sharedProviderMode, UC::ASU::SharedProviderMode::SHARED);
    ASSERT_EQ(asuConfig.transportConfigs.size(), std::size_t{2});
    EXPECT_EQ(asuConfig.transportConfigs[0].asuId, UC::ASU::AsuId{1001});
    EXPECT_EQ(asuConfig.transportConfigs[0].endpoints[0].ip, "127.0.0.1");
    EXPECT_EQ(asuConfig.transportConfigs[0].endpoints[0].port, std::uint16_t{12345});
    EXPECT_EQ(asuConfig.transportConfigs[1].asuId, UC::ASU::AsuId{1002});
    EXPECT_EQ(asuConfig.transportConfigs[1].endpoints[0].ip, "127.0.0.2");
    EXPECT_EQ(asuConfig.transportConfigs[1].endpoints[0].port, std::uint16_t{12346});
}

TEST(UCAsuStoreTest, RejectsInvalidSharedProviderMode)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("asu_shared_provider", 2);

    EXPECT_TRUE(store.Setup(config).Failure());
}

TEST(UCAsuStoreTest, RejectsAsuIntegerOverflow)
{
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.SetNumber("asu_max_error_count", std::uint64_t{0});

        EXPECT_TRUE(store.Setup(config).Failure());
    }
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.SetNumber("asu_max_error_count",
                         std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1);

        EXPECT_TRUE(store.Setup(config).Failure());
    }
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.SetNumber("asu_client_max_inflight_tasks",
                         std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1);

        EXPECT_TRUE(store.Setup(config).Failure());
    }
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.SetNumber("asu_transport_max_inflight_tasks",
                         std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1);

        EXPECT_TRUE(store.Setup(config).Failure());
    }
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.SetNumber("block_size",
                         std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1);

        EXPECT_TRUE(store.Setup(config).Failure());
    }
}

TEST(UCAsuStoreTest, AivRequiresDeviceId)
{
    {
        UC::AsuStore::AsuStore store;
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.Set("asu_trans_provider_backend", std::string{"aiv"});

        EXPECT_TRUE(store.Setup(config).Failure());
    }
    {
        UC::AsuStore::AsuStore store;
        UseFakeClient(store);
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.Set("asu_trans_provider_backend", std::string{"aiv"});
        config.SetNumber("device_id", 0);

        EXPECT_TRUE(store.Setup(config).Success());
    }
}

TEST(UCAsuStoreTest, PropagatesKvNamespaceToEveryTransport)
{
    UC::AsuStore::Config config;
    config.asuIds = {1001, 1002};
    config.kvNsIds = {100};
    config.deviceId = 3;
    config.localIp = "192.168.0.3";
    config.transProviderType = UC::ASU::TransProviderType::FAKE;

    for (std::size_t index = 0; index < config.asuIds.size(); ++index) {
        auto transportConfig = UC::AsuStore::BuildTransportConfig(config, index);
        EXPECT_EQ(transportConfig.deviceId, 3);
        EXPECT_EQ(transportConfig.attrs.at("localIp"), "192.168.0.3");
        auto iter = transportConfig.attrs.find("kv_ns_id");
        ASSERT_NE(iter, transportConfig.attrs.end());
        EXPECT_EQ(iter->second, "100");
    }
}

TEST(UCAsuStoreTest, SchedulerUsesConfiguredDeviceId)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("role", std::string{"scheduler"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("device_id", 5);

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().deviceId, 5);
}

TEST(UCAsuStoreTest, WorkerUsesLocalRankDeviceId)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("role", std::string{"worker"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("device_id", 2);

    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().deviceId, 2);
    EXPECT_EQ(state->registrationCalls, std::size_t{0});
}

TEST(UCAsuStoreTest, TransportModeSmoke)
{
    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block, buffer.data());
}

TEST(UCAsuStoreTest, ClientModeSmoke)
{
    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ports", std::vector<ssize_t>{12345, 12346});
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("b1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block, buffer.data());
}

TEST(UCAsuStoreTest, AllowsQueryOnlyConfigWithoutTensorSizes)
{
    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("role", std::string{"scheduler"});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
}

TEST(UCAsuStoreTest, SchedulerRoleSkipsTransferLayoutValidation)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("role", std::string{"scheduler"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{16});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_FALSE(state->initConfigs.empty());
    EXPECT_EQ(state->initConfigs.back().role, "scheduler");
    ASSERT_EQ(state->initConfigs.back().tensorSizes.size(), std::size_t{1});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[0], std::size_t{16});
}

TEST(UCAsuStoreTest, WorkerRoleRequiresTensorSizes)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("role", std::string{"worker"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{0});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Failure());
}
TEST(UCAsuStoreTest, UsesPersistentRegionHandleBeforeSubmitAndKeepsItAfterWait)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a2b2c3d4e5f6789012345678901234ab");
    auto dump = store.Dump(MakeTask(block, buffer.data()));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();

    ASSERT_EQ(state->lastStoreEntries.size(), std::size_t{1});
    ASSERT_EQ(state->registeredRegions.size(), std::size_t{1});
    EXPECT_NE(state->lastStoreEntries[0].buffer.handle, UC::ASU::kInvalidMRHandle);
    ASSERT_TRUE(store.Wait(dump.Value()).Success());
}

TEST(UCAsuStoreTest, PersistentRegionHandleIsSharedByEntriesAndNotReleasedPerTask)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{64, 64});
    config.SetNumber("shard_size", std::size_t{128});
    config.SetNumber("block_size", std::size_t{128});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes * 2> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());
    ASSERT_EQ(state->registeredRegions.size(), std::size_t{1});

    UC::Detail::TaskDesc task;
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("c2b2c3d4e5f6789012345678901234ab");
    task.push_back(UC::Detail::Shard{
        block, 0, {buffer.data(), buffer.data() + UC::ASU::kAsuAlignmentBytes}
    });

    auto dump = store.Dump(std::move(task));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->registeredRegions.size(), std::size_t{1});
    ASSERT_EQ(state->lastStoreEntries.size(), std::size_t{2});
    const auto sharedHandle = state->lastStoreEntries[0].buffer.handle;
    EXPECT_NE(sharedHandle, UC::ASU::kInvalidMRHandle);
    EXPECT_EQ(state->lastStoreEntries[1].buffer.handle, sharedHandle);

    ASSERT_TRUE(store.Wait(dump.Value()).Success());
}

TEST(UCAsuStoreTest, ResolvesEachEntryToItsContainingPersistentRegion)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{64, 64});
    config.SetNumber("shard_size", std::size_t{128});
    config.SetNumber("block_size", std::size_t{128});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> firstBuffer{};
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> secondBuffer{};
    RegisterPersistentRanges(config, {
                                         {firstBuffer.data(),  firstBuffer.size() },
                                         {secondBuffer.data(), secondBuffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());

    UC::Detail::TaskDesc task;
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("d2b2c3d4e5f6789012345678901234ab");
    task.push_back(UC::Detail::Shard{
        block, 0, {firstBuffer.data(), secondBuffer.data()}
    });

    auto dump = store.Dump(std::move(task));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->registeredRegions.size(), std::size_t{2});
    ASSERT_EQ(state->lastStoreEntries.size(), std::size_t{2});
    EXPECT_NE(state->lastStoreEntries[0].buffer.handle, state->lastStoreEntries[1].buffer.handle);
}

TEST(UCAsuStoreTest, LookupOnPrefixReturnsLastContiguousHit)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ports", std::vector<ssize_t>{12345, 12346});
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    ASSERT_TRUE(store.Setup(config).Success());
    std::vector<UC::Detail::BlockId> blocks{
        UC::Test::Detail::TypesHelper::MakeBlockId("e1b2c3d4e5f6789012345678901234ab"),
        UC::Test::Detail::TypesHelper::MakeBlockId("f1b2c3d4e5f6789012345678901234ab"),
    };
    auto dump = store.Dump(MakeTask(blocks[0], buffer.data()));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_TRUE(store.Wait(dump.Value()).Success());

    auto prefix = store.LookupOnPrefix(blocks.data(), blocks.size());
    ASSERT_TRUE(prefix.HasValue()) << prefix.Error().ToString();
    ASSERT_EQ(prefix.Value(), 0);
}

TEST(UCAsuStoreTest, ClientModeConfigPathSmoke)
{
    constexpr const char* kConfigPath = "asu_store_client_config_path_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "clientId=asu-store-test\n";
        configFile << "transport.asuIds=1001,1002\n";
        configFile << "wait_timeout_ms=1000\n";
    }

    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_config_path", std::string{kConfigPath});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    auto setupStatus = store.Setup(config);
    std::remove(kConfigPath);
    ASSERT_TRUE(setupStatus.Success()) << setupStatus.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block, buffer.data());
}

TEST(UCAsuStoreTest, TransportModeConfigPathSmoke)
{
    constexpr const char* kConfigPath = "asu_store_transport_config_path_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "asuId=1001\n";
        configFile << "asuName=asu-store-test\n";
        configFile << "endpoint=127.0.0.1:12345:tcp\n";
        configFile << "maxInflightTasks=16\n";
    }

    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_config_path", std::string{kConfigPath});
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    auto setupStatus = store.Setup(config);
    std::remove(kConfigPath);
    ASSERT_TRUE(setupStatus.Success()) << setupStatus.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("d1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block, buffer.data());
}

TEST(UCAsuStoreTest, RejectsOddMultiTensorLayout)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{16, 16, 16});
    config.SetNumber("shard_size", std::size_t{48});
    config.SetNumber("block_size", std::size_t{48});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Failure());
}

TEST(UCAsuStoreTest, UsesNonLayerwiseMlaTensorOffsets)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"mla"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 120, 140});
    config.SetNumber("shard_size", std::size_t{360});
    config.SetNumber("block_size", std::size_t{360});

    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> first{};
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> second{};
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> third{};
    RegisterPersistentRanges(config, {
                                         {first.data(),  first.size() },
                                         {second.data(), second.size()},
                                         {third.data(),  third.size() }
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_FALSE(state->initConfigs.empty());
    ASSERT_EQ(state->initConfigs.back().tensorSizes.size(), std::size_t{3});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[0], std::size_t{512});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[1], std::size_t{512});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[2], std::size_t{512});

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("d1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block, 0, {first.data(), second.data(), third.data()}
    });

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), 3);
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[0].offset, std::uint32_t{0});
    EXPECT_EQ(state->lastStoreEntries[1].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[1].offset, std::uint32_t{512});
    EXPECT_EQ(state->lastStoreEntries[2].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[2].offset, std::uint32_t{1024});
}

TEST(UCAsuStoreTest, UsesHmaTensorOffsetsInConnectorOrder)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"hma"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200, 300});
    config.SetNumber("shard_size", std::size_t{600});
    config.SetNumber("block_size", std::size_t{600});

    std::array<std::array<std::byte, 512>, 3> buffers{};
    RegisterPersistentRanges(config, {
                                         {buffers[0].data(), buffers[0].size()},
                                         {buffers[1].data(), buffers[1].size()},
                                         {buffers[2].data(), buffers[2].size()}
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("f1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block, 0, {buffers[0].data(), buffers[1].data(), buffers[2].data()}
    });

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), 3);
    const std::array<std::uint32_t, 3> expectedOffsets{0, 512, 1024};
    for (std::size_t index = 0; index < expectedOffsets.size(); ++index) {
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.addr,
                  reinterpret_cast<std::uint64_t>(buffers[index].data()));
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.size, std::size_t{512});
        EXPECT_EQ(state->lastStoreEntries[index].offset, expectedOffsets[index]);
    }
}

TEST(UCAsuStoreTest, UsesLayerwiseMlaTensorOffsets)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"mla"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{128});
    config.SetNumber("shard_size", std::size_t{128});
    config.SetNumber("block_size", std::size_t{384});

    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("e1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{block, 2, {buffer.data()}});

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), 1);
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.addr,
              reinterpret_cast<std::uint64_t>(buffer.data()));
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[0].offset, std::uint32_t{1024});
}

TEST(UCAsuStoreTest, UsesMultiSegmentLayerwiseMlaTensorOffsets)
{
    constexpr std::size_t shardIndex = 2;
    constexpr std::size_t alignment = UC::ASU::kAsuAlignmentBytes;
    constexpr std::size_t alignedShardSize = alignment * 2;

    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"mla"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{128, 64});
    config.SetNumber("shard_size", std::size_t{192});
    config.SetNumber("block_size", std::size_t{576});

    std::array<std::byte, alignment> mainCache{};
    std::array<std::byte, alignment> ropeCache{};
    const std::array<void*, 2> buffers{mainCache.data(), ropeCache.data()};
    RegisterPersistentRanges(
        config, {
                    {mainCache.data(), mainCache.size()},
                    {ropeCache.data(), ropeCache.size()}
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("e1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block, shardIndex, {buffers[0], buffers[1]}
    });

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), buffers.size());

    const std::array<std::uint32_t, 2> expectedOffsets{
        shardIndex * alignedShardSize,
        shardIndex * alignedShardSize + alignment,
    };
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.addr,
                  reinterpret_cast<std::uint64_t>(buffers[index]));
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.size, alignment);
        EXPECT_EQ(state->lastStoreEntries[index].offset, expectedOffsets[index]);
    }
}

TEST(UCAsuStoreTest, AlignsTensorSizeAndDerivesShardBlockSize)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{100});
    config.SetNumber("shard_size", std::size_t{100});
    config.SetNumber("block_size", std::size_t{300});

    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> buffer{};
    RegisterPersistentRanges(config, {
                                         {buffer.data(), buffer.size()}
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_FALSE(state->initConfigs.empty());
    ASSERT_EQ(state->initConfigs.back().tensorSizes.size(), std::size_t{1});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[0],
              static_cast<std::size_t>(UC::ASU::kAsuAlignmentBytes));
    EXPECT_EQ(state->initConfigs.back().shardSize,
              static_cast<std::size_t>(UC::ASU::kAsuAlignmentBytes));
    EXPECT_EQ(state->initConfigs.back().blockSize,
              static_cast<std::size_t>(UC::ASU::kAsuAlignmentBytes) * 3);

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("abb2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{block, 2, {buffer.data()}});

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_FALSE(state->lastStoreEntries.empty());
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.size,
              static_cast<std::size_t>(UC::ASU::kAsuAlignmentBytes));
    EXPECT_EQ(state->lastStoreEntries[0].offset, UC::ASU::kAsuAlignmentBytes * 2);
}

TEST(UCAsuStoreTest, AllowsMultipleShardsPerBlock)
{
    UC::AsuStore::AsuStore store;
    UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("block_size", std::size_t{128});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
}

TEST(UCAsuStoreTest, UsesLayerwiseGqaKeyValueOffsets)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200});
    config.SetNumber("shard_size", std::size_t{300});
    config.SetNumber("block_size", std::size_t{900});

    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> key{};
    std::array<std::byte, UC::ASU::kAsuAlignmentBytes> value{};
    RegisterPersistentRanges(config, {
                                         {key.data(),   key.size()  },
                                         {value.data(), value.size()}
    });
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("b1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block, 2, {key.data(), value.data()}
    });

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), 2);
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.addr,
              reinterpret_cast<std::uint64_t>(key.data()));
    EXPECT_EQ(state->lastStoreEntries[0].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[0].offset, std::uint32_t{1024});
    EXPECT_EQ(state->lastStoreEntries[1].buffer.region.addr,
              reinterpret_cast<std::uint64_t>(value.data()));
    EXPECT_EQ(state->lastStoreEntries[1].buffer.region.size, std::size_t{512});
    EXPECT_EQ(state->lastStoreEntries[1].offset, std::uint32_t{2560});
}

TEST(UCAsuStoreTest, UsesNonLayerwiseGqaKeyValueOffsets)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200, 100, 200, 100, 200});
    config.SetNumber("shard_size", std::size_t{900});
    config.SetNumber("block_size", std::size_t{900});

    std::array<std::array<std::byte, UC::ASU::kAsuAlignmentBytes>, 6> buffers{};
    std::vector<std::pair<void*, std::size_t>> ranges;
    for (auto& buffer : buffers) { ranges.emplace_back(buffer.data(), buffer.size()); }
    RegisterPersistentRanges(config, ranges);
    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block,
        0,
        {buffers[0].data(), buffers[1].data(), buffers[2].data(), buffers[3].data(),
          buffers[4].data(), buffers[5].data()}
    });

    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_EQ(state->lastStoreEntries.size(), 6);
    const std::array<std::uint32_t, 6> expectedOffsets{0, 1536, 512, 2048, 1024, 2560};
    for (std::size_t index = 0; index < expectedOffsets.size(); ++index) {
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.addr,
                  reinterpret_cast<std::uint64_t>(buffers[index].data()));
        EXPECT_EQ(state->lastStoreEntries[index].buffer.region.size, std::size_t{512});
        EXPECT_EQ(state->lastStoreEntries[index].offset, expectedOffsets[index]);
    }
}

TEST(UCAsuStoreTest, RegistersKvCacheRegions)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("role", std::string{"worker"});
    config.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000});
    config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024});
    ASSERT_TRUE(store.Setup(config).Success());
    EXPECT_EQ(state->registrationCalls, std::size_t{1});
    ASSERT_EQ(state->registeredRegions.size(), std::size_t{1});
    EXPECT_EQ(state->registeredRegions[0].addr, std::uint64_t{0x1000});
    EXPECT_EQ(state->registeredRegions[0].size, std::uint64_t{1024});
}

TEST(UCAsuStoreTest, SchedulerSkipsKvCacheRegistration)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeClient(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("role", std::string{"scheduler"});
    config.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000});
    config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024});
    ASSERT_TRUE(store.Setup(config).Success());
    EXPECT_EQ(state->registrationCalls, std::size_t{0});
}

TEST(UCAsuStoreTest, RejectsInvalidKvCacheConfigBeforeClientInit)
{
    const std::vector<std::pair<std::vector<ssize_t>, std::vector<ssize_t>>> invalid{
        {{0x1000}, {}    },
        {{},       {1024}},
        {{0},      {1024}},
        {{0x1000}, {0}   },
        {{-1},     {1024}},
        {{0x1000}, {-1}  }
    };
    for (const auto& [addrs, sizes] : invalid) {
        UC::AsuStore::AsuStore store;
        auto state = UseFakeClient(store);
        auto config = MakeBaseConfig();
        config.Set("asu_ids", std::vector<ssize_t>{1001});
        config.Set("role", std::string{"worker"});
        config.Set("gpu_kv_buffer_addrs", addrs);
        config.Set("gpu_kv_buffer_sizes", sizes);
        EXPECT_TRUE(store.Setup(config).Failure());
        EXPECT_TRUE(state->initConfigs.empty());
        EXPECT_EQ(state->registrationCalls, std::size_t{0});
    }
}

TEST(UCAsuStoreTest, RegistrationFailureShutsDownClient)
{
    for (const bool missingHandle : {false, true}) {
        auto state = std::make_shared<FakeAsuClientState>();
        {
            UC::AsuStore::AsuStore store;
            state = UseFakeClient(store);
            state->failRegistration = !missingHandle;
            state->omitRegisteredHandle = missingHandle;
            auto config = MakeBaseConfig();
            config.Set("asu_ids", std::vector<ssize_t>{1001});
            config.Set("role", std::string{"worker"});
            config.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{0x1000});
            config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{1024});
            EXPECT_TRUE(store.Setup(config).Failure());
            EXPECT_EQ(state->registrationCalls, std::size_t{1});
            EXPECT_EQ(state->shutdownCalls, std::size_t{1});
        }
        EXPECT_EQ(state->shutdownCalls, std::size_t{1});
    }
}
