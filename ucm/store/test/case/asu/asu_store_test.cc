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

struct FakeAsuBackendState {
    std::vector<UC::ASU::QueryMode> queryModes;
    std::vector<UC::AsuStore::Config> initConfigs;
    std::vector<UC::ASU::KVBuffer> lastLoadEntries;
    std::vector<UC::ASU::KVBuffer> lastStoreEntries;
};

class FakeAsuBackend final : public UC::AsuStore::AsuBackend {
public:
    explicit FakeAsuBackend(std::shared_ptr<FakeAsuBackendState> state) : state_(std::move(state))
    {
    }

    UC::ASU::Status Init(const UC::AsuStore::Config& config) override
    {
        config_ = config;
        state_->initConfigs.emplace_back(config);
        initialized_ = true;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Init(const std::string& configPath) override
    {
        configPath_ = configPath;
        initialized_ = true;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Shutdown() override
    {
        initialized_ = false;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Query(const std::vector<UC::ASU::CacheKey>& keys,
                          const UC::ASU::QueryOptions& options,
                          UC::ASU::QueryResult& result) override
    {
        if (!initialized_) { return NotInitialized(); }

        state_->queryModes.emplace_back(options.mode);
        result.exists.clear();
        result.exists.reserve(keys.size());
        for (const auto& key : keys) { result.exists.emplace_back(storedKeys_.count(key) != 0); }
        result.prefixHitKeys = 0;
        if (options.mode == UC::ASU::QueryMode::PREFIX) {
            for (auto exists : result.exists) {
                if (exists == 0) { break; }
                ++result.prefixHitKeys;
            }
        }
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

    UC::ASU::Status DeleteAsync(const std::vector<UC::ASU::CacheKey>& keys,
                                UC::ASU::TaskId& taskId) override
    {
        for (const auto& key : keys) { storedKeys_.erase(key); }
        return Submit(keys.size(), taskId);
    }

    UC::ASU::Status Check(UC::ASU::TaskId taskId, UC::ASU::TaskResult& result) override
    {
        if (!initialized_) { return NotInitialized(); }

        auto iter = taskResults_.find(taskId);
        if (iter == taskResults_.end()) {
            return UC::ASU::Status::Error(UC::ASU::StatusCode::TASK_NOT_FOUND,
                                          "fake task not found");
        }

        result = iter->second;
        return UC::ASU::Status::OK();
    }

    UC::ASU::Status Wait(UC::ASU::TaskId taskId, std::uint64_t timeoutMs,
                         UC::ASU::TaskResult& result) override
    {
        (void)timeoutMs;
        return Check(taskId, result);
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
                                      "fake ASU backend is not initialized");
    }

    UC::AsuStore::Config config_;
    std::string configPath_;
    std::shared_ptr<FakeAsuBackendState> state_;
    bool initialized_{false};
    UC::ASU::TaskId nextTaskId_{1};
    std::unordered_set<UC::ASU::CacheKey, CacheKeyHasher> storedKeys_;
    std::unordered_map<UC::ASU::TaskId, UC::ASU::TaskResult> taskResults_;
};

std::shared_ptr<FakeAsuBackendState> UseFakeBackend(UC::AsuStore::AsuStore& store)
{
    auto state = std::make_shared<FakeAsuBackendState>();
    store.SetBackendFactory(
        [state](const UC::AsuStore::Config&) { return std::make_unique<FakeAsuBackend>(state); });
    return state;
}

UC::Detail::Dictionary MakeBaseConfig()
{
    UC::Detail::Dictionary config;
    config.Set("asu_client_id", std::string{"asu-store-test"});
    config.Set("asu_name_prefix", std::string{"asu-store-test"});
    config.SetNumber("asu_port", 12345);
    config.SetNumber("device_id", -1);
    config.SetNumber("tensor_size", std::size_t{64});
    config.SetNumber("shard_size", std::size_t{64});
    config.SetNumber("block_size", std::size_t{64});
    config.SetNumber("asu_default_wait_timeout_ms", std::uint64_t{1000});
    config.SetNumber("asu_query_timeout_ms", std::uint64_t{1000});
    config.SetNumber("asu_load_timeout_ms", std::uint64_t{1000});
    config.SetNumber("asu_store_timeout_ms", std::uint64_t{1000});
    config.SetNumber("asu_max_inflight_tasks", std::uint64_t{16});
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

void ExpectLoadDumpSmoke(UC::StoreV1& store, const UC::Detail::BlockId& block)
{
    std::array<std::byte, 64> buffer{};
    auto dump = store.Dump(MakeTask(block, buffer.data()));
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_TRUE(store.Wait(dump.Value()).Success());

    auto load = store.Load(MakeTask(block, buffer.data()));
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
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Failure());
}

TEST(UCAsuStoreTest, ParsesKvNamespaces)
{
    {
        UC::AsuStore::AsuStore store;
        auto state = UseFakeBackend(store);
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
        auto state = UseFakeBackend(store);
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

TEST(UCAsuStoreTest, PropagatesKvNamespaceToEveryTransport)
{
    UC::AsuStore::Config config;
    config.asuIds = {1001, 1002};
    config.kvNsIds = {100};
    config.transProviderType = UC::ASU::TransProviderType::FAKE;

    for (std::size_t index = 0; index < config.asuIds.size(); ++index) {
        auto transportConfig = UC::AsuStore::BuildTransportConfig(config, index);
        auto iter = transportConfig.attrs.find("kv_ns_id");
        ASSERT_NE(iter, transportConfig.attrs.end());
        EXPECT_EQ(iter->second, "100");
    }
}

TEST(UCAsuStoreTest, TransportModeSmoke)
{
    UC::AsuStore::AsuStore store;
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    ASSERT_TRUE(store.Setup(config).Success());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block);
}

TEST(UCAsuStoreTest, ClientModeSmoke)
{
    UC::AsuStore::AsuStore store;
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    ASSERT_TRUE(store.Setup(config).Success());

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("b1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block);
}

TEST(UCAsuStoreTest, AllowsQueryOnlyConfigWithoutTensorSizes)
{
    UC::AsuStore::AsuStore store;
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("asu_ids", std::vector<ssize_t>{1001});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    auto block =
        UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
}

TEST(UCAsuStoreTest, LookupOnPrefixUsesPrefixQueryMode)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1", "127.0.0.2"});
    config.Set("asu_ids", std::vector<ssize_t>{1001, 1002});
    ASSERT_TRUE(store.Setup(config).Success());

    std::array<std::byte, 64> buffer{};
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
    ASSERT_FALSE(state->queryModes.empty());
    EXPECT_EQ(state->queryModes.back(), UC::ASU::QueryMode::PREFIX);
}

TEST(UCAsuStoreTest, ClientModeConfigPathSmoke)
{
    constexpr const char* kConfigPath = "asu_store_client_config_path_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "clientId=asu-store-test\n";
        configFile << "transport.asuIds=1001,1002\n";
        configFile << "defaultWaitTimeoutMs=1000\n";
    }

    UC::AsuStore::AsuStore store;
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_config_path", std::string{kConfigPath});
    auto setupStatus = store.Setup(config);
    std::remove(kConfigPath);
    ASSERT_TRUE(setupStatus.Success()) << setupStatus.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("c1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block);
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
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_config_path", std::string{kConfigPath});
    auto setupStatus = store.Setup(config);
    std::remove(kConfigPath);
    ASSERT_TRUE(setupStatus.Success()) << setupStatus.ToString();

    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("d1b2c3d4e5f6789012345678901234ab");
    ExpectLookupMiss(store, block);
    ExpectLoadDumpSmoke(store, block);
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
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"mla"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 120, 140});
    config.SetNumber("shard_size", std::size_t{360});
    config.SetNumber("block_size", std::size_t{360});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();
    ASSERT_FALSE(state->initConfigs.empty());
    ASSERT_EQ(state->initConfigs.back().tensorSizes.size(), std::size_t{3});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[0], std::size_t{512});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[1], std::size_t{512});
    EXPECT_EQ(state->initConfigs.back().tensorSizes[2], std::size_t{512});

    std::array<std::byte, 140> first{};
    std::array<std::byte, 140> second{};
    std::array<std::byte, 140> third{};
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
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"hma"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200, 300});
    config.SetNumber("shard_size", std::size_t{600});
    config.SetNumber("block_size", std::size_t{600});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::array<std::array<std::byte, 512>, 3> buffers{};
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
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"mla"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{128});
    config.SetNumber("shard_size", std::size_t{128});
    config.SetNumber("block_size", std::size_t{384});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::array<std::byte, 128> buffer{};
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

TEST(UCAsuStoreTest, AlignsTensorSizeAndDerivesShardBlockSize)
{
    UC::AsuStore::AsuStore store;
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{100});
    config.SetNumber("shard_size", std::size_t{100});
    config.SetNumber("block_size", std::size_t{300});

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

    std::array<std::byte, 100> buffer{};
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
    UseFakeBackend(store);
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
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200});
    config.SetNumber("shard_size", std::size_t{300});
    config.SetNumber("block_size", std::size_t{900});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::array<std::byte, 100> key{};
    std::array<std::byte, 200> value{};
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
    auto state = UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.Set("tensor_layout", std::string{"gqa"});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{100, 200, 100, 200, 100, 200});
    config.SetNumber("shard_size", std::size_t{900});
    config.SetNumber("block_size", std::size_t{900});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::array<std::array<std::byte, 200>, 6> buffers{};
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
