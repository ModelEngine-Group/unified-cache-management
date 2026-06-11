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

struct FakeAsuBackendState {
    std::vector<UC::ASU::QueryMode> queryModes;
};

class FakeAsuBackend final : public UC::AsuStore::AsuBackend {
public:
    explicit FakeAsuBackend(std::shared_ptr<FakeAsuBackendState> state) : state_(std::move(state))
    {
    }

    UC::ASU::Status Init(const UC::AsuStore::Config& config) override
    {
        config_ = config;
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
        return Submit(entries, taskId);
    }

    UC::ASU::Status StoreAsync(const std::vector<UC::ASU::KVBuffer>& entries,
                               UC::ASU::TaskId& taskId) override
    {
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
    std::unordered_set<UC::ASU::CacheKey> storedKeys_;
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

TEST(UCAsuStoreTest, RejectsInvalidTensorLayout)
{
    UC::AsuStore::AsuStore store;
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{32, 16});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Failure());
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

TEST(UCAsuStoreTest, AllowsMultipleTensorBuffersPerShard)
{
    UC::AsuStore::AsuStore store;
    UseFakeBackend(store);
    auto config = MakeBaseConfig();
    config.Set("asu_mode", std::string{"transport"});
    config.Set("asu_ips", std::vector<std::string>{"127.0.0.1"});
    config.Set("asu_ids", std::vector<ssize_t>{1001});
    config.SetNumber("tensor_size", std::size_t{0});
    config.Set("tensor_size_list", std::vector<ssize_t>{32, 32});

    auto status = store.Setup(config);
    ASSERT_TRUE(status.Success()) << status.ToString();

    std::array<std::byte, 32> first{};
    std::array<std::byte, 32> second{};
    auto block = UC::Test::Detail::TypesHelper::MakeBlockId("aab2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc task;
    task.brief = "asu-store-test";
    task.push_back(UC::Detail::Shard{
        block, 0, {first.data(), second.data()}
    });
    auto dump = store.Dump(task);
    ASSERT_TRUE(dump.HasValue()) << dump.Error().ToString();
    ASSERT_TRUE(store.Wait(dump.Value()).Success());
}
