#include "asu_client_impl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kv_common/router.h"

namespace UC::ASU {

static CacheKey MakeCacheKey(std::string_view text)
{
    CacheKey key{};
    if (text.size() <= key.size()) {
        if (!text.empty()) { std::memcpy(key.data(), text.data(), text.size()); }
        return key;
    }
    const auto hash = std::hash<std::string_view>{}(text);
    std::memcpy(key.data(), &hash, key.size());
    return key;
}

struct TestState {
    std::uint32_t createdTransports{0};
    std::unordered_map<AsuId, TransportConfig> initConfigs;
    bool failFirstQuery{false};
    bool firstQueryFailed{false};
    StatusCode firstQueryFailureCode{StatusCode::CONNECTION_ERROR};
    std::string firstQueryFailureMessage{"fake connection error"};
    bool failFirstLoad{false};
    bool firstLoadFailed{false};
    bool failFirstStore{false};
    bool firstStoreFailed{false};
    bool failStoreAfterFirstDispatch{false};
    std::size_t storeDispatchAttempts{0};
    bool failFirstDelete{false};
    bool firstDeleteFailed{false};
    std::unordered_map<AsuId, Status> queryFailures;
    std::unordered_map<AsuId, Status> loadFailures;
    std::unordered_map<AsuId, Status> storeFailures;
    std::unordered_map<AsuId, Status> deleteFailures;
    std::unordered_map<AsuId, std::vector<Status>> checkEntryStatus;
    std::unordered_map<AsuId, Status> checkResultStatus;
    std::unordered_map<AsuId, QueryResult> prefixQueryResults;
    std::vector<AsuId> registerCalls;
    std::vector<AsuId> bindCalls;
    std::unordered_map<AsuId, std::vector<RegisteredMemory>> boundRegions;
    std::vector<AsuId> unregisterCalls;
    std::vector<AsuId> queryCalls;
    std::unordered_map<AsuId, std::size_t> queryKeyCounts;
    std::unordered_map<AsuId, std::vector<CacheKey>> queryKeys;
    std::vector<AsuId> loadCalls;
    std::vector<AsuId> storeCalls;
    std::vector<AsuId> deleteCalls;
    std::vector<AsuId> checkCalls;
    std::vector<AsuId> waitCalls;
    std::vector<AsuId> cancelCalls;
    std::unordered_map<AsuId, TaskId> childTaskIds;
};

class FakeTransport : public AsuTransport {
public:
    explicit FakeTransport(std::shared_ptr<TestState> state) : state_(std::move(state)) {}

    Status Init(const TransportConfig& config) override
    {
        config_ = config;
        state_->initConfigs[config_.asuId] = config_;
        initialized_ = true;
        return Status::OK();
    }

    Status Init(const std::string& configPath) override
    {
        (void)configPath;
        return Status::Error(StatusCode::UNSUPPORTED, "fake transport config path is unsupported");
    }

    Status Shutdown() override
    {
        initialized_ = false;
        return Status::OK();
    }

    Status CheckHealth() override { return initialized_ ? Status::OK() : NotInitialized(); }

    Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                 QueryResult& result) override
    {
        if (!initialized_) { return NotInitialized(); }
        if (state_->failFirstQuery && !state_->firstQueryFailed) {
            state_->firstQueryFailed = true;
            return Status::Error(state_->firstQueryFailureCode, state_->firstQueryFailureMessage);
        }

        state_->queryCalls.emplace_back(config_.asuId);
        state_->queryKeyCounts[config_.asuId] += keys.size();
        auto& routedKeys = state_->queryKeys[config_.asuId];
        routedKeys.insert(routedKeys.end(), keys.begin(), keys.end());
        auto failureIter = state_->queryFailures.find(config_.asuId);
        if (failureIter != state_->queryFailures.end()) { return failureIter->second; }

        if (options.mode == QueryMode::PREFIX) {
            auto iter = state_->prefixQueryResults.find(config_.asuId);
            if (iter != state_->prefixQueryResults.end()) {
                result = iter->second;
                return Status::OK();
            }
        }

        result.exists.clear();
        result.exists.reserve(keys.size());
        for (const auto& key : keys) {
            result.exists.emplace_back(key == MakeCacheKey("k15") || key == MakeCacheKey("k25"));
        }
        result.prefixHitKeys = 0;
        return Status::OK();
    }

    Status QueryAsync(const std::vector<CacheKey>&, const QueryOptions&, TaskId& taskId) override
    {
        taskId = 0;
        return Status::OK();
    }

    Status LoadAsync(const std::vector<KVBuffer>&, TaskId& taskId) override
    {
        if (state_->failFirstLoad && !state_->firstLoadFailed) {
            state_->firstLoadFailed = true;
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake load connection error");
        }
        auto failureIter = state_->loadFailures.find(config_.asuId);
        if (failureIter != state_->loadFailures.end()) { return failureIter->second; }

        state_->loadCalls.emplace_back(config_.asuId);
        taskId = 1000 + config_.asuId;
        state_->childTaskIds[config_.asuId] = taskId;
        return Status::OK();
    }

    Status StoreAsync(const std::vector<KVBuffer>&, TaskId& taskId) override
    {
        if (state_->failFirstStore && !state_->firstStoreFailed) {
            state_->firstStoreFailed = true;
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake store connection error");
        }
        auto failureIter = state_->storeFailures.find(config_.asuId);
        if (failureIter != state_->storeFailures.end()) { return failureIter->second; }
        if (state_->failStoreAfterFirstDispatch && ++state_->storeDispatchAttempts > 1) {
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake partial dispatch failure");
        }

        state_->storeCalls.emplace_back(config_.asuId);
        taskId = 2000 + config_.asuId;
        state_->childTaskIds[config_.asuId] = taskId;
        return Status::OK();
    }

    Status DeleteAsync(const std::vector<CacheKey>&, TaskId& taskId) override
    {
        if (state_->failFirstDelete && !state_->firstDeleteFailed) {
            state_->firstDeleteFailed = true;
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake delete connection error");
        }
        auto failureIter = state_->deleteFailures.find(config_.asuId);
        if (failureIter != state_->deleteFailures.end()) { return failureIter->second; }

        state_->deleteCalls.emplace_back(config_.asuId);
        taskId = 3000 + config_.asuId;
        state_->childTaskIds[config_.asuId] = taskId;
        return Status::OK();
    }

    Status Cancel(TaskId) override
    {
        state_->cancelCalls.emplace_back(config_.asuId);
        return Status::OK();
    }

    Status Check(TaskId taskId, TaskResult& result) override
    {
        state_->checkCalls.emplace_back(config_.asuId);
        auto statusIter = state_->checkResultStatus.find(config_.asuId);
        result.status =
            statusIter == state_->checkResultStatus.end() ? Status::OK() : statusIter->second;
        auto entryIter = state_->checkEntryStatus.find(config_.asuId);
        result.entryStatus = entryIter == state_->checkEntryStatus.end()
                                 ? std::vector<Status>{result.status}
                                 : entryIter->second;
        result.queryResult.reset();
        if (taskId == 0) {
            return Status::Error(StatusCode::TASK_NOT_FOUND, "fake task not found");
        }
        return Status::OK();
    }

    Status Wait(TaskId taskId, std::uint64_t, TaskResult& result) override
    {
        state_->waitCalls.emplace_back(config_.asuId);
        return Check(taskId, result);
    }

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisterResult>& results) override
    {
        state_->registerCalls.emplace_back(config_.asuId);
        results.clear();
        for (std::size_t index = 0; index < regions.size(); ++index) {
            results.emplace_back(RegisterResult{Status::OK(), 500 + index, 0, 0,
                                                900 + static_cast<std::uint32_t>(index)});
        }
        return Status::OK();
    }

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                 std::vector<RegisterResult>& results) override
    {
        state_->bindCalls.emplace_back(config_.asuId);
        state_->boundRegions[config_.asuId] = regions;
        results.clear();
        for (const auto& region : regions) {
            results.emplace_back(RegisterResult{Status::OK(), region.handle, region.lkey,
                                                region.rkey, region.tokenId});
        }
        return Status::OK();
    }

    Status UnregisterRegions(const std::vector<MRHandle>&) override
    {
        state_->unregisterCalls.emplace_back(config_.asuId);
        return Status::OK();
    }

private:
    static Status NotInitialized()
    {
        return Status::Error(StatusCode::NOT_INITIALIZED, "fake transport is not initialized");
    }

    std::shared_ptr<TestState> state_;
    TransportConfig config_;
    bool initialized_{false};
};

class FakeViewServer final : public ViewServer {
public:
    explicit FakeViewServer(std::vector<std::vector<AsuId>> views) : views_(std::move(views)) {}

    FakeViewServer(std::vector<std::vector<AsuId>> views, std::vector<std::uint64_t> epochs)
        : views_(std::move(views)), epochs_(std::move(epochs))
    {
    }

    Status GetGlobalView(GlobalView& view) override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (failFetchAt_ != 0 && fetchCount_ + 1 == failFetchAt_) {
            ++fetchCount_;
            return Status::Error(StatusCode::IO_ERROR, "fake view fetch failed");
        }

        auto index = fetchCount_;
        if (index >= views_.size()) { index = views_.size() - 1; }
        ++fetchCount_;

        view = GlobalView{};
        for (auto asuId : views_[index]) { view.asuMap.emplace(asuId, AsuInfo{}); }
        view.viewEpoch = index < epochs_.size() ? epochs_[index] : fetchCount_;
        return Status::OK();
    }

    void FailFetchAt(std::size_t fetchCount)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        failFetchAt_ = fetchCount;
    }
    std::size_t FetchCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return fetchCount_;
    }

private:
    mutable std::mutex mutex_;
    std::size_t fetchCount_{0};
    std::size_t failFetchAt_{0};
    std::vector<std::vector<AsuId>> views_;
    std::vector<std::uint64_t> epochs_;
};

bool WaitForFetchCount(const std::shared_ptr<FakeViewServer>& viewServer,
                       std::size_t expectedFetchCount)
{
    for (std::uint32_t attempt = 0; attempt < 100; ++attempt) {
        if (viewServer->FetchCount() >= expectedFetchCount) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

ViewServerFactory MakeViewServerFactory(const std::shared_ptr<ViewServer>& viewServer)
{
    return [viewServer](const AsuClientConfig&) { return viewServer; };
}

AsuClientConfig MakeConfig(const std::vector<AsuId>& asuIds)
{
    AsuClientConfig config;
    for (auto asuId : asuIds) {
        TransportConfig transportConfig;
        transportConfig.asuId = asuId;
        config.transportConfigs.emplace_back(std::move(transportConfig));
    }
    return config;
}

TransportFactory MakeFactory(const std::shared_ptr<TestState>& state)
{
    return [state] {
        ++state->createdTransports;
        return std::make_unique<FakeTransport>(state);
    };
}

void ExpectSameAsuSet(std::vector<AsuId> actual, std::vector<AsuId> expected)
{
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(actual, expected);
}

CacheKey FindKeyForAsu(const std::vector<AsuId>& asuIds, AsuId targetAsuId)
{
    std::vector<UC::KV::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    auto router = UC::KV::CreateRouter(nodeIds, UC::KV::HashFunction{}, UC::KV::RouterConfig{});
    for (std::uint32_t index = 1; index < 1000000; ++index) {
        std::uint64_t combined = (static_cast<std::uint64_t>(targetAsuId) << 32) | index;
        CacheKey cacheKey{};
        std::memcpy(cacheKey.data(), &combined, sizeof(combined));
        auto routeKey = std::string(CacheKeyView(cacheKey));
        auto routes = router->RouteKeys({routeKey});
        if (routes.size() == 1 && routes.begin()->first == targetAsuId) { return cacheKey; }
    }
    return {};
}

std::vector<KVBuffer> BuildRoutedEntries(const std::vector<AsuId>& routeOrder)
{
    std::vector<KVBuffer> entries;
    entries.reserve(routeOrder.size());
    for (auto asuId : routeOrder) {
        entries.emplace_back(KVBuffer{FindKeyForAsu(routeOrder, asuId), {}});
    }
    return entries;
}

TEST(AsuClientImplTest, Lifecycle_OperationsBeforeInitReturnExpectedErrors)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));

    QueryResult queryResult;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, queryResult);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    TaskId taskId = kInvalidTaskId;
    status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    TaskResult taskResult;
    status = client->Check(1, taskResult);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Lifecycle_InitTwiceReturnsResourceBusy)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    auto config = MakeConfig({10});
    ASSERT_TRUE(client->Init(config).ok());

    auto status = client->Init(config);

    EXPECT_EQ(status.code, StatusCode::RESOURCE_BUSY);
}

TEST(AsuClientImplTest, Lifecycle_ShutdownClearsTasksAndRejectsFutureOperations)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_TRUE(client->Shutdown().ok());

    QueryResult queryResult;
    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, queryResult);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    TaskResult taskResult;
    status = client->Check(taskId, taskResult);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Input_EmptyQueryReturnsEmptyResult)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    QueryResult result;
    auto status = client->Query({}, QueryOptions{}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.exists.empty());
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{0});
    EXPECT_TRUE(state->queryCalls.empty());
}

TEST(AsuClientImplTest, Input_EmptyStoreCreatesCompletableEmptyTask)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync({}, taskId);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    EXPECT_TRUE(state->storeCalls.empty());

    TaskResult result;
    status = client->Check(taskId, result);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.entryStatus.empty());
}

TEST(AsuClientImplTest, Input_EmptyDeleteCreatesCompletableEmptyTask)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->DeleteAsync({}, taskId);
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    EXPECT_TRUE(state->deleteCalls.empty());

    TaskResult result;
    status = client->Check(taskId, result);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.entryStatus.empty());
}

TEST(AsuClientImplTest, Input_EmptyRegisterReturnsEmptyResults)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({}, results);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(results.empty());
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->bindCalls, std::vector<AsuId>({20}));
}

TEST(AsuClientImplTest, Lifecycle_PublicInitLoadsClientConfigFile)
{
    constexpr const char* kConfigPath = "asu_client_impl_client_config_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "clientId=file-init-test\n";
        configFile << "transport.asuIds=10,20\n";
        configFile << "transport.send_buffer_slot_size=8192\n";
        configFile << "transport.send_buffer_slot_num=2\n";
        configFile << "transport.flag_buffer_slot_size=256\n";
        configFile << "transport.flag_buffer_slot_num=32\n";
        configFile << "transport.batch_load_io_num=11\n";
        configFile << "transport.batch_store_io_num=12\n";
        configFile << "transport.delete_io_num=13\n";
        configFile << "transport.query_io_num=14\n";
        configFile << "asuInfo.20=protocol=roce,placement=device,port=6000,"
                   << "local.comm_id=192.168.1.20,local.logical_device_id=6\n";
    }

    auto state = std::make_shared<TestState>();
    std::unique_ptr<AsuClient> client = CreateAsuClient(MakeFactory(state));
    auto status = client->Init(kConfigPath);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    ASSERT_EQ(state->initConfigs[20].endpoints.size(), std::size_t{1});
    EXPECT_EQ(state->initConfigs[20].endpoints[0].ip, "192.168.1.20");
    EXPECT_EQ(state->initConfigs[20].endpoints[0].protocol, Protocol::ROCE);
    EXPECT_EQ(state->initConfigs[20].endpoints[0].deviceId, std::int32_t{6});
    for (auto asuId : {AsuId{10}, AsuId{20}}) {
        EXPECT_EQ(state->initConfigs[asuId].sendBufferSlotSize, std::size_t{8192});
        EXPECT_EQ(state->initConfigs[asuId].sendBufferSlotNum, std::size_t{2});
        EXPECT_EQ(state->initConfigs[asuId].flagBufferSlotSize, std::size_t{256});
        EXPECT_EQ(state->initConfigs[asuId].flagBufferSlotNum, std::size_t{32});
        EXPECT_EQ(state->initConfigs[asuId].asuBatchLoadIoNum, std::size_t{11});
        EXPECT_EQ(state->initConfigs[asuId].asuBatchStoreIoNum, std::size_t{12});
        EXPECT_EQ(state->initConfigs[asuId].asuDeleteIoNum, std::size_t{13});
        EXPECT_EQ(state->initConfigs[asuId].asuQueryIoNum, std::size_t{14});
    }
}

TEST(AsuClientImplTest, Routing_UsesRouterConfigFromClientConfigAttrs)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    config.attrs["hash_table.type"] = "CONTIGUOUS_BLOCK_AFFINITY";
    config.attrs["contiguous_block_affinity.block_count"] = "2";
    config.attrs["contiguous_block_affinity.full_spread_type"] = "RING_HASH";

    auto keyForAsu10 = FindKeyForAsu({10, 20}, 10);
    auto keyForAsu20 = FindKeyForAsu({10, 20}, 20);
    ASSERT_NE(keyForAsu10, CacheKey{});
    ASSERT_NE(keyForAsu20, CacheKey{});

    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(
        {
            KVBuffer{keyForAsu10, {}},
            KVBuffer{keyForAsu20, {}}
    },
        taskId);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->storeCalls, std::vector<AsuId>({10}));
}

TEST(AsuClientImplTest, ViewServer_InitFailsWhenViewReferencesMissingTransportConfig)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10, 20}
    },
        std::vector<std::uint64_t>{1});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));

    auto status = client->Init(config);

    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    EXPECT_NE(status.message.find("asuId=20"), std::string::npos);
}

TEST(AsuClientImplTest, Query_PerKeyKeepsOriginalOrder)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05"), MakeCacheKey("k15"), MakeCacheKey("k25")},
                                QueryOptions{}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({0, 1, 1}));
}

TEST(AsuClientImplTest, Query_PerKeyFailureIncludesAsuContext)
{
    auto state = std::make_shared<TestState>();
    state->queryFailures[20] = Status::Error(StatusCode::IO_ERROR, "fake per-key query failure");
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k15")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::IO_ERROR);
    EXPECT_NE(status.message.find("asuId=20"), std::string::npos);
    EXPECT_NE(status.message.find("key_count=1"), std::string::npos);
}

TEST(AsuClientImplTest, Query_PerKeyResultSizeMismatchReturnsInternalError)
{
    class ShortQueryTransport final : public FakeTransport {
    public:
        explicit ShortQueryTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status Query(const std::vector<CacheKey>&, const QueryOptions&,
                     QueryResult& result) override
        {
            result.exists.clear();
            result.prefixHitKeys = 0;
            return Status::OK();
        }
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        return std::unique_ptr<AsuTransport>(new ShortQueryTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("query result size mismatch"), std::string::npos);
    EXPECT_NE(status.message.find("asuId=10"), std::string::npos);
}

TEST(AsuClientImplTest, Query_PrefixBroadcastsAndMergesResults)
{
    auto state = std::make_shared<TestState>();
    state->prefixQueryResults[10] = QueryResult{
        {1, 0, 0},
        2
    };
    state->prefixQueryResults[20] = QueryResult{
        {0, 1, 0},
        3
    };
    state->prefixQueryResults[30] = QueryResult{
        {0, 0, 1},
        5
    };
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    QueryOptions options;
    options.mode = QueryMode::PREFIX;
    QueryResult result;
    auto status = client->Query(
        {MakeCacheKey("prefix-a"), MakeCacheKey("prefix-b"), MakeCacheKey("prefix-c")}, options,
        result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({1, 1, 1}));
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{10});
    ExpectSameAsuSet(state->queryCalls, {10, 20, 30});
    EXPECT_EQ(state->queryKeyCounts[10], std::size_t{3});
    EXPECT_EQ(state->queryKeyCounts[20], std::size_t{3});
    EXPECT_EQ(state->queryKeyCounts[30], std::size_t{3});
}

TEST(AsuClientImplTest, Query_PrefixPartialFailureIncludesAsuContext)
{
    auto state = std::make_shared<TestState>();
    state->prefixQueryResults[10] = QueryResult{
        {1, 0, 0},
        2
    };
    state->queryFailures[20] =
        Status::Error(StatusCode::INVALID_ARGUMENT, "fake prefix query failure");
    state->prefixQueryResults[30] = QueryResult{
        {0, 0, 1},
        5
    };
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    QueryOptions options;
    options.mode = QueryMode::PREFIX;
    QueryResult result;
    auto status = client->Query(
        {MakeCacheKey("prefix-a"), MakeCacheKey("prefix-b"), MakeCacheKey("prefix-c")}, options,
        result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_NE(status.message.find("asuId=20"), std::string::npos);
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({1, 0, 1}));
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{7});
    ExpectSameAsuSet(state->queryCalls, {10, 20, 30});
}

TEST(AsuClientImplTest, Query_PrefixResultSizeMismatchReturnsInternalError)
{
    auto state = std::make_shared<TestState>();
    state->prefixQueryResults[10] = QueryResult{{1}, 1};
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    QueryOptions options;
    options.mode = QueryMode::PREFIX;
    QueryResult result;
    auto status =
        client->Query({MakeCacheKey("prefix-a"), MakeCacheKey("prefix-b")}, options, result);

    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("prefix query result size mismatch"), std::string::npos);
    EXPECT_NE(status.message.find("asuId=10"), std::string::npos);
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryReturnsErrorWithoutRetry)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);

    status = client->Query({MakeCacheKey("k15")}, QueryOptions{}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({1}));
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryDoesNotRefreshNonRefreshableError)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    state->firstQueryFailureCode = StatusCode::INVALID_ARGUMENT;
    state->firstQueryFailureMessage = "fake invalid argument";
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto config = MakeConfig({10, 20});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(viewServer->FetchCount(), std::size_t{1});
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryRefreshesOnIoError)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    state->firstQueryFailureCode = StatusCode::IO_ERROR;
    state->firstQueryFailureMessage = "fake io error";
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::IO_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryRefreshesOnTimeout)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    state->firstQueryFailureCode = StatusCode::TIMEOUT;
    state->firstQueryFailureMessage = "fake timeout";
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::TIMEOUT);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryKeepsOriginalErrorWhenViewFetchFails)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    viewServer->FailFetchAt(2);
    auto config = MakeConfig({10, 20});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
}

TEST(AsuClientImplTest, BackgroundRefresh_LoadReturnsErrorWithoutRetry)
{
    auto state = std::make_shared<TestState>();
    state->failFirstLoad = true;
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->LoadAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(taskId, kInvalidTaskId);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->loadCalls.empty());
}

TEST(AsuClientImplTest, BackgroundRefresh_StoreReturnsErrorWithoutRetry)
{
    auto state = std::make_shared<TestState>();
    state->failFirstStore = true;
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(taskId, kInvalidTaskId);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->storeCalls.empty());
}

TEST(AsuClientImplTest, BackgroundRefresh_DeleteReturnsErrorWithoutRetry)
{
    auto state = std::make_shared<TestState>();
    state->failFirstDelete = true;
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = kInvalidTaskId;
    auto status = client->DeleteAsync({MakeCacheKey("k05")}, taskId);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(taskId, kInvalidTaskId);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->deleteCalls.empty());
}

TEST(AsuClientImplTest, ViewEpoch_DoesNotPublishSameOrOlderViewEpoch)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20},
            {10, 20}
    },
        std::vector<std::uint64_t>{5, 5, 4});
    auto config = MakeConfig({10, 20});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
}

TEST(AsuClientImplTest, SnapshotRefresh_BuildFailureKeepsOldSnapshot)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto config = MakeConfig({10});
    auto viewServer = std::make_shared<FakeViewServer>(std::vector<std::vector<AsuId>>{{10}, {20}},
                                                       std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));

    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
    EXPECT_EQ(state->queryCalls, std::vector<AsuId>({10}));
}

TEST(AsuClientImplTest, MemoryRegister_RegisterRegionsRegistersFirstTransportAndBindsFollowers)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}, MemoryRegion{}}, results);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->bindCalls, std::vector<AsuId>({20, 30}));
    ASSERT_EQ(results.size(), std::size_t{2});
    EXPECT_EQ(results[0].handle, MRHandle{500});
    EXPECT_EQ(results[1].handle, MRHandle{501});
    EXPECT_EQ(results[0].tokenId, std::uint32_t{900});
    EXPECT_EQ(results[1].tokenId, std::uint32_t{901});
    ASSERT_EQ(state->boundRegions[20].size(), std::size_t{2});
    EXPECT_EQ(state->boundRegions[20][0].handle, MRHandle{500});
    EXPECT_EQ(state->boundRegions[20][0].tokenId, std::uint32_t{900});
    EXPECT_EQ(state->boundRegions[20][1].handle, MRHandle{501});
    EXPECT_EQ(state->boundRegions[20][1].tokenId, std::uint32_t{901});
    ASSERT_EQ(state->boundRegions[30].size(), std::size_t{2});
    EXPECT_EQ(state->boundRegions[30][0].tokenId, std::uint32_t{900});
}

TEST(AsuClientImplTest, MemoryRegister_PartialRegisterFailureDoesNotBindFollowers)
{
    class PartialRegisterTransport final : public FakeTransport {
    public:
        explicit PartialRegisterTransport(std::shared_ptr<TestState> state)
            : FakeTransport(state), state_(std::move(state))
        {
        }

        Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                               std::vector<RegisterResult>& results) override
        {
            state_->registerCalls.emplace_back(10);
            results.clear();
            results.reserve(regions.size());
            if (!regions.empty()) {
                results.emplace_back(RegisterResult{Status::OK(), 500, 0, 0, 900});
            }
            if (regions.size() > 1) {
                results.emplace_back(RegisterResult{
                    Status::Error(StatusCode::INTERNAL_ERROR, "fake region register failure"),
                    kInvalidMRHandle});
            }
            return Status::Error(StatusCode::PARTIAL_FAILED,
                                 "one or more memory regions failed to register");
        }

    private:
        std::shared_ptr<TestState> state_;
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        if (state->createdTransports == 1) {
            return std::unique_ptr<AsuTransport>(new PartialRegisterTransport(state));
        }
        return std::unique_ptr<AsuTransport>(new FakeTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}, MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_NE(status.message.find("asuIndex=0"), std::string::npos);
    EXPECT_NE(status.message.find("asuId=10"), std::string::npos);
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_TRUE(state->bindCalls.empty());
    ASSERT_EQ(results.size(), std::size_t{2});
    EXPECT_TRUE(results[0].status.ok()) << results[0].status.message;
    EXPECT_EQ(results[1].status.code, StatusCode::INTERNAL_ERROR);
}

TEST(AsuClientImplTest, MemoryRegister_FirstRegisterFailureIncludesAsuContext)
{
    class FailingRegisterTransport final : public FakeTransport {
    public:
        explicit FailingRegisterTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status RegisterRegions(const std::vector<MemoryRegion>&,
                               std::vector<RegisterResult>&) override
        {
            return Status::Error(StatusCode::BUFFER_NOT_REGISTERED, "fake register failure");
        }
    };

    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        [state] {
            ++state->createdTransports;
            return std::unique_ptr<AsuTransport>(new FailingRegisterTransport(state));
        },
        MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::BUFFER_NOT_REGISTERED);
    EXPECT_NE(status.message.find("asuIndex=0"), std::string::npos);
    EXPECT_NE(status.message.find("asuId=10"), std::string::npos);
    EXPECT_NE(status.message.find("region_count=1"), std::string::npos);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, MemoryRegister_SuccessWithMismatchedResultCountReturnsInternalError)
{
    class MismatchedRegisterTransport final : public FakeTransport {
    public:
        explicit MismatchedRegisterTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status RegisterRegions(const std::vector<MemoryRegion>&,
                               std::vector<RegisterResult>& results) override
        {
            results.clear();
            return Status::OK();
        }
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        return std::unique_ptr<AsuTransport>(new MismatchedRegisterTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::INTERNAL_ERROR);
    EXPECT_NE(status.message.find("asuIndex=0"), std::string::npos);
    EXPECT_NE(status.message.find("region_count=1"), std::string::npos);
    EXPECT_NE(status.message.find("result_count=0"), std::string::npos);
    EXPECT_TRUE(state->bindCalls.empty());
}

TEST(AsuClientImplTest, MemoryRegister_BindFailureIncludesAsuContext)
{
    class FailingBindTransport final : public FakeTransport {
    public:
        explicit FailingBindTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status BindRegisteredRegions(const std::vector<RegisteredMemory>&,
                                     std::vector<RegisterResult>&) override
        {
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake bind failure");
        }
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        if (state->createdTransports == 1) {
            return std::unique_ptr<AsuTransport>(new FakeTransport(state));
        }
        return std::unique_ptr<AsuTransport>(new FailingBindTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_NE(status.message.find("asuIndex=1"), std::string::npos);
    EXPECT_NE(status.message.find("asuId=20"), std::string::npos);
    EXPECT_NE(status.message.find("region_count=1"), std::string::npos);
}

TEST(AsuClientImplTest, MemoryRegister_SuccessfulBindWithMismatchedResultCountFails)
{
    class MismatchedBindTransport final : public FakeTransport {
    public:
        explicit MismatchedBindTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status BindRegisteredRegions(const std::vector<RegisteredMemory>&,
                                     std::vector<RegisterResult>& results) override
        {
            results.clear();
            return Status::OK();
        }
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        if (state->createdTransports == 2) {
            return std::unique_ptr<AsuTransport>(new MismatchedBindTransport(state));
        }
        return std::unique_ptr<AsuTransport>(new FakeTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_NE(status.message.find("asuIndex=1"), std::string::npos);
    EXPECT_NE(status.message.find("region_count=1"), std::string::npos);
    EXPECT_NE(status.message.find("result_count=0"), std::string::npos);
}

TEST(AsuClientImplTest, MemoryRegister_BindFailureDoesNotCacheResource)
{
    class FailingBindTransport final : public FakeTransport {
    public:
        explicit FailingBindTransport(std::shared_ptr<TestState> state)
            : FakeTransport(state), state_(std::move(state))
        {
        }

        Status BindRegisteredRegions(const std::vector<RegisteredMemory>&,
                                     std::vector<RegisterResult>&) override
        {
            state_->bindCalls.emplace_back(20);
            return Status::Error(StatusCode::CONNECTION_ERROR, "fake bind failure");
        }

    private:
        std::shared_ptr<TestState> state_;
    };

    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20, 30});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10, 20},
            {10, 20, 30}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        [state] {
            ++state->createdTransports;
            if (state->createdTransports == 2) {
                return std::unique_ptr<AsuTransport>(new FailingBindTransport(state));
            }
            return std::unique_ptr<AsuTransport>(new FakeTransport(state));
        },
        MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_EQ(status.code, StatusCode::PARTIAL_FAILED);

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, queryResult);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{3});
    EXPECT_EQ(state->bindCalls, std::vector<AsuId>({20}));
}

TEST(AsuClientImplTest, MemoryRegister_UnregisterFailureIncludesAsuContext)
{
    class FailingUnregisterTransport final : public FakeTransport {
    public:
        explicit FailingUnregisterTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status UnregisterRegions(const std::vector<MRHandle>&) override
        {
            return Status::Error(StatusCode::IO_ERROR, "fake unregister failure");
        }
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        return std::unique_ptr<AsuTransport>(new FailingUnregisterTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    auto status = client->UnregisterRegions({7});

    EXPECT_EQ(status.code, StatusCode::IO_ERROR);
    EXPECT_NE(status.message.find("asuId=10"), std::string::npos);
    EXPECT_NE(status.message.find("handle_count=1"), std::string::npos);
}

TEST(AsuClientImplTest, MemoryRegister_UnregisterRemovesCachedResourceBeforeFutureAsuIsAdded)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(results.size(), std::size_t{1});

    status = client->UnregisterRegions({results[0].handle});
    ASSERT_TRUE(status.ok()) << status.message;

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, queryResult);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->bindCalls.empty());
}

TEST(AsuClientImplTest, Task_CheckRemovesTaskAfterCompletion)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);
    ASSERT_TRUE(status.ok()) << status.message;

    status = client->Check(taskId, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Task_CheckKeepsInProgressTaskUntilCompletion)
{
    auto state = std::make_shared<TestState>();
    state->checkResultStatus[10] = Status::Error(StatusCode::IN_PROGRESS, "fake in progress");
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);
    EXPECT_EQ(status.code, StatusCode::IN_PROGRESS);

    state->checkResultStatus.erase(10);
    status = client->Check(taskId, result);
    ASSERT_TRUE(status.ok()) << status.message;

    status = client->Check(taskId, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Task_CheckRefreshesViewOnRefreshableChildFailure)
{
    auto state = std::make_shared<TestState>();
    state->checkResultStatus[10] = Status::Error(StatusCode::IO_ERROR, "fake child io error");
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, Task_PartialDispatchFailureCancelsDispatchedSubtasks)
{
    auto state = std::make_shared<TestState>();
    state->failStoreAfterFirstDispatch = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());
    auto keyForAsu10 = FindKeyForAsu({10, 20}, 10);
    auto keyForAsu20 = FindKeyForAsu({10, 20}, 20);
    ASSERT_NE(keyForAsu10, CacheKey{});
    ASSERT_NE(keyForAsu20, CacheKey{});

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(
        {
            KVBuffer{keyForAsu10, {}},
            KVBuffer{keyForAsu20, {}}
    },
        taskId);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    EXPECT_EQ(taskId, kInvalidTaskId);
    ASSERT_EQ(state->storeCalls.size(), std::size_t{1});
    ASSERT_EQ(state->cancelCalls.size(), std::size_t{1});
    EXPECT_EQ(state->cancelCalls[0], state->storeCalls[0]);
}

TEST(AsuClientImplTest, Task_WaitRemovesTaskAfterCompletion)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Wait(taskId, 10, result);
    ASSERT_TRUE(status.ok()) << status.message;

    status = client->Wait(taskId, 10, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Task_WaitTimeoutKeepsTaskForLaterCompletion)
{
    class TimeoutOnceTransport final : public FakeTransport {
    public:
        explicit TimeoutOnceTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status Wait(TaskId, std::uint64_t, TaskResult& result) override
        {
            if (!timedOut_) {
                timedOut_ = true;
                result.status = Status::Error(StatusCode::TIMEOUT, "fake wait timeout");
                return result.status;
            }
            return FakeTransport::Check(1000, result);
        }

    private:
        bool timedOut_{false};
    };

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient([state] {
        ++state->createdTransports;
        return std::unique_ptr<AsuTransport>(new TimeoutOnceTransport(state));
    });
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Wait(taskId, 10, result);
    EXPECT_EQ(status.code, StatusCode::TIMEOUT);

    status = client->Wait(taskId, 10, result);
    EXPECT_TRUE(status.ok()) << status.message;

    status = client->Check(taskId, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Task_CheckKeepsEntryStatusInOriginalOrderAcrossAsus)
{
    auto state = std::make_shared<TestState>();
    state->checkEntryStatus[10] = {Status::OK()};
    state->checkEntryStatus[20] = {Status::Error(StatusCode::IO_ERROR, "entry on asu 20")};
    state->checkEntryStatus[30] = {Status::Error(StatusCode::NOT_FOUND, "entry on asu 30")};
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());
    auto entries = BuildRoutedEntries({30, 10, 20});
    ASSERT_EQ(entries.size(), std::size_t{3});
    ASSERT_NE(entries[0].key, CacheKey{});
    ASSERT_NE(entries[1].key, CacheKey{});
    ASSERT_NE(entries[2].key, CacheKey{});

    TaskId taskId = 0;
    auto status = client->StoreAsync(entries, taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);

    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.entryStatus.size(), std::size_t{3});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::NOT_FOUND);
    EXPECT_EQ(result.entryStatus[1].code, StatusCode::OK);
    EXPECT_EQ(result.entryStatus[2].code, StatusCode::IO_ERROR);
}

TEST(AsuClientImplTest, Task_LoadKeepsEntryStatusInOriginalOrderAcrossAsus)
{
    auto state = std::make_shared<TestState>();
    state->checkEntryStatus[10] = {Status::OK()};
    state->checkEntryStatus[20] = {Status::Error(StatusCode::IO_ERROR, "load entry on asu 20")};
    state->checkEntryStatus[30] = {Status::Error(StatusCode::NOT_FOUND, "load entry on asu 30")};
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());
    auto entries = BuildRoutedEntries({30, 10, 20});
    ASSERT_EQ(entries.size(), std::size_t{3});
    ASSERT_NE(entries[0].key, CacheKey{});
    ASSERT_NE(entries[1].key, CacheKey{});
    ASSERT_NE(entries[2].key, CacheKey{});

    TaskId taskId = kInvalidTaskId;
    auto status = client->LoadAsync(entries, taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);

    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.entryStatus.size(), std::size_t{3});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::NOT_FOUND);
    EXPECT_EQ(result.entryStatus[1].code, StatusCode::OK);
    EXPECT_EQ(result.entryStatus[2].code, StatusCode::IO_ERROR);
}

TEST(AsuClientImplTest, Task_DeleteKeepsEntryStatusInOriginalOrderAcrossAsus)
{
    auto state = std::make_shared<TestState>();
    state->checkEntryStatus[10] = {Status::OK()};
    state->checkEntryStatus[20] = {Status::Error(StatusCode::IO_ERROR, "delete entry on asu 20")};
    state->checkEntryStatus[30] = {Status::Error(StatusCode::NOT_FOUND, "delete entry on asu 30")};
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());
    auto entries = BuildRoutedEntries({30, 10, 20});
    ASSERT_EQ(entries.size(), std::size_t{3});
    ASSERT_NE(entries[0].key, CacheKey{});
    ASSERT_NE(entries[1].key, CacheKey{});
    ASSERT_NE(entries[2].key, CacheKey{});
    std::vector<CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) { keys.emplace_back(entry.key); }

    TaskId taskId = kInvalidTaskId;
    auto status = client->DeleteAsync(keys, taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    status = client->Check(taskId, result);

    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.entryStatus.size(), std::size_t{3});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::NOT_FOUND);
    EXPECT_EQ(result.entryStatus[1].code, StatusCode::OK);
    EXPECT_EQ(result.entryStatus[2].code, StatusCode::IO_ERROR);
}

TEST(AsuClientImplTest, SnapshotRefresh_ReusesExistingTransportAndBindsResourcesToAddedAsu)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(std::vector<std::vector<AsuId>>{
        {10},
        {10, 20}
    });
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisterResult> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_TRUE(status.ok()) << status.message;
    state->failFirstQuery = true;

    QueryResult result;
    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, result);

    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_EQ(state->bindCalls, std::vector<AsuId>({20}));
    ASSERT_EQ(state->boundRegions[20].size(), std::size_t{1});
    EXPECT_EQ(state->boundRegions[20][0].handle, MRHandle{500});
    EXPECT_EQ(state->boundRegions[20][0].tokenId, std::uint32_t{900});
}

TEST(AsuClientImplTest,
     SnapshotRefresh_RemovedAsuStopsReceivingNewRequestsButExistingTaskCanComplete)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10, 20},
            {10}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k15"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(state->storeCalls, std::vector<AsuId>({20}));

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = client->Query({MakeCacheKey("k05")}, QueryOptions{}, queryResult);
    ASSERT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    TaskResult taskResult;
    status = client->Check(taskId, taskResult);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->checkCalls, std::vector<AsuId>({20}));

    status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k15"), {}}
    },
        taskId);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->storeCalls, std::vector<AsuId>({20, 10}));
}

}  // namespace UC::ASU
