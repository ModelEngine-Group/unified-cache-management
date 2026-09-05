#include "kv_client_impl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
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
#include "utils/config_utils.h"
#include "router/router.h"

namespace kv::test {

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

MRHandle MakeTestMrHandle(std::uintptr_t value) { return static_cast<MRHandle>(value); }

struct TestState {
    std::uint32_t createdTransports{0};
    std::uint32_t createdProviders{0};
    std::unordered_map<AsuId, TransportConfig> initConfigs;
    std::unordered_map<AsuId, TransProvider*> initProviders;
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
    bool deferCompletionCallbacks{false};
    bool blockStoreDispatch{false};
    bool storeDispatchStarted{false};
    bool releaseStoreDispatch{false};
    std::mutex storeDispatchMu;
    std::condition_variable storeDispatchCv;
    std::mutex completionMu;
    std::condition_variable completionCv;
    std::unordered_map<AsuId, Status> queryFailures;
    std::unordered_map<AsuId, Status> loadFailures;
    std::unordered_map<AsuId, Status> storeFailures;
    std::unordered_map<AsuId, Status> deleteFailures;
    std::unordered_map<AsuId, std::vector<Status>> checkEntryStatus;
    std::unordered_map<AsuId, Status> checkResultStatus;
    std::vector<AsuId> registerCalls;
    std::vector<AsuId> providerBindCalls;
    std::vector<KVBuffer> submittedEntries;
    std::vector<AsuId> unregisterCalls;
    bool failRegister{false};
    bool failProviderBind{false};
    bool returnPartialRegister{false};
    bool mismatchRegisterResultCount{false};
    bool failUnregister{false};
    std::vector<AsuId> queryCalls;
    std::unordered_map<AsuId, std::size_t> queryKeyCounts;
    std::unordered_map<AsuId, std::vector<CacheKey>> queryKeys;
    std::vector<AsuId> loadCalls;
    std::vector<AsuId> storeCalls;
    std::vector<AsuOpType> submittedOpTypes;
    std::vector<AsuId> deleteCalls;
    std::vector<AsuId> cancelCalls;
    std::unordered_map<AsuId, TaskId> childTaskIds;
    std::unordered_map<AsuId, TaskCompletionCallback> pendingCompletionCallbacks;
    std::size_t completionCallbackCount{0};
};

namespace {

class ClientTestTransProvider final : public TransProvider {
public:
    ClientTestTransProvider(AsuId asuId, std::shared_ptr<TestState> state)
        : asuId_(asuId), state_(std::move(state))
    {
    }

    Status CreateConnection(const std::string&, const std::string&, uint32_t, uint32_t, uint32_t,
                            std::vector<ConnectionHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> DeleteConnections(const std::vector<ConnectionHandle>& handles) override
    {
        return std::vector<Status>(handles.size(), Status::OK());
    }

    std::vector<Status> Send(const std::vector<SendIoBatch>& ioBatches, uint32_t, uint32_t) override
    {
        return std::vector<Status>(ioBatches.size(), Status::OK());
    }

    Status RegisterMemory(const std::vector<RegisterMemoryDesc>& memoryDescs,
                          std::vector<MRHandle>& mrHandles) override
    {
        state_->registerCalls.emplace_back(asuId_);
        mrHandles.clear();
        if (state_->failRegister) {
            return Status::Error(StatusCode::BUFFER_NOT_REGISTERED, "fake register failure");
        }
        if (state_->returnPartialRegister) {
            if (!memoryDescs.empty()) { mrHandles.emplace_back(MakeTestMrHandle(500)); }
            return Status::Error(StatusCode::PARTIAL_FAILED,
                                 "one or more memory regions failed to register");
        }
        auto resultCount = memoryDescs.size();
        if (state_->mismatchRegisterResultCount && resultCount > 0) { --resultCount; }
        for (std::size_t index = 0; index < resultCount; ++index) {
            mrHandles.emplace_back(MakeTestMrHandle(500 + index));
        }
        return Status::OK();
    }

    Status BindMemory(const std::vector<BindMemoryDesc>& memoryDescs,
                      std::vector<MRHandle>& mrHandles) override
    {
        state_->providerBindCalls.emplace_back(asuId_);
        mrHandles.clear();
        if (state_->failProviderBind) {
            return Status::Error(StatusCode::BUFFER_NOT_REGISTERED, "fake provider bind failure");
        }
        for (std::size_t index = 0; index < memoryDescs.size(); ++index) {
            mrHandles.emplace_back(MakeTestMrHandle(500 + index));
        }
        return Status::OK();
    }

    std::vector<Status> UnregisterMemory(
        const std::vector<UnregisterMemoryDesc>& memoryDescs) override
    {
        state_->unregisterCalls.emplace_back(asuId_);
        const auto status = state_->failUnregister
                                ? Status::Error(StatusCode::IO_ERROR, "fake unregister failure")
                                : Status::OK();
        return std::vector<Status>(memoryDescs.size(), status);
    }

    Status AllocThread(uint32_t, const std::vector<uint32_t>&, std::vector<ThreadHandle>&) override
    {
        return Status::OK();
    }

    std::vector<Status> FreeThread(const std::vector<ThreadHandle>& threads) override
    {
        return std::vector<Status>(threads.size(), Status::OK());
    }

    Status GetMemTokenId(MRHandle mrHandle, uint32_t& tokenId) override
    {
        tokenId = 900 + static_cast<std::uint32_t>(mrHandle - MakeTestMrHandle(500));
        return Status::OK();
    }

private:
    AsuId asuId_;
    std::shared_ptr<TestState> state_;
};

}  // namespace

class FakeTransport : public AsuTransport {
public:
    explicit FakeTransport(std::shared_ptr<TestState> state) : state_(std::move(state)) {}

    Status Init(const TransportConfig& config,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)transProvider;
        config_ = config;
        state_->initConfigs[config_.asuId] = config_;
        state_->initProviders[config_.asuId] = transProvider.get();
        initialized_ = true;
        return Status::OK();
    }

    Status Init(const std::string& configPath,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)configPath;
        (void)transProvider;
        return Status::Error(StatusCode::UNSUPPORTED, "fake transport config path is unsupported");
    }

    Status Shutdown() override
    {
        initialized_ = false;
        return Status::OK();
    }

    Status CheckHealth() override { return initialized_ ? Status::OK() : NotInitialized(); }

    virtual Status RunQuery(BatchView<CacheKey> keys, QueryResult& result)
    {
        if (!initialized_) { return NotInitialized(); }
        if (state_->failFirstQuery && !state_->firstQueryFailed) {
            state_->firstQueryFailed = true;
            return Status::Error(state_->firstQueryFailureCode, state_->firstQueryFailureMessage);
        }

        state_->queryCalls.emplace_back(config_.asuId);
        state_->queryKeyCounts[config_.asuId] += keys.size;
        auto& routedKeys = state_->queryKeys[config_.asuId];
        routedKeys.reserve(routedKeys.size() + keys.size);
        for (std::size_t index = 0; index < keys.size; ++index) {
            routedKeys.emplace_back(keys[index]);
        }
        auto failureIter = state_->queryFailures.find(config_.asuId);
        if (failureIter != state_->queryFailures.end()) { return failureIter->second; }

        result.exists.clear();
        result.exists.reserve(keys.size);
        for (std::size_t index = 0; index < keys.size; ++index) {
            result.exists.emplace_back(keys[index] == MakeCacheKey("k15") ||
                                       keys[index] == MakeCacheKey("k25"));
        }
        result.prefixHitKeys = 0;
        return Status::OK();
    }

    Status Submit(const TransportTaskPtr& task) override
    {
        if (!task) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "fake transport task is null");
        }
        state_->submittedEntries = task->entries;
        state_->submittedOpTypes.emplace_back(task->opType);
        switch (task->opType) {
            case AsuOpType::QUERY: {
                QueryResult queryResult;
                auto status = RunQuery({task->keys.data(), task->keys.size()}, queryResult);
                if (!status.ok()) {
                    task->taskId = kInvalidTaskId;
                    return status;
                }

                task->taskId = 4000 + config_.asuId;
                TaskResult taskResult;
                auto statusIter = state_->checkResultStatus.find(config_.asuId);
                taskResult.status = statusIter == state_->checkResultStatus.end()
                                        ? Status::OK()
                                        : statusIter->second;
                taskResult.entryStatus.assign(task->keys.size(), taskResult.status);
                taskResult.queryResult = std::move(queryResult);
                Complete(std::move(taskResult), std::move(task->onComplete));
                return Status::OK();
            }
            case AsuOpType::LOAD:
            case AsuOpType::BATCH_LOAD: {
                if (state_->failFirstLoad && !state_->firstLoadFailed) {
                    state_->firstLoadFailed = true;
                    return Status::Error(StatusCode::CONNECTION_ERROR,
                                         "fake load connection error");
                }
                auto failureIter = state_->loadFailures.find(config_.asuId);
                if (failureIter != state_->loadFailures.end()) { return failureIter->second; }

                state_->loadCalls.emplace_back(config_.asuId);
                task->taskId = 1000 + config_.asuId;
                state_->childTaskIds[config_.asuId] = task->taskId;
                Complete(task->entries.size(), std::move(task->onComplete));
                return Status::OK();
            }
            case AsuOpType::STORE:
            case AsuOpType::BATCH_STORE: {
                if (state_->blockStoreDispatch) {
                    std::unique_lock<std::mutex> lock{state_->storeDispatchMu};
                    state_->storeDispatchStarted = true;
                    state_->storeDispatchCv.notify_all();
                    state_->storeDispatchCv.wait(lock,
                                                 [this] { return state_->releaseStoreDispatch; });
                }
                if (state_->failFirstStore && !state_->firstStoreFailed) {
                    state_->firstStoreFailed = true;
                    return Status::Error(StatusCode::CONNECTION_ERROR,
                                         "fake store connection error");
                }
                auto failureIter = state_->storeFailures.find(config_.asuId);
                if (failureIter != state_->storeFailures.end()) { return failureIter->second; }
                if (state_->failStoreAfterFirstDispatch && ++state_->storeDispatchAttempts > 1) {
                    return Status::Error(StatusCode::CONNECTION_ERROR,
                                         "fake partial dispatch failure");
                }

                state_->storeCalls.emplace_back(config_.asuId);
                task->taskId = 2000 + config_.asuId;
                state_->childTaskIds[config_.asuId] = task->taskId;
                Complete(task->entries.size(), std::move(task->onComplete));
                return Status::OK();
            }
            case AsuOpType::DELETE: {
                if (state_->failFirstDelete && !state_->firstDeleteFailed) {
                    state_->firstDeleteFailed = true;
                    return Status::Error(StatusCode::CONNECTION_ERROR,
                                         "fake delete connection error");
                }
                auto failureIter = state_->deleteFailures.find(config_.asuId);
                if (failureIter != state_->deleteFailures.end()) { return failureIter->second; }

                state_->deleteCalls.emplace_back(config_.asuId);
                task->taskId = 3000 + config_.asuId;
                state_->childTaskIds[config_.asuId] = task->taskId;
                Complete(task->keys.size(), std::move(task->onComplete));
                return Status::OK();
            }
            default:
                task->taskId = kInvalidTaskId;
                return Status::Error(StatusCode::INVALID_ARGUMENT,
                                     "unsupported fake transport operation");
        }
    }

    Status Cancel(TaskId) override
    {
        state_->cancelCalls.emplace_back(config_.asuId);
        return Status::OK();
    }

private:
    void Complete(std::size_t entryCount, TaskCompletionCallback onComplete)
    {
        TaskResult result;
        auto statusIter = state_->checkResultStatus.find(config_.asuId);
        result.status =
            statusIter == state_->checkResultStatus.end() ? Status::OK() : statusIter->second;
        auto entryIter = state_->checkEntryStatus.find(config_.asuId);
        result.entryStatus = entryIter == state_->checkEntryStatus.end()
                                 ? std::vector<Status>(entryCount, result.status)
                                 : entryIter->second;
        Complete(std::move(result), std::move(onComplete));
    }

    void Complete(TaskResult result, TaskCompletionCallback onComplete)
    {
        if (!onComplete) { return; }
        if (state_->deferCompletionCallbacks) {
            {
                std::lock_guard<std::mutex> lock{state_->completionMu};
                state_->pendingCompletionCallbacks[config_.asuId] = std::move(onComplete);
            }
            state_->completionCv.notify_all();
            return;
        }

        ++state_->completionCallbackCount;
        onComplete(std::move(result));
    }

    static Status NotInitialized()
    {
        return Status::Error(StatusCode::NOT_INITIALIZED, "fake transport is not initialized");
    }

    std::shared_ptr<TestState> state_;
    TransportConfig config_;
    bool initialized_{false};
};

bool InvokePendingCompletion(const std::shared_ptr<TestState>& state, AsuId asuId,
                             TaskResult result)
{
    TaskCompletionCallback callback;
    {
        std::lock_guard<std::mutex> lock{state->completionMu};
        auto callbackIter = state->pendingCompletionCallbacks.find(asuId);
        if (callbackIter == state->pendingCompletionCallbacks.end()) { return false; }
        callback = std::move(callbackIter->second);
        state->pendingCompletionCallbacks.erase(callbackIter);
    }
    ++state->completionCallbackCount;
    callback(std::move(result));
    return true;
}

bool WaitForPendingCompletion(const std::shared_ptr<TestState>& state, AsuId asuId)
{
    std::unique_lock<std::mutex> lock{state->completionMu};
    return state->completionCv.wait_for(lock, std::chrono::milliseconds(100), [&] {
        return state->pendingCompletionCallbacks.find(asuId) !=
               state->pendingCompletionCallbacks.end();
    });
}

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
        if (failFetchFrom_ != 0 && fetchCount_ + 1 >= failFetchFrom_) {
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

    void FailFetchFrom(std::size_t fetchCount)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        failFetchFrom_ = fetchCount;
    }
    std::size_t FetchCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return fetchCount_;
    }

private:
    mutable std::mutex mutex_;
    std::size_t fetchCount_{0};
    std::size_t failFetchFrom_{0};
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

bool CheckUntilComplete(AsuClient& client, TaskId taskId)
{
    for (std::uint32_t attempt = 0; attempt < 100; ++attempt) {
        if (client.Check(taskId)) { return true; }
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
        transportConfig.providerType = TransProviderType::FAKE;
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

TransProviderFactory MakeProviderFactory(const std::shared_ptr<TestState>& state)
{
    return [state](const TransportConfig& config, std::shared_ptr<TransProvider>& transProvider) {
        ++state->createdProviders;
        transProvider = std::make_shared<ClientTestTransProvider>(config.asuId, state);
        return Status::OK();
    };
}

void ExpectSameAsuSet(std::vector<AsuId> actual, std::vector<AsuId> expected)
{
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(actual, expected);
}

Status QueryAndWait(AsuClient& client, const std::vector<CacheKey>& keys, QueryResult& result,
                    std::uint64_t timeoutMs = 0)
{
    TaskId taskId{kInvalidTaskId};
    auto status = client.QueryAsync(keys, taskId);
    if (!status.ok()) { return status; }

    TaskResult taskResult;
    status = client.Wait(taskId, timeoutMs, taskResult);
    if (taskResult.queryResult.has_value()) {
        result = std::move(*taskResult.queryResult);
    } else if (status.ok()) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "client query result is missing");
    }
    return status;
}

CacheKey FindKeyForAsu(const std::vector<AsuId>& asuIds, AsuId targetAsuId)
{
    std::vector<kv::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    auto router =
        kv::CreateRouter(nodeIds, kv::HashFunction{}, kv::RouterConfig{});
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    TaskId taskId = kInvalidTaskId;
    status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    EXPECT_TRUE(client->Check(1));
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

TEST(AsuClientImplTest, Provider_IndependentModeCreatesMemoryProviderAndOnePerTransport)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));

    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    EXPECT_EQ(state->createdProviders, std::uint32_t{4});
    EXPECT_NE(state->initProviders[10], state->initProviders[20]);
    EXPECT_NE(state->initProviders[20], state->initProviders[30]);
}

TEST(AsuClientImplTest, Provider_SharedModeUsesOneProviderForAllTransports)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20, 30});
    config.sharedProviderMode = SharedProviderMode::SHARED;
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));

    ASSERT_TRUE(client->Init(config).ok());

    EXPECT_EQ(state->createdProviders, std::uint32_t{1});
    EXPECT_EQ(state->initProviders[10], state->initProviders[20]);
    EXPECT_EQ(state->initProviders[20], state->initProviders[30]);
}

TEST(AsuClientImplTest, Provider_InvalidSharedModeIsRejected)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10});
    config.sharedProviderMode = static_cast<SharedProviderMode>(2);
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));

    const auto status = client->Init(config);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(state->createdProviders, std::uint32_t{0});
    EXPECT_EQ(state->createdTransports, std::uint32_t{0});
}

TEST(AsuClientImplTest, Provider_SharedModeReusesProviderForAddedTransport)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    config.sharedProviderMode = SharedProviderMode::SHARED;
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    state->failFirstQuery = true;
    QueryResult result;
    EXPECT_EQ(QueryAndWait(*client, {MakeCacheKey("k05")}, result).code,
              StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    ASSERT_TRUE(client->Shutdown().ok());

    EXPECT_EQ(state->createdProviders, std::uint32_t{1});
    EXPECT_EQ(state->initProviders[10], state->initProviders[20]);
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
    status = QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult);
    EXPECT_EQ(status.code, StatusCode::NOT_INITIALIZED);

    EXPECT_TRUE(client->Check(taskId));
}

TEST(AsuClientImplTest, Lifecycle_ShutdownDrainsFullTaskQueue)
{
    auto state = std::make_shared<TestState>();
    state->blockStoreDispatch = true;
    auto client = CreateAsuClient(MakeFactory(state));
    auto config = MakeConfig({10});
    config.maxInflightTasks = 2;
    ASSERT_TRUE(client->Init(config).ok());

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(client
                    ->StoreAsync(
                        {
                            KVBuffer{MakeCacheKey("active"), {}}
    },
                        taskId)
                    .ok());

    {
        std::unique_lock<std::mutex> lock{state->storeDispatchMu};
        EXPECT_TRUE(state->storeDispatchCv.wait_for(lock, std::chrono::milliseconds(100),
                                                    [&] { return state->storeDispatchStarted; }));
    }

    for (std::size_t index = 0; index < 2; ++index) {
        EXPECT_TRUE(client
                        ->StoreAsync(
                            {
                                KVBuffer{MakeCacheKey("queued-" + std::to_string(index)), {}}
        },
                            taskId)
                        .ok());
    }

    auto shutdown = std::async(std::launch::async, [&] { return client->Shutdown(); });
    EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);

    {
        std::lock_guard<std::mutex> lock{state->storeDispatchMu};
        state->releaseStoreDispatch = true;
    }
    state->storeDispatchCv.notify_all();

    ASSERT_EQ(shutdown.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
    EXPECT_TRUE(shutdown.get().ok());
    EXPECT_EQ(state->storeCalls.size(), std::size_t{3});
}

TEST(AsuClientImplTest, Input_EmptyQueryReturnsEmptyResult)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    QueryResult result;
    auto status = QueryAndWait(*client, {}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.exists.empty());
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{0});
    EXPECT_TRUE(state->queryCalls.empty());
}

TEST(AsuClientImplTest, Dispatch_UsesSingleProtocolForSingleEntryOperations)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(client
                    ->StoreAsync(
                        {
                            KVBuffer{MakeCacheKey("store-single"), {}}
    },
                        taskId)
                    .ok());
    TaskResult result;
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());

    ASSERT_TRUE(client
                    ->LoadAsync(
                        {
                            KVBuffer{MakeCacheKey("load-single"), {}}
    },
                        taskId)
                    .ok());
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());

    ASSERT_TRUE(client
                    ->BatchStoreAsync(
                        {
                            KVBuffer{MakeCacheKey("store-batch-0"), {}},
                            KVBuffer{MakeCacheKey("store-batch-1"), {}}
    },
                        taskId)
                    .ok());
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());

    ASSERT_TRUE(client
                    ->BatchLoadAsync(
                        {
                            KVBuffer{MakeCacheKey("load-batch-0"), {}},
                            KVBuffer{MakeCacheKey("load-batch-1"), {}}
    },
                        taskId)
                    .ok());
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());

    EXPECT_EQ(state->submittedOpTypes,
              std::vector<AsuOpType>({AsuOpType::STORE, AsuOpType::LOAD, AsuOpType::BATCH_STORE,
                                      AsuOpType::BATCH_LOAD}));
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

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.entryStatus.empty());
    EXPECT_TRUE(state->storeCalls.empty());
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

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(result.entryStatus.empty());
    EXPECT_TRUE(state->deleteCalls.empty());
}

TEST(AsuClientImplTest, Input_EmptyRegisterReturnsEmptyResults)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({}, results);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(results.empty());
    EXPECT_TRUE(state->registerCalls.empty());
}

TEST(AsuClientImplTest, Lifecycle_PublicInitLoadsClientConfigFile)
{
    constexpr const char* kConfigPath = "kv_client_impl_client_config_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "clientId=file-init-test\n";
        configFile << "asu_shared_provider=1\n";
        configFile << "transport.asuIds=10,20\n";
        configFile << "transport.send_buffer_slot_size=8192\n";
        configFile << "transport.send_buffer_slot_num=2\n";
        configFile << "transport.flag_buffer_slot_size=256\n";
        configFile << "transport.flag_buffer_slot_num=32\n";
        configFile << "transport.batch_load_io_num=11\n";
        configFile << "transport.batch_store_io_num=12\n";
        configFile << "transport.delete_io_num=13\n";
        configFile << "transport.query_io_num=14\n";
        configFile << "transport.device_id=6\n";
        configFile << "transport.provider_type=fake\n";
        configFile << "transport.max_error_count=5\n";
        configFile << "transport.completion_poll_spin_limit=23\n";
        configFile << "asuInfo.20=protocol=roce,placement=device,port=6000,"
                   << "local.comm_id=192.168.1.20\n";
    }

    auto state = std::make_shared<TestState>();
    std::unique_ptr<AsuClient> client =
        CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    auto status = client->Init(kConfigPath);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_EQ(state->createdProviders, std::uint32_t{1});
    EXPECT_EQ(state->initProviders[10], state->initProviders[20]);
    ASSERT_EQ(state->initConfigs[20].endpoints.size(), std::size_t{1});
    EXPECT_EQ(state->initConfigs[20].endpoints[0].ip, "192.168.1.20");
    EXPECT_EQ(state->initConfigs[20].endpoints[0].protocol, Protocol::ROCE);
    EXPECT_EQ(state->initConfigs[20].deviceId, std::int32_t{6});
    for (auto asuId : {AsuId{10}, AsuId{20}}) {
        EXPECT_EQ(state->initConfigs[asuId].sendBufferSlotSize, std::size_t{8192});
        EXPECT_EQ(state->initConfigs[asuId].sendBufferSlotNum, std::size_t{2});
        EXPECT_EQ(state->initConfigs[asuId].flagBufferSlotSize, std::size_t{256});
        EXPECT_EQ(state->initConfigs[asuId].flagBufferSlotNum, std::size_t{32});
        EXPECT_EQ(state->initConfigs[asuId].asuBatchLoadIoNum, std::size_t{11});
        EXPECT_EQ(state->initConfigs[asuId].asuBatchStoreIoNum, std::size_t{12});
        EXPECT_EQ(state->initConfigs[asuId].asuDeleteIoNum, std::size_t{13});
        EXPECT_EQ(state->initConfigs[asuId].asuQueryIoNum, std::size_t{14});
        EXPECT_EQ(state->initConfigs[asuId].maxErrorCount, std::uint32_t{5});
        EXPECT_EQ(state->initConfigs[asuId].completionPollSpinLimit, std::size_t{23});
    }
}

TEST(AsuClientImplTest, Config_SeparatesClientAndTransportMaxInflightTasks)
{
    constexpr const char* kConfigPath = "kv_client_impl_max_inflight_tasks_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "client.maxInflightTasks=3\n";
        configFile << "wait_timeout_ms=321\n";
        configFile << "transport.asuIds=10\n";
        configFile << "transport.maxInflightTasks=7\n";
    }

    AsuClientConfig config;
    const auto status = LoadAsuClientConfig(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.maxInflightTasks, std::uint32_t{3});
    EXPECT_EQ(config.defaultWaitTimeoutMs, std::uint64_t{321});
    EXPECT_EQ(config.timeoutMs, std::uint64_t{321});
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    EXPECT_EQ(config.transportConfigs.front().timeoutMs, std::uint64_t{321});
    EXPECT_EQ(config.transportConfigs.front().maxInflightTasks, std::uint32_t{7});
}

TEST(AsuClientImplTest, Lifecycle_PublicInitRejectsInvalidSharedProviderMode)
{
    constexpr const char* kConfigPath = "kv_client_impl_invalid_shared_provider_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "asu_shared_provider=2\n";
        configFile << "transport.asuIds=10,20\n";
    }

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    const auto status = client->Init(kConfigPath);
    std::remove(kConfigPath);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(state->createdProviders, std::uint32_t{0});
    EXPECT_EQ(state->createdTransports, std::uint32_t{0});
}

TEST(AsuClientImplTest, Lifecycle_PublicInitIndependentModeBindsMemoryAcrossMultipleAsus)
{
    constexpr const char* kConfigPath = "kv_client_impl_multi_asu_bind_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "asu_shared_provider=0\n";
        configFile << "transport.asuIds=10,20\n";
    }

    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    auto status = client->Init(kConfigPath);
    std::remove(kConfigPath);
    ASSERT_TRUE(status.ok()) << status.message;

    std::vector<RegisteredMemory> registeredRegions;
    status = client->RegisterRegions({MemoryRegion{}}, registeredRegions);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->createdProviders, std::uint32_t{3});
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10, 20}));
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
    TaskResult result;
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());
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
    auto status = QueryAndWait(
        *client, {MakeCacheKey("k05"), MakeCacheKey("k15"), MakeCacheKey("k25")}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({0, 1, 1}));
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{0});
    ExpectSameAsuSet(state->queryCalls, {10, 20});
}

TEST(AsuClientImplTest, QueryAsync_CompletesThroughTransportCallback)
{
    auto state = std::make_shared<TestState>();
    state->deferCompletionCallbacks = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId{kInvalidTaskId};
    auto status = client->QueryAsync({MakeCacheKey("k15")}, taskId);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_NE(taskId, kInvalidTaskId);

    EXPECT_FALSE(client->Check(taskId));

    ASSERT_TRUE(WaitForPendingCompletion(state, 10));
    TaskResult completionResult;
    completionResult.status = Status::OK();
    completionResult.entryStatus = {Status::OK()};
    completionResult.queryResult = QueryResult{{1}, 0};
    ASSERT_TRUE(InvokePendingCompletion(state, 10, std::move(completionResult)));

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_TRUE(result.queryResult.has_value());
    EXPECT_EQ(result.queryResult->exists, std::vector<std::uint8_t>({1}));
}

TEST(AsuClientImplTest, Query_PerKeyDispatchFailureCancelsOtherTransports)
{
    auto state = std::make_shared<TestState>();
    state->queryFailures[20] = Status::Error(StatusCode::IO_ERROR, "fake per-key query failure");
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    QueryResult result;
    auto status = QueryAndWait(*client, {MakeCacheKey("k05"), MakeCacheKey("k15")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({0, 0}));
    ExpectSameAsuSet(state->queryCalls, {20});
}

TEST(AsuClientImplTest, Query_PerKeyResultSizeMismatchReturnsPartialFailed)
{
    class ShortQueryTransport final : public FakeTransport {
    public:
        explicit ShortQueryTransport(std::shared_ptr<TestState> state)
            : FakeTransport(std::move(state))
        {
        }

        Status RunQuery(BatchView<CacheKey>, QueryResult& result) override
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
}

TEST(AsuClientImplTest, Query_ComputesPrefixInOriginalKeyOrder)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    QueryResult result;
    auto status = QueryAndWait(
        *client, {MakeCacheKey("k15"), MakeCacheKey("k25"), MakeCacheKey("k05")}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({1, 1, 0}));
    EXPECT_EQ(result.prefixHitKeys, std::uint32_t{2});
    ExpectSameAsuSet(state->queryCalls, {10, 20});
}

TEST(AsuClientImplTest, Query_CompletionFailuresDoNotSkipOtherTransports)
{
    auto state = std::make_shared<TestState>();
    state->checkResultStatus[10] =
        Status::Error(StatusCode::IO_ERROR, "fake query completion failure");
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    QueryResult result;
    auto status = QueryAndWait(*client, {MakeCacheKey("k05"), MakeCacheKey("k15")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ExpectSameAsuSet(state->queryCalls, {10, 20});
    EXPECT_EQ(result.exists, std::vector<std::uint8_t>({0, 1}));
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryReturnsErrorWithoutRetry)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});

    QueryResult result;
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);

    status = QueryAndWait(*client, {MakeCacheKey("k15")}, result);

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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, BackgroundRefresh_QueryReturnsPartialFailedWhenViewFetchFails)
{
    auto state = std::make_shared<TestState>();
    state->failFirstQuery = true;
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    viewServer->FailFetchFrom(2);
    auto config = MakeConfig({10, 20});
    auto client =
        std::make_unique<AsuClientImpl>(MakeFactory(state), MakeViewServerFactory(viewServer));
    ASSERT_TRUE(client->Init(config).ok());

    QueryResult result;
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
}

TEST(AsuClientImplTest, BackgroundRefresh_LoadRecordsDispatchErrorWithoutRetry)
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

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_EQ(result.entryStatus.size(), std::size_t{1});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->loadCalls.empty());
}

TEST(AsuClientImplTest, BackgroundRefresh_StoreRecordsDispatchErrorWithoutRetry)
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

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_EQ(result.entryStatus.size(), std::size_t{1});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::CONNECTION_ERROR);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->storeCalls.empty());
}

TEST(AsuClientImplTest, BackgroundRefresh_DeleteRecordsDispatchErrorWithoutRetry)
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

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_EQ(result.entryStatus.size(), std::size_t{1});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::CONNECTION_ERROR);
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
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
    auto status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));

    status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->createdTransports, std::uint32_t{1});
    EXPECT_EQ(state->queryCalls, std::vector<AsuId>({10}));
}

TEST(AsuClientImplTest, MemoryRegister_RegistersOwnerAndBindsAllIndependentProviders)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}, MemoryRegion{}}, results);

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10, 20, 30}));
    ASSERT_EQ(results.size(), std::size_t{2});
    EXPECT_EQ(results[0].handle, MakeTestMrHandle(500));
    EXPECT_EQ(results[1].handle, MakeTestMrHandle(501));
    EXPECT_EQ(results[0].tokenId, std::uint32_t{900});
    EXPECT_EQ(results[1].tokenId, std::uint32_t{901});
}

TEST(AsuClientImplTest, MemoryRegister_NewIndependentProviderBindsRememberedRegions)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisteredMemory> results;
    ASSERT_TRUE(client->RegisterRegions({MemoryRegion{}}, results).ok());
    ASSERT_EQ(results.size(), std::size_t{1});

    state->failFirstQuery = true;
    QueryResult queryResult;
    EXPECT_EQ(QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult).code,
              StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    ASSERT_TRUE(client->Shutdown().ok());

    EXPECT_EQ(state->createdProviders, std::uint32_t{3});
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10, 20}));
}

TEST(AsuClientImplTest, MemoryRegister_PartialRegisterFailureDoesNotBindProviders)
{
    auto state = std::make_shared<TestState>();
    state->returnPartialRegister = true;
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}, MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->unregisterCalls, std::vector<AsuId>({10}));
    EXPECT_TRUE(results.empty());
}

TEST(AsuClientImplTest, MemoryRegister_ProviderBindFailureRollsBackOwner)
{
    auto state = std::make_shared<TestState>();
    state->failProviderBind = true;
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisteredMemory> results;
    const auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(state->registerCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10}));
    EXPECT_EQ(state->unregisterCalls, std::vector<AsuId>({10}));
    EXPECT_TRUE(results.empty());
}

TEST(AsuClientImplTest, MemoryRegister_RegisterFailureDoesNotBindProviders)
{
    auto state = std::make_shared<TestState>();
    state->failRegister = true;
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10},
            {10, 20}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, MemoryRegister_SuccessWithMismatchedResultCountReturnsInternalError)
{
    auto state = std::make_shared<TestState>();
    state->mismatchRegisterResultCount = true;
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
}

TEST(AsuClientImplTest, MemoryRegister_ProviderBindFailureIncludesProviderContext)
{
    auto state = std::make_shared<TestState>();
    state->failProviderBind = true;
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_NE(status.message.find("providerIndex=1"), std::string::npos);
}

TEST(AsuClientImplTest, MemoryRegister_ProviderBindFailureDoesNotCacheResource)
{
    auto state = std::make_shared<TestState>();
    state->failProviderBind = true;
    auto config = MakeConfig({10, 20, 30});
    auto viewServer = std::make_shared<FakeViewServer>(
        std::vector<std::vector<AsuId>>{
            {10, 20},
            {10, 20, 30}
    },
        std::vector<std::uint64_t>{1, 2});
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    state->failProviderBind = false;

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{3});
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10}));
}

TEST(AsuClientImplTest, MemoryRegister_UnregisterFailureIncludesAsuContext)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    std::vector<RegisteredMemory> registeredRegions;
    ASSERT_TRUE(client->RegisterRegions({MemoryRegion{}}, registeredRegions).ok());
    ASSERT_EQ(registeredRegions.size(), std::size_t{1});
    state->failUnregister = true;

    auto status = client->UnregisterRegions({registeredRegions[0].handle});

    EXPECT_EQ(status.code, StatusCode::IO_ERROR);
    EXPECT_NE(status.message.find("handle_count=1"), std::string::npos);
}

TEST(AsuClientImplTest, MemoryRegister_UnregisterReleasesBoundProvidersBeforeOwner)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20, 30})).ok());

    std::vector<RegisteredMemory> registeredRegions;
    ASSERT_TRUE(client->RegisterRegions({MemoryRegion{}}, registeredRegions).ok());
    state->unregisterCalls.clear();

    const auto status = client->UnregisterRegions({registeredRegions[0].handle});

    EXPECT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(state->unregisterCalls, std::vector<AsuId>({30, 20, 10, 10}));
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
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(results.size(), std::size_t{1});

    status = client->UnregisterRegions({results[0].handle});
    ASSERT_TRUE(status.ok()) << status.message;
    state->providerBindCalls.clear();

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_TRUE(state->providerBindCalls.empty());
}

TEST(AsuClientImplTest, Task_CheckKeepsTaskForWait)
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

    ASSERT_TRUE(CheckUntilComplete(*client, taskId));
    EXPECT_TRUE(client->Check(taskId));

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    ASSERT_TRUE(status.ok()) << status.message;

    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(ClientTaskTest, TransportTaskCompletionDoesNotReplaceLifecycleState)
{
    ClientTask ctx;
    ctx.remainingTransportTasks.store(1, std::memory_order_release);
    ctx.state.store(ClientTaskState::INFLIGHT, std::memory_order_release);

    EXPECT_FALSE(ctx.AllTransportTasksCompleted());
    EXPECT_FALSE(ctx.Done());

    ctx.remainingTransportTasks.fetch_sub(1, std::memory_order_acq_rel);

    EXPECT_TRUE(ctx.AllTransportTasksCompleted());
    EXPECT_FALSE(ctx.Done());

    ctx.state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    EXPECT_TRUE(ctx.Done());
}

TEST(AsuClientImplTest, Task_PassesOneCompletionCallbackPerTransportTask)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10, 20})).ok());
    auto entries = BuildRoutedEntries({10, 20});
    ASSERT_EQ(entries.size(), std::size_t{2});

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(entries, taskId);

    ASSERT_TRUE(status.ok()) << status.message;

    TaskResult result;
    EXPECT_TRUE(client->Wait(taskId, 100, result).ok());
    EXPECT_EQ(state->completionCallbackCount, std::size_t{2});
}

TEST(AsuClientImplTest, Task_SubmitReturnsWhileWorkerDispatchIsBlocked)
{
    auto state = std::make_shared<TestState>();
    state->blockStoreDispatch = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = kInvalidTaskId;
    auto submit = std::async(std::launch::async, [&] {
        return client->StoreAsync(
            {
                KVBuffer{MakeCacheKey("k05"), {}}
        },
            taskId);
    });

    bool dispatchStarted = false;
    {
        std::unique_lock<std::mutex> lock{state->storeDispatchMu};
        dispatchStarted = state->storeDispatchCv.wait_for(
            lock, std::chrono::milliseconds(100), [&] { return state->storeDispatchStarted; });
    }
    const bool submitReturned =
        submit.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
    {
        std::lock_guard<std::mutex> lock{state->storeDispatchMu};
        state->releaseStoreDispatch = true;
    }
    state->storeDispatchCv.notify_all();

    auto status = submit.get();
    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_TRUE(dispatchStarted);
    EXPECT_TRUE(submitReturned);
    EXPECT_NE(taskId, kInvalidTaskId);

    TaskResult result;
    EXPECT_TRUE(client->Wait(taskId, 100, result).ok());
}

TEST(AsuClientImplTest, Task_SubmitReturnsResourceBusyWhenQueueIsFull)
{
    auto state = std::make_shared<TestState>();
    state->blockStoreDispatch = true;
    auto client = CreateAsuClient(MakeFactory(state));
    auto config = MakeConfig({10});
    config.maxInflightTasks = 2;
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<TaskId> taskIds;
    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(client
                    ->StoreAsync(
                        {
                            KVBuffer{MakeCacheKey("k05"), {}}
    },
                        taskId)
                    .ok());
    taskIds.emplace_back(taskId);

    bool dispatchStarted = false;
    {
        std::unique_lock<std::mutex> lock{state->storeDispatchMu};
        dispatchStarted = state->storeDispatchCv.wait_for(
            lock, std::chrono::milliseconds(100), [&] { return state->storeDispatchStarted; });
    }

    std::vector<Status> queuedStatuses;
    for (std::size_t index = 0; index < 2; ++index) {
        taskId = kInvalidTaskId;
        queuedStatuses.emplace_back(client->StoreAsync(
            {
                KVBuffer{MakeCacheKey("queued-" + std::to_string(index)), {}}
        },
            taskId));
        if (queuedStatuses.back().ok()) { taskIds.emplace_back(taskId); }
    }

    taskId = 0;
    const auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("queue-full"), {}}
    },
        taskId);
    EXPECT_EQ(status.code, StatusCode::RESOURCE_BUSY);
    EXPECT_EQ(taskId, kInvalidTaskId);

    {
        std::lock_guard<std::mutex> lock{state->storeDispatchMu};
        state->releaseStoreDispatch = true;
    }
    state->storeDispatchCv.notify_all();

    ASSERT_TRUE(dispatchStarted);
    ASSERT_EQ(queuedStatuses.size(), std::size_t{2});
    EXPECT_TRUE(queuedStatuses[0].ok()) << queuedStatuses[0].message;
    EXPECT_TRUE(queuedStatuses[1].ok()) << queuedStatuses[1].message;
    for (const auto submittedTaskId : taskIds) {
        TaskResult result;
        EXPECT_TRUE(client->Wait(submittedTaskId, 100, result).ok());
    }
}

TEST(AsuClientImplTest, Task_CheckUsesClientStateUntilCompletionCallback)
{
    auto state = std::make_shared<TestState>();
    state->deferCompletionCallbacks = true;
    auto client = CreateAsuClient(MakeFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    TaskId taskId = 0;
    auto status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k05"), {}}
    },
        taskId);
    ASSERT_TRUE(status.ok()) << status.message;

    EXPECT_FALSE(client->Check(taskId));

    ASSERT_TRUE(WaitForPendingCompletion(state, 10));
    TaskResult completionResult;
    completionResult.status = Status::OK();
    completionResult.entryStatus = {Status::OK()};
    ASSERT_TRUE(InvokePendingCompletion(state, 10, std::move(completionResult)));
    EXPECT_TRUE(client->Check(taskId));

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    ASSERT_TRUE(status.ok()) << status.message;

    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::TASK_NOT_FOUND);
}

TEST(AsuClientImplTest, Task_CheckThenWaitHandlesRefreshableChildFailure)
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

    ASSERT_TRUE(CheckUntilComplete(*client, taskId));
    EXPECT_TRUE(client->Check(taskId));
    EXPECT_EQ(viewServer->FetchCount(), std::size_t{1});

    TaskResult result;
    status = client->Wait(taskId, 100, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
}

TEST(AsuClientImplTest, Task_PartialDispatchFailureWaitsForDispatchedSubtasks)
{
    auto state = std::make_shared<TestState>();
    state->failStoreAfterFirstDispatch = true;
    state->deferCompletionCallbacks = true;
    auto client = CreateAsuClient(MakeFactory(state));
    const std::vector<AsuId> asuIds{10, 20, 30};
    ASSERT_TRUE(client->Init(MakeConfig(asuIds)).ok());
    auto entries = BuildRoutedEntries(asuIds);
    ASSERT_EQ(entries.size(), asuIds.size());

    TaskId taskId = kInvalidTaskId;
    auto status = client->StoreAsync(entries, taskId);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_NE(taskId, kInvalidTaskId);
    AsuId dispatchedAsuId = 0;
    {
        std::unique_lock<std::mutex> lock{state->completionMu};
        ASSERT_TRUE(state->completionCv.wait_for(lock, std::chrono::milliseconds(100), [&] {
            return !state->pendingCompletionCallbacks.empty();
        }));
        dispatchedAsuId = state->pendingCompletionCallbacks.begin()->first;
    }
    ASSERT_EQ(state->storeCalls.size(), std::size_t{1});
    EXPECT_EQ(state->storeCalls[0], dispatchedAsuId);
    EXPECT_FALSE(client->Check(taskId));
    EXPECT_TRUE(state->cancelCalls.empty());

    TaskResult completionResult;
    completionResult.status = Status::OK();
    completionResult.entryStatus = {Status::OK()};
    ASSERT_TRUE(InvokePendingCompletion(state, dispatchedAsuId, std::move(completionResult)));

    TaskResult result;
    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    EXPECT_EQ(state->completionCallbackCount, std::size_t{1});
    ASSERT_EQ(result.entryStatus.size(), std::size_t{3});
    EXPECT_EQ(std::count_if(result.entryStatus.begin(), result.entryStatus.end(),
                            [](const Status& entryStatus) { return entryStatus.ok(); }),
              1);
    EXPECT_EQ(std::count_if(result.entryStatus.begin(), result.entryStatus.end(),
                            [](const Status& entryStatus) {
                                return entryStatus.code == StatusCode::CONNECTION_ERROR;
                            }),
              1);
    EXPECT_EQ(std::count_if(result.entryStatus.begin(), result.entryStatus.end(),
                            [](const Status& entryStatus) {
                                return entryStatus.code == StatusCode::CANCELED;
                            }),
              1);
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

TEST(AsuClientImplTest, Task_WaitTimeoutRemovesTask)
{
    auto state = std::make_shared<TestState>();
    state->deferCompletionCallbacks = true;
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
    status = client->Wait(taskId, 100, result);
    EXPECT_EQ(status.code, StatusCode::TIMEOUT);

    EXPECT_TRUE(client->Check(taskId));

    ASSERT_TRUE(WaitForPendingCompletion(state, 10));
    TaskResult completionResult;
    completionResult.status = Status::OK();
    completionResult.entryStatus = {Status::OK()};
    EXPECT_TRUE(InvokePendingCompletion(state, 10, std::move(completionResult)));
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
    status = client->Wait(taskId, 100, result);

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
    status = client->Wait(taskId, 100, result);

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
    status = client->Wait(taskId, 100, result);

    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(result.entryStatus.size(), std::size_t{3});
    EXPECT_EQ(result.entryStatus[0].code, StatusCode::NOT_FOUND);
    EXPECT_EQ(result.entryStatus[1].code, StatusCode::OK);
    EXPECT_EQ(result.entryStatus[2].code, StatusCode::IO_ERROR);
}

TEST(AsuClientImplTest, SnapshotRefresh_ReusesTransportAndBindsProviderForAddedAsu)
{
    auto state = std::make_shared<TestState>();
    auto config = MakeConfig({10, 20});
    auto viewServer = std::make_shared<FakeViewServer>(std::vector<std::vector<AsuId>>{
        {10},
        {10, 20}
    });
    auto client = std::make_unique<AsuClientImpl>(
        MakeFactory(state), MakeViewServerFactory(viewServer), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(config).ok());

    std::vector<RegisteredMemory> results;
    auto status = client->RegisterRegions({MemoryRegion{}}, results);
    ASSERT_TRUE(status.ok()) << status.message;
    state->failFirstQuery = true;

    QueryResult result;
    status = QueryAndWait(*client, {MakeCacheKey("k05")}, result);

    EXPECT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(state->createdTransports, std::uint32_t{2});
    EXPECT_EQ(state->providerBindCalls, std::vector<AsuId>({10, 20}));
}

TEST(AsuClientImplTest, MemoryRegister_StoreTaskCarriesMrKeyInEntry)
{
    auto state = std::make_shared<TestState>();
    auto client = CreateAsuClient(MakeFactory(state), MakeProviderFactory(state));
    ASSERT_TRUE(client->Init(MakeConfig({10})).ok());

    std::vector<RegisteredMemory> registeredRegions;
    ASSERT_TRUE(client->RegisterRegions({MemoryRegion{}}, registeredRegions).ok());
    ASSERT_EQ(registeredRegions.size(), std::size_t{1});
    KVBuffer entry{MakeCacheKey("k05"), {}};
    entry.buffer.handle = registeredRegions[0].handle;

    TaskId taskId = kInvalidTaskId;
    ASSERT_TRUE(client->StoreAsync({entry}, taskId).ok());
    TaskResult result;
    ASSERT_TRUE(client->Wait(taskId, 100, result).ok());

    ASSERT_EQ(state->submittedEntries.size(), std::size_t{1});
    ASSERT_TRUE(state->submittedEntries[0].mrKey.has_value());
    EXPECT_EQ(*state->submittedEntries[0].mrKey, registeredRegions[0].tokenId);
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

    state->failFirstQuery = true;
    QueryResult queryResult;
    status = QueryAndWait(*client, {MakeCacheKey("k05")}, queryResult);
    ASSERT_EQ(status.code, StatusCode::PARTIAL_FAILED);
    ASSERT_TRUE(WaitForFetchCount(viewServer, 2));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    TaskResult taskResult;
    ASSERT_TRUE(CheckUntilComplete(*client, taskId));
    status = client->Wait(taskId, 100, taskResult);
    EXPECT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(state->storeCalls, std::vector<AsuId>({20}));

    status = client->StoreAsync(
        {
            KVBuffer{MakeCacheKey("k15"), {}}
    },
        taskId);
    EXPECT_TRUE(status.ok()) << status.message;
    TaskResult secondTaskResult;
    EXPECT_TRUE(client->Wait(taskId, 100, secondTaskResult).ok());
    EXPECT_EQ(state->storeCalls, std::vector<AsuId>({20, 10}));
}

}  // namespace kv::test
