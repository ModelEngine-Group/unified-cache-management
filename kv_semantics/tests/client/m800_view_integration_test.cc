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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "kv_client_impl.h"
#include "view/m800_view_master_server.h"

namespace kv::test {
namespace {

GlobalView MakeView(std::uint64_t epoch, std::vector<NodeId> nodeIds)
{
    GlobalView view;
    view.viewEpoch = epoch;
    view.viewId = epoch;
    for (auto nodeId : nodeIds) {
        NodeEndpoint endpoint;
        endpoint.ip = "127.0.0.1";
        endpoint.port = 6000;
        endpoint.protocol = Protocol::UB;
        NodeInfo info;
        info.endpoints.emplace_back(endpoint);
        view.asuMap.emplace(nodeId, info);
    }
    return view;
}

class FakeM800Master final : public M800ViewMasterServer {
public:
    Status Start() override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        ++startCount_;
        return startStatus_;
    }

    void Stop() override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        ++stopCount_;
    }

    Status GetAllViews(std::unordered_map<KvNsId, GlobalView>& views) override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        ++getAllCount_;
        std::uint64_t requestEpoch{0};
        for (const auto& item : views) {
            requestEpoch = std::max(requestEpoch, item.second.viewEpoch);
        }
        lastRequestEpoch_ = requestEpoch;
        if (!getAllStatus_.ok()) { return getAllStatus_; }
        if (views_.empty()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "fake M800 view map is empty");
        }
        views = views_;
        return Status::OK();
    }

    Status GetGlobalViewByKvnsId(KvNsId nsId, GlobalView& view) override
    {
        std::unordered_map<KvNsId, GlobalView> views;
        const auto status = GetAllViews(views);
        if (!status.ok()) { return status; }
        const auto iter = views.find(nsId);
        if (iter == views.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "namespace not found");
        }
        view = iter->second;
        return Status::OK();
    }

    Status GetGlobalView(GlobalView& view) override
    {
        (void)view;
        return Status::Error(StatusCode::UNSUPPORTED,
                             "M800 view master serves multiple namespaces; use GetAllViews");
    }

    void SetViews(std::unordered_map<KvNsId, GlobalView> views)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        views_ = std::move(views);
    }

    void SetGetAllStatus(Status status)
    {
        std::lock_guard<std::mutex> lock{mutex_};
        getAllStatus_ = std::move(status);
    }

    int StartCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return startCount_;
    }

    int StopCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return stopCount_;
    }

    int GetAllCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return getAllCount_;
    }

    std::uint64_t LastRequestEpoch() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return lastRequestEpoch_;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<KvNsId, GlobalView> views_;
    Status startStatus_{Status::OK()};
    Status getAllStatus_{Status::OK()};
    int startCount_{0};
    int stopCount_{0};
    int getAllCount_{0};
    std::uint64_t lastRequestEpoch_{0};
};

class SequenceViewServer final : public ViewServer {
public:
    explicit SequenceViewServer(std::vector<std::vector<NodeId>> views) : views_(std::move(views))
    {
    }

    Status GetGlobalView(GlobalView& view) override
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto index = fetchCount_;
        if (index >= views_.size()) { index = views_.empty() ? 0 : views_.size() - 1; }
        ++fetchCount_;
        view = MakeView(index + 1, views_[index]);
        return Status::OK();
    }

    std::size_t FetchCount() const
    {
        std::lock_guard<std::mutex> lock{mutex_};
        return fetchCount_;
    }

private:
    mutable std::mutex mutex_;
    std::size_t fetchCount_{0};
    std::vector<std::vector<NodeId>> views_;
};

bool WaitForFetchCount(const std::shared_ptr<SequenceViewServer>& viewServer, std::size_t expected)
{
    for (int attempt = 0; attempt < 300; ++attempt) {
        if (viewServer->FetchCount() >= expected) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool WaitForGetAllCount(const std::shared_ptr<FakeM800Master>& master, int expected)
{
    for (int attempt = 0; attempt < 300; ++attempt) {
        if (master->GetAllCount() >= expected) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

struct MultiViewTransportState {
    mutable std::mutex mutex;
    std::vector<NodeId> initNodeIds;
    std::vector<NodeId> shutdownNodeIds;

    void RecordInit(NodeId nodeId)
    {
        std::lock_guard<std::mutex> lock{mutex};
        initNodeIds.emplace_back(nodeId);
    }

    void RecordShutdown(NodeId nodeId)
    {
        std::lock_guard<std::mutex> lock{mutex};
        shutdownNodeIds.emplace_back(nodeId);
    }

    std::size_t Count(const std::vector<NodeId>& nodeIds, NodeId nodeId) const
    {
        std::size_t count = 0;
        for (const auto item : nodeIds) {
            if (item == nodeId) { ++count; }
        }
        return count;
    }

    std::size_t InitCount(NodeId nodeId) const
    {
        std::lock_guard<std::mutex> lock{mutex};
        return Count(initNodeIds, nodeId);
    }

    std::size_t ShutdownCount(NodeId nodeId) const
    {
        std::lock_guard<std::mutex> lock{mutex};
        return Count(shutdownNodeIds, nodeId);
    }
};

class MultiViewTransport final : public AsuTransport {
public:
    explicit MultiViewTransport(std::shared_ptr<MultiViewTransportState> state)
        : state_(std::move(state))
    {
    }

    Status Init(const TransportConfig& config,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)transProvider;
        config_ = config;
        state_->RecordInit(config.nodeId);
        return Status::OK();
    }

    Status Init(const std::string& configPath,
                std::shared_ptr<TransProvider> transProvider) override
    {
        (void)configPath;
        (void)transProvider;
        return Status::Error(StatusCode::UNSUPPORTED,
                             "multi-view test transport does not use a config path");
    }

    Status Shutdown() override
    {
        state_->RecordShutdown(config_.nodeId);
        return Status::OK();
    }

    Status CheckHealth() override { return Status::OK(); }

    Status Submit(const std::shared_ptr<TransportTask>& task) override
    {
        (void)task;
        return Status::OK();
    }

    Status Cancel(TaskId taskId) override
    {
        (void)taskId;
        return Status::OK();
    }

private:
    TransportConfig config_;
    std::shared_ptr<MultiViewTransportState> state_;
};

TransportFactory MakeMultiViewTransportFactory(
    const std::shared_ptr<MultiViewTransportState>& state)
{
    return [state] { return std::make_unique<MultiViewTransport>(state); };
}

}  // namespace

ViewSnapshotPtr Snapshot(const KvClientImpl& client, KvNsId nsId)
{
    return client.GetSnapshotByKvnsId(nsId);
}

bool WaitForSnapshotEpoch(const KvClientImpl& client, KvNsId nsId, std::uint64_t epoch)
{
    for (int attempt = 0; attempt < 300; ++attempt) {
        const auto snapshot = Snapshot(client, nsId);
        if (snapshot != nullptr && snapshot->view.viewEpoch == epoch) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

KvClientConfig MakeClientConfig(const std::vector<NodeId>& nodeIds)
{
    KvClientConfig config;
    for (auto nodeId : nodeIds) {
        TransportConfig transportConfig;
        transportConfig.nodeId = nodeId;
        transportConfig.providerType = TransProviderType::FAKE;
        config.transportConfigs.emplace_back(transportConfig);
    }
    config.attrs["view.provider"] = "m800";
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addrs"] = "127.0.0.1:9100";
    config.attrs["view.refresh_interval_ms"] = "10";
    return config;
}

KvClientConfig MakeM800ConfigWithTransportDefault()
{
    KvClientConfig config;
    config.attrs["view.provider"] = "m800";
    config.attrs["view.role"] = "worker";
    config.attrs["view.master_addrs"] = "127.0.0.1:9100";
    config.attrs["view.refresh_interval_ms"] = "10";
    TransportConfig transportDefault;
    transportDefault.providerType = TransProviderType::FAKE;
    transportDefault.deviceId = 0;
    config.transportConfigs.emplace_back(std::move(transportDefault));
    return config;
}

TEST(M800ViewIntegration, InvalidProviderConfigIsReturnedByDefaultViewServer)
{
    KvClientConfig config;
    config.attrs["view.provider"] = "m800";
    config.attrs["view.role"] = "worker";

    auto viewServer = CreateDefaultViewServer(config);
    ASSERT_NE(viewServer, nullptr);
    GlobalView view;
    const auto status = viewServer->GetGlobalView(view);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view.master_addrs"), std::string::npos);
}

TEST(M800ViewIntegration, UnsupportedViewProviderFailsInsteadOfFallingBack)
{
    KvClientConfig config;
    config.attrs["view.provider"] = "unknown";

    auto viewServer = CreateDefaultViewServer(config);
    GlobalView view;
    const auto status = viewServer->GetGlobalView(view);

    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view.provider"), std::string::npos);
}

TEST(M800ViewIntegration, ConfigFilePreservesViewAttrs)
{
    constexpr const char* kConfigPath = "m800_view_integration_config_attrs_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "view.provider=m800\n";
        configFile << "view.role=worker\n";
        configFile << "view.master_addrs=127.0.0.1:9100\n";
        configFile << "view.refresh_interval_ms=10\n";
    }

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(config.attrs.at("view.provider"), "m800");
    EXPECT_EQ(config.attrs.at("view.role"), "worker");
    EXPECT_EQ(config.attrs.at("view.master_addrs"), "127.0.0.1:9100");
    EXPECT_EQ(config.attrs.at("view.refresh_interval_ms"), "10");
}

TEST(M800ViewIntegration, KvClientSchedulerRefreshesM800ViewMap)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100})).ok());
    master->SetViews({
        {10, MakeView(2, {100})}
    });
    EXPECT_TRUE(WaitForGetAllCount(master, 2));

    ASSERT_TRUE(client->Shutdown().ok());
    client.reset();
    EXPECT_EQ(master->StopCount(), 1);
}

TEST(M800ViewIntegration, RefreshSeedsConditionalViewRequest)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100})).ok());
    EXPECT_EQ(master->LastRequestEpoch(), std::uint64_t{0});

    master->SetViews({
        {10, MakeView(2, {100})}
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, 10, 2));
    EXPECT_EQ(master->LastRequestEpoch(), std::uint64_t{1});

    ASSERT_TRUE(client->Shutdown().ok());
}

TEST(M800ViewIntegration, InitBuildsSnapshotForEachNamespace)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(7, {100})},
        {20, MakeView(7, {200})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100, 200})).ok());

    const auto ns10 = Snapshot(*client, 10);
    const auto ns20 = Snapshot(*client, 20);
    ASSERT_NE(ns10, nullptr);
    ASSERT_NE(ns20, nullptr);
    EXPECT_EQ(ns10->nsId, KvNsId{10});
    EXPECT_EQ(ns10->asuIds, std::vector<NodeId>{100});
    EXPECT_EQ(ns10->view.viewEpoch, std::uint64_t{7});
    EXPECT_NE(ns10->router, nullptr);
    EXPECT_EQ(ns20->nsId, KvNsId{20});
    EXPECT_EQ(ns20->asuIds, std::vector<NodeId>{200});
    EXPECT_EQ(ns20->view.viewEpoch, std::uint64_t{7});
    EXPECT_NE(ns20->router, nullptr);

    ASSERT_TRUE(client->Shutdown().ok());
    client.reset();
    EXPECT_EQ(master->StartCount(), 1);
    EXPECT_EQ(master->StopCount(), 1);
}

TEST(M800ViewIntegration, InitRejectsMixedNamespaceEpochs)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(7, {100})},
        {20, MakeView(8, {200})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    const auto status = client->Init(MakeClientConfig({100, 200}));
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view epochs"), std::string::npos);
    EXPECT_EQ(master->StopCount(), 1);
}

TEST(M800ViewIntegration, RefreshReplacesEveryNamespaceSnapshot)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})},
        {20, MakeView(1, {200})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100, 200, 300})).ok());
    const auto initialNs20 = Snapshot(*client, 20);
    master->SetViews({
        {10, MakeView(2, {100, 300})},
        {20, MakeView(2, {200})     }
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, 10, 2));

    const auto ns10 = Snapshot(*client, 10);
    const auto ns20 = Snapshot(*client, 20);
    ASSERT_NE(ns10, nullptr);
    ASSERT_NE(ns20, nullptr);
    EXPECT_EQ(ns10->view.viewEpoch, std::uint64_t{2});
    EXPECT_EQ(ns10->asuIds, std::vector<NodeId>({100, 300}));
    EXPECT_EQ(ns20->view.viewEpoch, std::uint64_t{2});
    EXPECT_EQ(ns20->asuIds, std::vector<NodeId>{200});
    EXPECT_NE(ns20, initialNs20);

    ASSERT_TRUE(client->Shutdown().ok());
}

TEST(M800ViewIntegration, RefreshFailureKeepsOldSnapshots)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})},
        {20, MakeView(1, {200})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100, 200})).ok());
    const auto initialNs10 = Snapshot(*client, 10);
    const auto initialNs20 = Snapshot(*client, 20);
    master->SetViews({
        {10, MakeView(2, {100})     },
        {20, MakeView(2, {200, 300})}
    });
    master->SetGetAllStatus(Status::Error(StatusCode::CONNECTION_ERROR, "refresh failed"));
    ASSERT_TRUE(WaitForGetAllCount(master, 3));
    master->SetGetAllStatus(Status::OK());

    EXPECT_EQ(Snapshot(*client, 10), initialNs10);
    EXPECT_EQ(Snapshot(*client, 20), initialNs20);
    EXPECT_EQ(initialNs10->view.viewEpoch, std::uint64_t{1});
    EXPECT_EQ(initialNs20->view.viewEpoch, std::uint64_t{1});

    ASSERT_TRUE(client->Shutdown().ok());
}

TEST(M800ViewIntegration, RefreshAddsNamespaceSnapshotAndSharesTransport)
{
    auto transportState = std::make_shared<MultiViewTransportState>();
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})}
    });
    auto client = std::make_unique<KvClientImpl>(
        MakeMultiViewTransportFactory(transportState),
        [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100})).ok());
    master->SetViews({
        {10, MakeView(2, {100})},
        {20, MakeView(2, {100})}
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, 20, 2));

    const auto ns10 = Snapshot(*client, 10);
    const auto ns20 = Snapshot(*client, 20);
    ASSERT_NE(ns10, nullptr);
    ASSERT_NE(ns20, nullptr);
    EXPECT_EQ(ns10->nsId, KvNsId{10});
    EXPECT_EQ(ns20->nsId, KvNsId{20});
    ASSERT_NE(ns10->transports.find(100), ns10->transports.end());
    ASSERT_NE(ns20->transports.find(100), ns20->transports.end());
    EXPECT_EQ(ns10->transports.at(100).get(), ns20->transports.at(100).get());
    EXPECT_EQ(transportState->InitCount(100), std::size_t{1});

    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(transportState->ShutdownCount(100), std::size_t{1});
}

TEST(M800ViewIntegration, RefreshRemovesNamespaceSnapshotAndKeepsPooledTransport)
{
    auto transportState = std::make_shared<MultiViewTransportState>();
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})},
        {20, MakeView(1, {100})}
    });
    auto client = std::make_unique<KvClientImpl>(
        MakeMultiViewTransportFactory(transportState),
        [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeClientConfig({100})).ok());
    const auto removedSnapshot = Snapshot(*client, 20);
    ASSERT_NE(removedSnapshot, nullptr);
    ASSERT_NE(removedSnapshot->transports.find(100), removedSnapshot->transports.end());

    master->SetViews({
        {10, MakeView(2, {100})}
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, 10, 2));
    EXPECT_EQ(Snapshot(*client, 20), nullptr);

    const auto activeSnapshot = Snapshot(*client, 10);
    ASSERT_NE(activeSnapshot, nullptr);
    ASSERT_NE(activeSnapshot->transports.find(100), activeSnapshot->transports.end());
    EXPECT_EQ(activeSnapshot->transports.at(100).get(), removedSnapshot->transports.at(100).get());

    ASSERT_TRUE(client->Shutdown().ok());
    EXPECT_EQ(transportState->InitCount(100), std::size_t{1});
    EXPECT_EQ(transportState->ShutdownCount(100), std::size_t{1});
}

TEST(M800ViewIntegration, LegacyProviderUsesRefreshScheduler)
{
    auto viewServer = std::make_shared<SequenceViewServer>(std::vector<std::vector<NodeId>>{
        {10},
        {10, 20}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [viewServer](const KvClientConfig&) { return viewServer; }, nullptr);

    KvClientConfig config;
    TransportConfig transportConfig;
    transportConfig.nodeId = 10;
    transportConfig.providerType = TransProviderType::FAKE;
    config.transportConfigs.emplace_back(transportConfig);
    config.attrs["view.refresh_interval_ms"] = "10";

    ASSERT_TRUE(client->Init(config).ok());
    EXPECT_TRUE(WaitForFetchCount(viewServer, 2));
    ASSERT_TRUE(client->Shutdown().ok());

    const auto fetchCountAfterShutdown = viewServer->FetchCount();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_EQ(viewServer->FetchCount(), fetchCountAfterShutdown);
}

TEST(M800ViewIntegration, InvalidLegacyRefreshIntervalFailsInitialization)
{
    auto viewServer = std::make_shared<SequenceViewServer>(std::vector<std::vector<NodeId>>{{10}});
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [viewServer](const KvClientConfig&) { return viewServer; }, nullptr);

    KvClientConfig config;
    TransportConfig transportConfig;
    transportConfig.nodeId = 10;
    transportConfig.providerType = TransProviderType::FAKE;
    config.transportConfigs.emplace_back(transportConfig);
    config.attrs["view.refresh_interval_ms"] = "0";

    const auto status = client->Init(config);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view.refresh_interval_ms"), std::string::npos);
    EXPECT_EQ(viewServer->FetchCount(), std::size_t{0});
}

TEST(M800ViewIntegration, InvalidM800RefreshIntervalFailsInitialization)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {10, MakeView(1, {100})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    auto config = MakeClientConfig({100});
    config.attrs["view.refresh_interval_ms"] = "0";

    const auto status = client->Init(config);
    EXPECT_EQ(status.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message.find("view.refresh_interval_ms"), std::string::npos);
    EXPECT_EQ(master->GetAllCount(), 0);
}

TEST(M800ViewIntegration, ConfigFileSynthesizesTransportDefaultWithoutAsuIds)
{
    constexpr const char* kConfigPath = "m800_view_transport_defaults_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "view.provider=m800\n";
        configFile << "view.role=worker\n";
        configFile << "view.master_addrs=127.0.0.1:9100\n";
        configFile << "transport.provider_type=FAKE\n";
        configFile << "transport.device_id=3\n";
        configFile << "transport.roce_gid_index=7\n";
        configFile << "asu_info.10=protocol=UB,local.comm_id=127.0.0.1,port=6000\n";
    }

    KvClientConfig config;
    const auto status = KvClientConfig::LoadFromFile(kConfigPath, config);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(config.transportConfigs.size(), std::size_t{1});
    EXPECT_EQ(config.transportConfigs.front().nodeId, NodeId{0});
    EXPECT_EQ(config.transportConfigs.front().providerType, TransProviderType::FAKE);
    EXPECT_EQ(config.transportConfigs.front().deviceId, 3);
    // Unknown transport.* keys are kept for the provider, not dropped.
    EXPECT_EQ(config.transportConfigs.front().attrs.at("roce_gid_index"), "7");
}

TEST(M800ViewIntegration, InitBuildsSnapshotFromViewWithoutTransportConfigs)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {kDefaultKvNsId, MakeView(1, {10, 20, 30})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeM800ConfigWithTransportDefault()).ok());

    const auto snapshot = Snapshot(*client, kDefaultKvNsId);
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->asuIds, std::vector<NodeId>({10, 20, 30}));
    EXPECT_EQ(snapshot->transports.size(), std::size_t{3});

    ASSERT_TRUE(client->Shutdown().ok());
}

TEST(M800ViewIntegration, RefreshFollowsViewNodesWithoutTransportConfigs)
{
    auto master = std::make_shared<FakeM800Master>();
    master->SetViews({
        {kDefaultKvNsId, MakeView(1, {10, 20})}
    });
    auto client = std::make_unique<KvClientImpl>(
        nullptr, [master](const KvClientConfig&) { return master; }, nullptr);

    ASSERT_TRUE(client->Init(MakeM800ConfigWithTransportDefault()).ok());
    ASSERT_EQ(Snapshot(*client, kDefaultKvNsId)->asuIds, std::vector<NodeId>({10, 20}));

    master->SetViews({
        {kDefaultKvNsId, MakeView(2, {10, 20, 30})}
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, kDefaultKvNsId, 2));
    const auto added = Snapshot(*client, kDefaultKvNsId);
    ASSERT_NE(added, nullptr);
    EXPECT_EQ(added->asuIds, std::vector<NodeId>({10, 20, 30}));
    EXPECT_EQ(added->transports.size(), std::size_t{3});

    master->SetViews({
        {kDefaultKvNsId, MakeView(3, {20, 30})}
    });
    ASSERT_TRUE(WaitForSnapshotEpoch(*client, kDefaultKvNsId, 3));
    const auto removed = Snapshot(*client, kDefaultKvNsId);
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->asuIds, std::vector<NodeId>({20, 30}));
    EXPECT_EQ(removed->transports.size(), std::size_t{2});

    ASSERT_TRUE(client->Shutdown().ok());
}

}  // namespace kv::test
