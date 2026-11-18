#include "view/view_server.h"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include "kv_client.h"
#include "view/view_universe.h"

namespace kv::test {
namespace {

TEST(ViewServerTest, ConfigFileViewServerLoadsView)
{
    constexpr const char* kConfigPath = "kv_client_view_server_test.conf";
    {
        std::ofstream configFile{kConfigPath};
        ASSERT_TRUE(configFile.is_open());
        configFile << "viewEpoch=7\n";
        configFile << "viewId=3\n";
        configFile << "asuIds=10,20\n";
        configFile << "asuInfo.20=protocol=uboe,placement=device,port=6000,"
                   << "local.comm_id=192.168.1.20,send_size=4096,"
                   << "remote_send_addr=0x100000000\n";
    }

    KvClientConfig config;
    config.viewServiceAddrs = {kConfigPath};
    auto viewServer = CreateDefaultViewServer(config);

    GlobalView view;
    auto status = viewServer->GetGlobalView(view);
    std::remove(kConfigPath);

    ASSERT_TRUE(status.ok()) << status.message;
    EXPECT_EQ(view.viewEpoch, std::uint64_t{7});
    EXPECT_EQ(view.viewId, std::uint64_t{3});
    ASSERT_EQ(view.asuMap.size(), std::size_t{2});
    ASSERT_EQ(view.asuMap[20].endpoints.size(), std::size_t{1});
    EXPECT_EQ(view.asuMap[20].endpoints[0].ip, "192.168.1.20");
    EXPECT_EQ(view.asuMap[20].endpoints[0].port, std::uint16_t{6000});
    EXPECT_EQ(view.asuMap[20].endpoints[0].protocol, Protocol::UB);
    EXPECT_EQ(view.asuMap[20].endpoints[0].attrs["protocol"], "uboe");
    EXPECT_EQ(view.asuMap[20].endpoints[0].attrs["placement"], "device");
    EXPECT_EQ(view.asuMap[20].endpoints[0].attrs["send_size"], "4096");
    EXPECT_EQ(view.asuMap[20].endpoints[0].attrs["remote_send_addr"], "0x100000000");
}

TEST(ViewServerTest, ConfigBackedViewServerBuildsViewFromTransportConfigs)
{
    KvClientConfig config;
    TransportConfig transportConfig;
    transportConfig.nodeId = 10;
    NodeEndpoint endpoint;
    endpoint.ip = "127.0.0.1";
    endpoint.port = 6000;
    endpoint.protocol = Protocol::ROCE;
    transportConfig.endpoints.emplace_back(endpoint);
    config.transportConfigs.emplace_back(transportConfig);

    auto viewServer = CreateDefaultViewServer(config);

    GlobalView view;
    auto status = viewServer->GetGlobalView(view);

    ASSERT_TRUE(status.ok()) << status.message;
    ASSERT_EQ(view.asuMap.size(), std::size_t{1});
    ASSERT_EQ(view.asuMap[10].endpoints.size(), std::size_t{1});
    EXPECT_EQ(view.asuMap[10].endpoints[0].ip, "127.0.0.1");
    EXPECT_EQ(view.asuMap[10].endpoints[0].protocol, Protocol::ROCE);
}

TEST(ViewServerTest, LegacySingleViewIsAdaptedToAllViews)
{
    KvClientConfig config;
    TransportConfig transportConfig;
    transportConfig.nodeId = 10;
    config.transportConfigs.emplace_back(transportConfig);
    auto viewServer = CreateDefaultViewServer(config);
    auto universe = CreateViewUniverse();

    ASSERT_TRUE(viewServer->GetAllViews(*universe).ok());
    const auto nsIds = universe->GetAllNsIds();
    ASSERT_EQ(nsIds.size(), 1);
    EXPECT_EQ(nsIds.front(), 0);
    GlobalView view;
    ASSERT_TRUE(universe->GetView(nsIds.front(), view).ok());
    EXPECT_EQ(view.asuMap.count(10), 1);
}

class MultiNamespaceViewServer final : public ViewServer {
public:
    Status GetAllViews(ViewUniverse& universe) override
    {
        std::unordered_map<KvNsId, GlobalView> views;
        for (const auto nsId : {KvNsId{20}, KvNsId{10}}) {
            GlobalView view;
            view.viewId = nsId;
            view.viewEpoch = 9;
            views.emplace(nsId, std::move(view));
        }
        auto next = CreateViewUniverse(std::move(views), 9);
        return universe.ReplaceAll(*next);
    }
};

class EmptyNamespaceViewServer final : public ViewServer {
public:
    Status GetAllViews(ViewUniverse&) override { return Status::OK(); }
};

TEST(ViewServerTest, DefaultGetGlobalViewReturnsFirstNamespace)
{
    MultiNamespaceViewServer server;
    GlobalView view;
    ASSERT_TRUE(server.GetGlobalView(view).ok());
    EXPECT_EQ(view.viewId, 10);
    EXPECT_EQ(view.viewEpoch, 9);
}

TEST(ViewServerTest, DefaultGetGlobalViewReturnsNotFoundForEmptyUniverse)
{
    EmptyNamespaceViewServer server;
    GlobalView view;
    EXPECT_EQ(server.GetGlobalView(view).code, StatusCode::NOT_FOUND);
}

TEST(ViewServerTest, PublishAndRefreshPolicies)
{
    KvClientConfig config;
    auto viewServer = CreateDefaultViewServer(config);

    GlobalView published;
    published.viewEpoch = 4;
    GlobalView older;
    older.viewEpoch = 3;
    GlobalView newer;
    newer.viewEpoch = 5;
    GlobalView unknownEpoch;

    EXPECT_FALSE(viewServer->ShouldPublishView(published, older));
    EXPECT_TRUE(viewServer->ShouldPublishView(published, newer));
    EXPECT_TRUE(viewServer->ShouldPublishView(published, unknownEpoch));

    EXPECT_TRUE(viewServer->ShouldRefreshView(Status::Error(StatusCode::IO_ERROR, "io")));
    EXPECT_TRUE(viewServer->ShouldRefreshView(Status::Error(StatusCode::TIMEOUT, "timeout")));
    EXPECT_FALSE(viewServer->ShouldRefreshView(Status::Error(StatusCode::INVALID_ARGUMENT, "bad")));

    TaskResult result;
    result.status = Status::OK();
    result.entryStatus = {Status::OK(), Status::Error(StatusCode::NOT_FOUND, "missing")};
    EXPECT_TRUE(viewServer->ShouldRefreshView(result));
}

}  // namespace
}  // namespace kv::test
