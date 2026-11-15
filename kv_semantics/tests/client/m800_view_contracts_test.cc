#include <gtest/gtest.h>
#include <type_traits>
#include "KVClientAgentABI.h"
#include "router/router.h"
#include "view/m800_view_agent_client.h"
#include "view/m800_view_codec.h"
#include "view/m800_view_common.h"
#include "view/m800_view_master_server.h"
#include "view/m800_view_slave_server.h"
#include "view/refresh_scheduler.h"
#include "view/view_http_server.h"
#include "view/view_universe.h"

namespace kv::test {
namespace {

class FixedRouter final : public Router {
public:
    explicit FixedRouter(HashFunction hash) : Router(std::move(hash)) {}

private:
    NodeId RouteKey(const Key&) const override { return 10; }
};

TEST(M800ViewContractsTest, RouterFallbackContractPreservesLegacyRouting)
{
    FixedRouter router([](const Key&) { return 0; });
    const std::vector<Key> keys = {"key-0", "key-1", "key-2"};
    const UnavailableNodeSet unavailableNodes = {10};

    const auto legacyRoutes = router.RouteKeys(keys);
    const auto fallbackRoutes = router.RouteKeys(keys, unavailableNodes);

    EXPECT_EQ(legacyRoutes, fallbackRoutes);
}

TEST(M800ViewContractsTest, M800TypesAndHTTPContractCompile)
{
    static_assert(M800ViewRole::MASTER == M800ViewRole{0});
    static_assert(M800ViewRole::SLAVE == M800ViewRole{1});
    static_assert(std::is_same<std::underlying_type_t<M800ViewRole>, std::uint8_t>::value);
    static_assert(kKvNsIdHashSeed == KvNsId{5381});
    static_assert(KV_CLIENT_AGENT_ABI_VERSION == 1U);
    static_assert(KV_CLIENT_AGENT_OK == 0);
    static_assert(KV_CLIENT_AGENT_INVALID_ARGUMENT > 0);
    static_assert(KV_CLIENT_AGENT_ABI_VERSION_MISMATCH > KV_CLIENT_AGENT_INVALID_ARGUMENT);
    static_assert(KV_CLIENT_AGENT_INTERNAL_ERROR > KV_CLIENT_AGENT_EMPTY_VIEW);

    M800ViewServerConfig config;

    EXPECT_EQ(config.role, M800ViewRole::MASTER);
    EXPECT_EQ(config.listenAddr, "0.0.0.0:9100");
    EXPECT_EQ(config.agentFetchTimeoutMs, std::uint64_t{10000});
    EXPECT_EQ(config.httpTimeoutMs, std::uint64_t{10000});
    EXPECT_EQ(config.refreshIntervalMs, std::uint64_t{30000});

    EXPECT_STREQ(M800ViewHTTPContract::kViewsPath, "/v1/views");
    EXPECT_STREQ(M800ViewHTTPContract::kSinceEpochParam, "since_epoch");
    EXPECT_STREQ(M800ViewHTTPContract::kContentType, "text/plain");
    EXPECT_STREQ(M800ViewHTTPContract::kConnection, "close");
    EXPECT_EQ(M800ViewHTTPContract::kStatusOk, 200);
    EXPECT_EQ(M800ViewHTTPContract::kStatusNotModified, 304);
    EXPECT_EQ(M800ViewHTTPContract::kStatusServerError, 500);

    EXPECT_STREQ(M800ViewTextContract::kGlobalEpochField, "globalEpoch");
    EXPECT_STREQ(M800ViewTextContract::kNsCountField, "nsCount");
    EXPECT_STREQ(M800ViewTextContract::kNsField, "ns");
    EXPECT_STREQ(M800ViewTextContract::kViewEpochField, "viewEpoch");
    EXPECT_STREQ(M800ViewTextContract::kViewIdField, "viewId");
    EXPECT_STREQ(M800ViewTextContract::kAsuIdsField, "asuIds");
    EXPECT_STREQ(M800ViewTextContract::kAsuInfoFieldPrefix, "asuInfo.");
}

TEST(M800ViewContractsTest, M800ServerInterfacesForceMultiNamespaceImplementation)
{
    static_assert(std::is_abstract<ViewUniverse>::value);
    static_assert(std::is_abstract<M800ViewMasterServer>::value);
    static_assert(std::is_abstract<M800ViewSlaveServer>::value);
    static_assert(std::is_abstract<RefreshScheduler>::value);
    static_assert(std::is_abstract<M800ViewAgentClient>::value);
    static_assert(std::is_abstract<M800ViewAgentClientFactory>::value);
    static_assert(std::is_abstract<ViewHttpServer>::value);

    static_assert(std::is_same<decltype(&ViewUniverse::PublishAll),
                               Status (ViewUniverse::*)(const ViewUniverse&)>::value);
    static_assert(std::is_same<decltype(&ViewUniverse::ReplaceAll),
                               Status (ViewUniverse::*)(const ViewUniverse&)>::value);
    static_assert(std::is_same<decltype(&ViewUniverse::GetView),
                               Status (ViewUniverse::*)(KvNsId, GlobalView&) const>::value);
    static_assert(std::is_same<decltype(&M800ViewMasterServer::GetAllViews),
                               Status (M800ViewMasterServer::*)(ViewUniverse&)>::value);
    static_assert(std::is_same<decltype(&M800ViewSlaveServer::GetAllViews),
                               Status (M800ViewSlaveServer::*)(ViewUniverse&)>::value);
}

TEST(M800ViewContractsTest, M800AgentAndHTTPContractDefaultsCompile)
{
    const M800AgentNodeView nodeView;
    EXPECT_EQ(nodeView.osdId, std::int32_t{-1});
    EXPECT_FALSE(nodeView.up);
    EXPECT_FALSE(nodeView.in);
    EXPECT_TRUE(nodeView.addrs.empty());

    const ViewHttpRequest request;
    EXPECT_EQ(request.sinceEpoch, std::uint64_t{0});

    const ViewHttpReply reply;
    EXPECT_EQ(reply.kind, ViewHttpReplyKind::INTERNAL_ERROR);
    EXPECT_TRUE(reply.body.empty());
}

}  // namespace
}  // namespace kv::test
