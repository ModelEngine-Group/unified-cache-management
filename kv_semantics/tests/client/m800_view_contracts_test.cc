#include <gtest/gtest.h>
#include <type_traits>
#include "KVClientAgentABI.h"
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

// 测试场景：编译并检查 M800 角色枚举、哈希种子、Agent ABI 版本与状态码，以及默认服务配置、HTTP 和文本协议常量。
// 预期结果：角色值依次为 0/1 且底层类型为 uint8_t，哈希种子为 5381、ABI 版本为 1，状态码顺序、默认配置及 HTTP/文本协议常量均符合固定契约。
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
    EXPECT_EQ(config.httpTimeoutMs, std::uint64_t{2000});
    EXPECT_EQ(config.refreshIntervalMs, std::uint64_t{30000});

    EXPECT_STREQ(M800ViewHttpContract::kViewsPath, "/v1/views");
    EXPECT_STREQ(M800ViewHttpContract::kSinceEpochParam, "since_epoch");
    EXPECT_STREQ(M800ViewHttpContract::kContentType, "text/plain");
    EXPECT_STREQ(M800ViewHttpContract::kConnection, "close");
    EXPECT_EQ(M800ViewHttpContract::kStatusOk, 200);
    EXPECT_EQ(M800ViewHttpContract::kStatusNotModified, 304);
    EXPECT_EQ(M800ViewHttpContract::kStatusBadRequest, 400);
    EXPECT_EQ(M800ViewHttpContract::kStatusNotFound, 404);
    EXPECT_EQ(M800ViewHttpContract::kStatusMethodNotAllowed, 405);
    EXPECT_EQ(M800ViewHttpContract::kStatusServerError, 500);

    EXPECT_STREQ(M800ViewTextContract::kGlobalEpochField, "globalEpoch");
    EXPECT_STREQ(M800ViewTextContract::kNsCountField, "nsCount");
    EXPECT_STREQ(M800ViewTextContract::kNsField, "ns");
    EXPECT_STREQ(M800ViewTextContract::kViewEpochField, "viewEpoch");
    EXPECT_STREQ(M800ViewTextContract::kViewIdField, "viewId");
    EXPECT_STREQ(M800ViewTextContract::kAsuIdsField, "asuIds");
    EXPECT_STREQ(M800ViewTextContract::kAsuInfoFieldPrefix, "asuInfo.");
}

// 测试场景：通过编译期类型检查审视 ViewUniverse、Master/Slave Server、RefreshScheduler、AgentClient/Factory 和 ViewHttpServer 的接口及多命名空间方法签名。
// 预期结果：这些类型均为抽象类，PublishAll、ReplaceAll、GetView 与 Master/Slave 的 GetAllViews 签名严格匹配 ViewUniverse 多命名空间契约。
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

// 测试场景：默认构造 M800AgentNodeView、ViewHttpRequest 和 ViewHttpReply。
// 预期结果：节点 osdId 为 -1、up/in 均为 false 且地址为空；请求 sinceEpoch 为 0；响应类型为 INTERNAL_ERROR 且 body 为空。
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
