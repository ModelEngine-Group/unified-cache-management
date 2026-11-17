#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include "KVClientAgentABI.h"
#include "view/m800_view_agent_client.h"
#include "view/view_http_server.h"

namespace kv::test {
namespace {

using GetInByNsAbi = int32_t (*)(kv_client_agent_handle*, const char*,
                                 kv_client_agent_node_view**, size_t*);
static_assert(std::is_same<decltype(&kv_client_agent_get_in_by_ns), GetInByNsAbi>::value,
              "the twelfth ABI entry must retain its C signature");
static_assert(KV_CLIENT_AGENT_ABI_VERSION == 1U, "the M800 ABI must remain v1");

TEST(M800ViewAgentClientTest, FactoryRejectsMissingLibrariesWithoutCeph)
{
    auto factory = CreateM800ViewAgentClientFactory();
    ASSERT_NE(factory, nullptr);

    M800ViewServerConfig config;
    config.agentLibPath.clear();
    std::unique_ptr<M800ViewAgentClient> client;
    auto status = factory->Create(config, client);
    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    EXPECT_EQ(client, nullptr);

    config.agentLibPath = "/definitely/not/a/real/libkv_agent.so";
    status = factory->Create(config, client);
    EXPECT_EQ(status.code, StatusCode::NOT_FOUND);
    EXPECT_EQ(client, nullptr);
}

TEST(M800ViewAgentClientTest, FakeLibraryPreservesInternalErrorAndEmptyViewDistinction)
{
    auto factory = CreateM800ViewAgentClientFactory();
    ASSERT_NE(factory, nullptr);
    M800ViewServerConfig config;
    config.agentLibPath = M800_FAKE_AGENT_PATH;
    config.monHost = "127.0.0.1:6789";
    config.clientName = "admin";
    config.clientKey = "test-key";
    std::unique_ptr<M800ViewAgentClient> client;
    ASSERT_TRUE(factory->Create(config, client).ok());
    ASSERT_NE(client, nullptr);
    ASSERT_TRUE(client->Initialize().ok());
    ASSERT_TRUE(client->Start().ok());

    std::vector<M800AgentNodeView> views;
    setenv("M800_FAKE_GET_IN_CODE", "6", 1);
    EXPECT_EQ(client->GetInByNs("pool-a", views).code, StatusCode::INTERNAL_ERROR);
    setenv("M800_FAKE_GET_IN_CODE", "4", 1);
    EXPECT_EQ(client->GetInByNs("pool-a", views).code, StatusCode::NOT_FOUND);
    setenv("M800_FAKE_GET_IN_CODE", "0", 1);
    EXPECT_TRUE(client->GetInByNs("pool-a", views).ok());
    EXPECT_TRUE(views.empty());
    unsetenv("M800_FAKE_GET_IN_CODE");
    client->Stop();
}

int ReserveLoopbackPort()
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return -1; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
        close(fd);
        return -1;
    }
    const int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

bool SendAll(int fd, const std::string& data)
{
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto sent = send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
        if (sent > 0) {
            offset += static_cast<std::size_t>(sent);
        } else if (sent < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool RoundTrip(int port, const std::vector<std::string>& requestParts, std::string& response)
{
    response.clear();
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return false; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        return false;
    }
    for (const auto& part : requestParts) {
        if (!SendAll(fd, part)) {
            close(fd);
            return false;
        }
        if (requestParts.size() > 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    char buffer[4096];
    while (true) {
        const auto received = recv(fd, buffer, sizeof(buffer), 0);
        if (received > 0) {
            response.append(buffer, static_cast<std::size_t>(received));
        } else if (received < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    close(fd);
    return true;
}

std::string ResponseBody(const std::string& response)
{
    const auto split = response.find("\r\n\r\n");
    return split == std::string::npos ? std::string{} : response.substr(split + 4);
}

TEST(ViewHttpServerTest, ServesViewsAndHandlesFragmentedAndInvalidRequests)
{
    const int port = ReserveLoopbackPort();
    ASSERT_GT(port, 0);
    auto server = CreateViewHttpServer();
    ASSERT_NE(server, nullptr);
    const auto address = "127.0.0.1:" + std::to_string(port);
    EXPECT_EQ(server->Start(address, ViewHttpHandler{}).code, StatusCode::INVALID_ARGUMENT);

    const auto handler = [](const ViewHttpRequest& request) {
        if (request.sinceEpoch == 7) {
            ViewHttpReply reply;
            reply.kind = ViewHttpReplyKind::NOT_MODIFIED;
            reply.body = "must not be sent";
            return reply;
        }
        if (request.sinceEpoch == 42) { throw std::runtime_error{"injected handler failure"}; }
        ViewHttpReply reply;
        reply.kind = ViewHttpReplyKind::FULL_VIEW;
        reply.body = "since=" + std::to_string(request.sinceEpoch);
        return reply;
    };
    ASSERT_TRUE(server->Start(address, handler).ok());

    std::string response;
    ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=12 HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
    EXPECT_NE(response.find("Connection: close\r\n"), std::string::npos);
    EXPECT_EQ(ResponseBody(response), "since=12");

    ASSERT_TRUE(RoundTrip(port,
                          {"GET /v1/vi", "ews?since_epoch=13 HTTP/1.1\r\nHost: local\r\n",
                           "\r\n"},
                          response));
    EXPECT_EQ(ResponseBody(response), "since=13");

    ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=bad HTTP/1.1\r\n\r\n"}, response));
    EXPECT_EQ(ResponseBody(response), "since=0");

    ASSERT_TRUE(RoundTrip(
        port, {"GET /v1/views?since_epoch=8&since_epoch=9 HTTP/1.1\r\n\r\n"}, response));
    EXPECT_EQ(ResponseBody(response), "since=0");

    ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=7 HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 304 Not Modified\r\n"), std::string::npos);
    EXPECT_NE(response.find("Content-Length: 0\r\n"), std::string::npos);
    EXPECT_TRUE(ResponseBody(response).empty());

    ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=42 HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 500 Internal Server Error\r\n"), std::string::npos);

    ASSERT_TRUE(RoundTrip(port, {"GET /not/views HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 500 Internal Server Error\r\n"), std::string::npos);

    ASSERT_TRUE(RoundTrip(port, {"POST /v1/views HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 500 Internal Server Error\r\n"), std::string::npos);

    server->Stop();
    server->Stop();
}

TEST(ViewHttpServerTest, EnforcesRequestLineAndResponseBodyLimits)
{
    const int port = ReserveLoopbackPort();
    ASSERT_GT(port, 0);
    auto server = CreateViewHttpServer(1000);
    ASSERT_NE(server, nullptr);
    const auto handler = [](const ViewHttpRequest& request) {
        ViewHttpReply reply;
        reply.kind = ViewHttpReplyKind::FULL_VIEW;
        reply.body = request.sinceEpoch == 1 ? std::string(4 * 1024 * 1024 + 1, 'x') : "ok";
        return reply;
    };
    ASSERT_TRUE(server->Start("127.0.0.1:" + std::to_string(port), handler).ok());

    std::string response;
    ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=1 HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 500 Internal Server Error\r\n"), std::string::npos);
    EXPECT_TRUE(ResponseBody(response).empty());

    const auto oversizedRequest = "GET /v1/views?ignored=" + std::string(9 * 1024, 'a') +
                                  " HTTP/1.1\r\n\r\n";
    ASSERT_TRUE(RoundTrip(port, {oversizedRequest}, response));
    EXPECT_NE(response.find("HTTP/1.1 500 Internal Server Error\r\n"), std::string::npos);
    server->Stop();
}

TEST(ViewHttpServerTest, OccupiedPortFailsWithoutReusePort)
{
    const int blocker = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(blocker, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    ASSERT_EQ(bind(blocker, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    ASSERT_EQ(listen(blocker, 1), 0);
    socklen_t length = sizeof(address);
    ASSERT_EQ(getsockname(blocker, reinterpret_cast<sockaddr*>(&address), &length), 0);

    auto server = CreateViewHttpServer();
    ASSERT_NE(server, nullptr);
    const auto handler = [](const ViewHttpRequest&) { return ViewHttpReply{}; };
    const auto status = server->Start("127.0.0.1:" + std::to_string(ntohs(address.sin_port)),
                                      handler);
    EXPECT_EQ(status.code, StatusCode::CONNECTION_ERROR);
    server->Stop();
    close(blocker);
}

}  // namespace
}  // namespace kv::test
