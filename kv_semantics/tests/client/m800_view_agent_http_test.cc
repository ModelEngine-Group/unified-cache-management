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

// 测试场景：未安装 Ceph 代理库时，分别使用空路径和不存在的 libkv_agent.so 路径创建客户端。
// 预期结果：两次 Create 均返回 NOT_FOUND，且不生成客户端实例。
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

// 测试场景：代理动态库路径中间包含 NUL 字符，工厂尝试按该路径创建客户端。
// 预期结果：路径校验拒绝存在截断歧义的输入，Create 返回 INVALID_ARGUMENT。
TEST(M800ViewAgentClientTest, FactoryRejectsEmbeddedNulAgentLibraryPath)
{
    auto factory = CreateM800ViewAgentClientFactory();
    ASSERT_NE(factory, nullptr);
    M800ViewServerConfig config;
    config.agentLibPath = std::string("/tmp/lib\0junk", 13);
    std::unique_ptr<M800ViewAgentClient> client;
    EXPECT_EQ(factory->Create(config, client).code, StatusCode::INVALID_ARGUMENT);
}

// 测试场景：使用伪代理库成功创建客户端，但 monHost 在 127.0.0.1 后嵌入 NUL 和额外字符。
// 预期结果：Initialize 拒绝非法监控地址并返回 INVALID_ARGUMENT。
TEST(M800ViewAgentClientTest, InitializeRejectsEmbeddedNulMonHost)
{
    auto factory = CreateM800ViewAgentClientFactory();
    ASSERT_NE(factory, nullptr);
    M800ViewServerConfig config;
    config.agentLibPath = M800_FAKE_AGENT_PATH;
    config.monHost = std::string("127.0.0.1\0junk", 14);
    config.clientName = "admin";
    config.clientKey = "test-key";
    std::unique_ptr<M800ViewAgentClient> client;
    ASSERT_TRUE(factory->Create(config, client).ok());
    EXPECT_EQ(client->Initialize().code, StatusCode::INVALID_ARGUMENT);
}

// 测试场景：通过伪代理库依次注入返回码 6、4 和 0，查询同一命名空间 pool-a。
// 预期结果：返回码 6 映射为 INTERNAL_ERROR，4 映射为 NOT_FOUND，0 则查询成功且视图列表为空，三种结果保持明确区分。
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

// 测试场景：伪代理库被配置为阻塞 GetInByNs，查询在线程中尚未返回时调用客户端 Stop。
// 预期结果：Stop 唤醒进行中的命名空间查询，查询线程在 Stop 后 1 秒内结束，且查询最终返回成功。
TEST(M800ViewAgentClientTest, StopUnblocksInFlightNamespaceQuery)
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
    ASSERT_TRUE(client->Initialize().ok());
    ASSERT_TRUE(client->Start().ok());

    setenv("M800_FAKE_GET_IN_BLOCK", "1", 1);
    unsetenv("M800_FAKE_GET_IN_UNBLOCK");
    std::vector<M800AgentNodeView> views;
    Status queryStatus;
    std::thread query([&] { queryStatus = client->GetInByNs("pool-a", views); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto begin = std::chrono::steady_clock::now();
    client->Stop();
    query.join();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    EXPECT_TRUE(queryStatus.ok()) << queryStatus.message;
    unsetenv("M800_FAKE_GET_IN_BLOCK");
    unsetenv("M800_FAKE_GET_IN_UNBLOCK");
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

int ConnectLoopback(int port)
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { return -1; }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
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

// 测试场景：先以空处理器启动服务，再发送完整和分片 GET 请求，并覆盖非法或重复 since_epoch、304、处理器异常、错误路径、POST、损坏请求行及重复 Stop。
// 预期结果：空处理器启动返回 INVALID_ARGUMENT；正常 GET 返回 200、Connection: close 和 since=12，分片请求返回 since=13；非法或重复参数按 0 处理；304 无响应体；异常、错误路径、POST 和坏请求分别返回 500、404、405（Allow: GET）和 400，Stop 可重复调用。
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
    EXPECT_NE(response.find("HTTP/1.1 404 Not Found\r\n"), std::string::npos);

    ASSERT_TRUE(RoundTrip(port, {"POST /v1/views HTTP/1.1\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 405 Method Not Allowed\r\n"), std::string::npos);
    EXPECT_NE(response.find("Allow: GET\r\n"), std::string::npos);

    ASSERT_TRUE(RoundTrip(port, {"BROKEN\r\n\r\n"}, response));
    EXPECT_NE(response.find("HTTP/1.1 400 Bad Request\r\n"), std::string::npos);

    server->Stop();
    server->Stop();
}

// 测试场景：处理器为 since_epoch=1 生成超过 4MiB 的响应体，随后客户端发送带约 9KiB 查询参数的超长请求行。
// 预期结果：超大响应体被转换为 500 且不发送响应体，超长请求行被拒绝并返回 400。
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
    EXPECT_NE(response.find("HTTP/1.1 400 Bad Request\r\n"), std::string::npos);
    server->Stop();
}

// 测试场景：HTTP 服务请求超时为 5000ms，已接受的客户端只发送部分请求行后保持连接，随后调用 server->Stop。
// 预期结果：Stop 主动关闭慢客户端连接并在 1 秒内返回，不等待完整的 5000ms 请求超时。
TEST(ViewHttpServerTest, StopShutsDownAcceptedSlowClient)
{
    const auto port = ReserveLoopbackPort();
    auto server = CreateViewHttpServer(5000);
    ASSERT_TRUE(server->Start("127.0.0.1:" + std::to_string(port),
                              [](const ViewHttpRequest&) {
                                  return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW, "ok"};
                              })
                    .ok());
    const int client = ConnectLoopback(port);
    ASSERT_GE(client, 0);
    const std::string partial = "GET /v1/views";
    ASSERT_GT(send(client, partial.data(), partial.size(), MSG_NOSIGNAL), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const auto begin = std::chrono::steady_clock::now();
    server->Stop();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    shutdown(client, SHUT_RDWR);
    close(client);
}

// 测试场景：服务请求超时为 200ms，客户端每隔 50ms 仅发送 1 个字节，始终不组成完整请求。
// 预期结果：逐字节到达不会续延绝对截止时间，服务在 350ms 内主动关闭连接。
TEST(ViewHttpServerTest, SlowDripRequestIsBoundedByAbsoluteDeadline)
{
    const auto port = ReserveLoopbackPort();
    auto server = CreateViewHttpServer(200);
    ASSERT_TRUE(server->Start("127.0.0.1:" + std::to_string(port),
                              [](const ViewHttpRequest&) {
                                  return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW, "ok"};
                              })
                    .ok());
    const int client = ConnectLoopback(port);
    ASSERT_GE(client, 0);
    const auto begin = std::chrono::steady_clock::now();
    bool closedByServer{false};
    for (int index = 0; index < 8 && !closedByServer; ++index) {
        const char byte = 'x';
        const auto sent = send(client, &byte, 1, MSG_NOSIGNAL);
        if (sent < 0) {
            closedByServer = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        char response[64];
        const auto received = recv(client, response, sizeof(response), MSG_DONTWAIT);
        if (received == 0) { closedByServer = true; }
    }
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    EXPECT_TRUE(closedByServer);
    EXPECT_LT(elapsed, std::chrono::milliseconds(350));
    shutdown(client, SHUT_RDWR);
    close(client);
    server->Stop();
}

// 测试场景：连续 5 轮先停止仍持有半请求客户端连接的服务，再在同一回环端口创建并启动新服务处理 since_epoch=1 请求。
// 预期结果：每轮端口都能立即重新监听，新客户端收到 200 和正文 since=1，不受上一轮连接或 Stop 状态影响。
TEST(ViewHttpServerTest, StopThenRestartServesNewClient)
{
    for (int iteration = 0; iteration < 5; ++iteration) {
        const auto port = ReserveLoopbackPort();
        ASSERT_GT(port, 0);
        const auto handler = [](const ViewHttpRequest& request) {
            return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW,
                                 "since=" + std::to_string(request.sinceEpoch)};
        };
        auto first = CreateViewHttpServer(1000);
        ASSERT_TRUE(first->Start("127.0.0.1:" + std::to_string(port), handler).ok());
        const int client = ConnectLoopback(port);
        ASSERT_GE(client, 0);
        const std::string partial = "GET /v1/views";
        ASSERT_GT(send(client, partial.data(), partial.size(), MSG_NOSIGNAL), 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        first->Stop();
        shutdown(client, SHUT_RDWR);
        close(client);

        auto second = CreateViewHttpServer(1000);
        ASSERT_TRUE(second->Start("127.0.0.1:" + std::to_string(port), handler).ok());
        std::string response;
        ASSERT_TRUE(RoundTrip(port, {"GET /v1/views?since_epoch=1 HTTP/1.1\r\n\r\n"}, response));
        EXPECT_NE(response.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
        EXPECT_EQ(ResponseBody(response), "since=1");
        second->Stop();
    }
}

// 测试场景：分别使用端口号 0，以及在 127.0.0.1 后嵌入 NUL 和额外字符的监听地址启动 HTTP 服务。
// 预期结果：两种非法监听地址均在 Start 参数校验阶段被拒绝，并返回 INVALID_ARGUMENT。
TEST(ViewHttpServerTest, RejectsZeroPortAndEmbeddedNulListenAddress)
{
    auto server = CreateViewHttpServer(100);
    const auto handler = [](const ViewHttpRequest&) {
        return ViewHttpReply{ViewHttpReplyKind::FULL_VIEW, "ok"};
    };
    EXPECT_EQ(server->Start("127.0.0.1:0", handler).code, StatusCode::INVALID_ARGUMENT);
    const std::string embeddedNul("127.0.0.1\0junk:9100", 19);
    EXPECT_EQ(server->Start(embeddedNul, handler).code, StatusCode::INVALID_ARGUMENT);
}

// 测试场景：已有一个 socket 在回环地址上完成 bind 和 listen 后，再让 HTTP 服务绑定同一端口。
// 预期结果：服务不会复用已占用端口，Start 返回 CONNECTION_ERROR。
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
