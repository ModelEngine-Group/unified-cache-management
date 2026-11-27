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
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "view/m800_view_slave_server.h"

namespace kv::test {
namespace {

class RawResponseServer {
public:
    explicit RawResponseServer(std::string response, int responseDelayMs = 0,
                               std::size_t fragmentAt = std::string::npos)
        : response_(std::move(response)), responseDelayMs_(responseDelayMs), fragmentAt_(fragmentAt)
    {
        listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd_ < 0) { throw std::runtime_error("failed to create raw server socket"); }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (bind(listenFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            listen(listenFd_, 1) != 0) {
            close(listenFd_);
            throw std::runtime_error("failed to bind raw server");
        }
        socklen_t size = sizeof(address);
        if (getsockname(listenFd_, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
            close(listenFd_);
            throw std::runtime_error("failed to get raw server port");
        }
        port_ = ntohs(address.sin_port);
        worker_ = std::thread([this] { ServeOne(); });
    }

    ~RawResponseServer()
    {
        shutdown(listenFd_, SHUT_RDWR);
        close(listenFd_);
        if (worker_.joinable()) { worker_.join(); }
    }

    std::uint16_t Port() const { return port_; }

private:
    void ServeOne()
    {
        const int client = accept(listenFd_, nullptr, nullptr);
        if (client < 0) { return; }
        char request[2048];
        recv(client, request, sizeof(request), 0);
        if (responseDelayMs_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(responseDelayMs_));
        }
        auto sendPart = [&](const std::string& part) {
            std::size_t sent{0};
            while (sent < part.size()) {
                const auto result =
                    send(client, part.data() + sent, part.size() - sent, MSG_NOSIGNAL);
                if (result <= 0) { return false; }
                sent += static_cast<std::size_t>(result);
            }
            return true;
        };
        if (fragmentAt_ != std::string::npos && fragmentAt_ < response_.size()) {
            if (sendPart(response_.substr(0, fragmentAt_))) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                sendPart(response_.substr(fragmentAt_));
            }
        } else {
            sendPart(response_);
        }
        shutdown(client, SHUT_RDWR);
        close(client);
    }

    int listenFd_{-1};
    std::uint16_t port_{0};
    std::string response_;
    int responseDelayMs_{0};
    std::size_t fragmentAt_{std::string::npos};
    std::thread worker_;
};

std::uint64_t MaxViewEpoch(const std::unordered_map<KvNsId, GlobalView>& views)
{
    std::uint64_t epoch{0};
    for (const auto& item : views) { epoch = std::max(epoch, item.second.viewEpoch); }
    return epoch;
}

std::string ValidText(std::uint64_t epoch)
{
    const auto value = std::to_string(epoch);
    return "globalEpoch=" + value + "\nnsCount=1\nns=1\nviewEpoch=" + value +
           "\nviewId=1\nasuIds=1\n"
           "asuInfo.1=protocol=uboe,port=6000,local.comm_id=10.0.0.1\n";
}

// 测试场景：从节点在已有 globalEpoch=8、仅含命名空间 9 的 Universe 上拉取视图，主节点依次返回
// 304、畸形状态行、chunked、声明长度超过实际正文和 Content-Length 截断有效正文的响应。
// 预期结果：304、畸形状态行、chunked 和正文被截断均返回 INTERNAL_ERROR，正文不足声明长度返回
// CONNECTION_ERROR；每次失败后 globalEpoch 仍为 8，命名空间数量仍为 1。
TEST(M800ViewSlaveHttpNegativeTest, RejectsMalformedResponsesWithoutChangingUniverse)
{
    const auto validBody = ValidText(10);
    const std::vector<std::pair<std::string, StatusCode>> cases = {
        {"HTTP/1.1 304 Not Modified\r\nConnection: close\r\n\r\n",         StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 200OK\r\nContent-Length: 0\r\n\r\n",                    StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
         StatusCode::INTERNAL_ERROR                                                                  },
        {"HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(validBody.size() + 10) +
             "\r\n\r\n" + validBody,
         StatusCode::CONNECTION_ERROR                                                                },
        {"HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(validBody.size() - 1) + "\r\n\r\n" +
             validBody,                                           StatusCode::INTERNAL_ERROR},
    };

    for (const auto& [response, expectedCode] : cases) {
        RawResponseServer server(response);
        M800ViewServerConfig config;
        config.role = M800ViewRole::SLAVE;
        config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
        config.httpTimeoutMs = 300;
        auto worker = CreateM800ViewSlaveServer(config);
        GlobalView initial;
        initial.viewEpoch = 8;
        initial.viewId = 9;
        std::unordered_map<KvNsId, GlobalView> universe{
            {9, initial}
        };
        const auto status = worker->GetAllViews(universe);
        EXPECT_EQ(status.code, expectedCode) << response;
        EXPECT_EQ(MaxViewEpoch(universe), 8) << response;
        EXPECT_EQ(universe.size(), 1) << response;
        worker->Stop();
    }
}

// 测试场景：主节点返回 HTTP 500，分别不携带 Content-Length、声明 4 字节却只发送 3 字节，以及声明 3
// 字节却额外发送 1 字节。 预期结果：无 Content-Length 和正文超出声明长度时返回
// INTERNAL_ERROR，正文不足时返回 CONNECTION_ERROR；三种响应均不会向空 Universe 写入视图。
TEST(M800ViewSlaveHttpNegativeTest, RejectsNon200FramingErrors)
{
    const std::vector<std::pair<std::string, StatusCode>> cases = {
        {"HTTP/1.1 500 Internal Server Error\r\n\r\n",                          StatusCode::INTERNAL_ERROR},
        {"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\n\r\nabc",
         StatusCode::CONNECTION_ERROR                                                                     },
        {"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 3\r\n\r\nabcd",
         StatusCode::INTERNAL_ERROR                                                                       },
    };
    for (const auto& [response, expectedCode] : cases) {
        RawResponseServer server(response);
        M800ViewServerConfig config;
        config.role = M800ViewRole::SLAVE;
        config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
        config.httpTimeoutMs = 300;
        auto worker = CreateM800ViewSlaveServer(config);
        std::unordered_map<KvNsId, GlobalView> universe;
        EXPECT_EQ(worker->GetAllViews(universe).code, expectedCode) << response;
        EXPECT_TRUE(universe.empty());
        worker->Stop();
    }
}

// 测试场景：主节点以 200 返回合法视图正文，Content-Length
// 恰好等于正文长度，但分片发送时在该长度之后额外附加字符 X。 预期结果：从节点拒绝 Content-Length
// 之外的多余字节，GetAllViews 返回 INTERNAL_ERROR，Universe 保持为空。
TEST(M800ViewSlaveHttpNegativeTest, RejectsBodyBytesBeyondContentLength)
{
    const auto validBody = ValidText(10);
    const std::string header =
        "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(validBody.size()) +
        "\r\nConnection: close\r\n\r\n";
    RawResponseServer server(header + validBody + "X", 0, header.size() + 2);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
    config.httpTimeoutMs = 300;
    auto worker = CreateM800ViewSlaveServer(config);
    std::unordered_map<KvNsId, GlobalView> universe;
    EXPECT_EQ(worker->GetAllViews(universe).code, StatusCode::INTERNAL_ERROR);
    EXPECT_TRUE(universe.empty());
}

// 测试场景：主节点延迟 500ms 且不发送任何响应，从节点的 HTTP 超时为
// 2000ms；拉取线程阻塞读取期间调用 Stop。 预期结果：Stop 主动中断阻塞读，拉取线程在 1
// 秒内退出并返回 CANCELED，而不是等待响应或 HTTP 超时。
TEST(M800ViewSlaveHttpNegativeTest, StopInterruptsBlockedRead)
{
    RawResponseServer server("", 500);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(server.Port())};
    config.httpTimeoutMs = 2000;
    auto worker = CreateM800ViewSlaveServer(config);
    std::unordered_map<KvNsId, GlobalView> universe;
    Status fetchStatus;
    std::thread fetcher([&] { fetchStatus = worker->GetAllViews(universe); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const auto begin = std::chrono::steady_clock::now();
    worker->Stop();
    fetcher.join();
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(1));
    EXPECT_EQ(fetchStatus.code, StatusCode::CANCELED);
}

// 测试场景：连续 10 轮先在慢主节点连接仍未响应时 Stop 旧 worker 并等待拉取线程退出，再创建新 worker
// 连接返回合法 epoch 10 视图的主节点。 预期结果：每轮旧连接都能完成停止和线程回收，新 worker
// 均可独立建立连接、成功拉取视图，并将 globalEpoch 更新为 10。
TEST(M800ViewSlaveHttpNegativeTest, StopThenNewConnectionWorks)
{
    for (int iteration = 0; iteration < 10; ++iteration) {
        {
            RawResponseServer slowServer("", 200);
            M800ViewServerConfig config;
            config.role = M800ViewRole::SLAVE;
            config.masterAddrs = {"127.0.0.1:" + std::to_string(slowServer.Port())};
            config.httpTimeoutMs = 1000;
            auto worker = CreateM800ViewSlaveServer(config);
            std::unordered_map<KvNsId, GlobalView> universe;
            std::thread fetcher([&] { (void)worker->GetAllViews(universe); });
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            worker->Stop();
            fetcher.join();
        }
        RawResponseServer validServer(
            "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(ValidText(10).size()) +
            "\r\nConnection: close\r\n\r\n" + ValidText(10));
        M800ViewServerConfig config;
        config.role = M800ViewRole::SLAVE;
        config.masterAddrs = {"127.0.0.1:" + std::to_string(validServer.Port())};
        config.httpTimeoutMs = 300;
        auto worker = CreateM800ViewSlaveServer(config);
        std::unordered_map<KvNsId, GlobalView> universe;
        ASSERT_TRUE(worker->GetAllViews(universe).ok()) << "iteration=" << iteration;
        EXPECT_EQ(MaxViewEpoch(universe), 10);
        worker->Stop();
    }
}

// 测试场景：配置两个各延迟 500ms 才响应的主节点地址，从节点整体 HTTP 超时仅为
// 100ms，并依次尝试两个地址。 预期结果：两个主节点共用同一绝对截止时间，GetAllViews 返回
// TIMEOUT，总耗时小于 180ms，而非为每个地址分别等待 100ms。
TEST(M800ViewSlaveHttpNegativeTest, AllMasterAddressesShareOneDeadline)
{
    RawResponseServer first("", 500);
    RawResponseServer second("", 500);
    M800ViewServerConfig config;
    config.role = M800ViewRole::SLAVE;
    config.masterAddrs = {"127.0.0.1:" + std::to_string(first.Port()),
                          "127.0.0.1:" + std::to_string(second.Port())};
    config.httpTimeoutMs = 100;
    auto worker = CreateM800ViewSlaveServer(config);
    std::unordered_map<KvNsId, GlobalView> universe;
    const auto begin = std::chrono::steady_clock::now();
    const auto status = worker->GetAllViews(universe);
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    EXPECT_EQ(status.code, StatusCode::TIMEOUT);
    EXPECT_LT(elapsed, std::chrono::milliseconds(180));
    worker->Stop();
}

}  // namespace
}  // namespace kv::test
