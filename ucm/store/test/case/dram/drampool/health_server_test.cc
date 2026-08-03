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
#include "health_server.h"
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include "drampool_config.h"

namespace UC::DramPool {
namespace {

constexpr char kLoopback[] = "127.0.0.1";

class ScopedHealthConfig final {
public:
    explicit ScopedHealthConfig(std::uint16_t port) : previous_(g_config)
    {
        g_config.addr.host = kLoopback;
        g_config.healthPort = port;
    }

    ~ScopedHealthConfig() { g_config = std::move(previous_); }

    ScopedHealthConfig(const ScopedHealthConfig&) = delete;
    ScopedHealthConfig& operator=(const ScopedHealthConfig&) = delete;

private:
    DramPoolConfig previous_;
};

class Socket final {
public:
    explicit Socket(int socket) : socket_(socket) {}
    ~Socket()
    {
        if (socket_ >= 0) { ::close(socket_); }
    }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept : socket_(other.socket_) { other.socket_ = -1; }

    int Get() const { return socket_; }

private:
    int socket_;
};

sockaddr_in LoopbackAddress(std::uint16_t port)
{
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return address;
}

std::uint16_t ReservePort()
{
    Socket socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    EXPECT_GE(socket.Get(), 0);

    auto address = LoopbackAddress(0);
    EXPECT_EQ(::bind(socket.Get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);

    socklen_t length = sizeof(address);
    EXPECT_EQ(::getsockname(socket.Get(), reinterpret_cast<sockaddr*>(&address), &length), 0);
    return ntohs(address.sin_port);
}

Socket Connect(std::uint16_t port)
{
    Socket socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    EXPECT_GE(socket.Get(), 0);

    auto address = LoopbackAddress(port);
    EXPECT_EQ(::connect(socket.Get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    return socket;
}

void Send(int socket, std::string_view data)
{
    while (!data.empty()) {
        const auto sent = ::send(socket, data.data(), data.size(), MSG_NOSIGNAL);
        ASSERT_GT(sent, 0);
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
}

std::string ReceiveUntilClosed(int socket)
{
    timeval timeout{2, 0};
    EXPECT_EQ(::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);

    std::string response;
    char buffer[256];
    while (true) {
        const auto received = ::recv(socket, buffer, sizeof(buffer), 0);
        if (received <= 0) { return response; }
        response.append(buffer, static_cast<std::size_t>(received));
    }
}

std::string Request(std::uint16_t port, std::string_view request)
{
    auto socket = Connect(port);
    Send(socket.Get(), request);
    return ReceiveUntilClosed(socket.Get());
}

TEST(HealthServerTest, PortZeroDisablesServer)
{
    ScopedHealthConfig config(0);
    HealthServer server;

    EXPECT_TRUE(server.Start().Success());
    server.Stop();
}

TEST(HealthServerTest, ServesHttp10AndHttp11HealthRequests)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());

    for (const auto* version : {"HTTP/1.0", "HTTP/1.1"}) {
        const auto response =
            Request(port, std::string{"GET /health "} + version + "\r\nHost: localhost\r\n\r\n");
        EXPECT_EQ(response,
                  "HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/plain\r\n"
                  "Content-Length: 2\r\n"
                  "Connection: close\r\n"
                  "\r\n"
                  "OK");
    }
}

TEST(HealthServerTest, AcceptsFragmentedRequest)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());
    auto socket = Connect(port);

    Send(socket.Get(), "GET /health HTTP/1.1\r\n");
    Send(socket.Get(), "Host: localhost\r\n\r\n");

    EXPECT_NE(ReceiveUntilClosed(socket.Get()).find("200 OK"), std::string::npos);
}

TEST(HealthServerTest, ClosesUnsupportedRequestsWithoutResponse)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());

    EXPECT_TRUE(Request(port, "GET /missing HTTP/1.1\r\n\r\n").empty());
    EXPECT_TRUE(Request(port, "POST /health HTTP/1.1\r\n\r\n").empty());
    EXPECT_TRUE(Request(port, "GET /health HTTP/2\r\n\r\n").empty());
}

TEST(HealthServerTest, TimesOutIncompleteRequest)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());

    const auto begin = std::chrono::steady_clock::now();
    EXPECT_TRUE(Request(port, "GET /health HTTP/1.1\r\n").empty());
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    EXPECT_GE(elapsed, std::chrono::milliseconds(900));
    EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST(HealthServerTest, RejectsOversizedRequest)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());

    std::string request = "GET /health HTTP/1.1\r\nX-Padding: ";
    request.append(4096, 'x');
    EXPECT_TRUE(Request(port, request).empty());
}

TEST(HealthServerTest, StopInterruptsIdleClientAndCanBeRepeated)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer server;
    ASSERT_TRUE(server.Start().Success());
    auto socket = Connect(port);

    const auto begin = std::chrono::steady_clock::now();
    server.Stop();
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    server.Stop();

    EXPECT_LT(elapsed, std::chrono::milliseconds(500));
}

TEST(HealthServerTest, RejectsOccupiedPortAndDuplicateStart)
{
    const auto port = ReservePort();
    ScopedHealthConfig config(port);
    HealthServer first;
    ASSERT_TRUE(first.Start().Success());

    EXPECT_TRUE(first.Start().Failure());
    HealthServer second;
    EXPECT_TRUE(second.Start().Failure());
}

}  // namespace
}  // namespace UC::DramPool
