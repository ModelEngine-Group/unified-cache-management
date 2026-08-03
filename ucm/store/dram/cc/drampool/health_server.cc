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
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <exception>
#include <netdb.h>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <unistd.h>
#include "drampool_config.h"
#include "logger/logger.h"

namespace UC::DramPool {
namespace {

constexpr int kListenBacklog = 16;
constexpr std::size_t kMaxRequestBytes = 4096;
constexpr std::chrono::steady_clock::duration kIoPollInterval = std::chrono::milliseconds(100);
constexpr auto kClientRequestTimeout = std::chrono::seconds(1);
constexpr std::string_view kHeaderTerminator = "\r\n\r\n";
constexpr std::string_view kHealthResponse =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/plain\r\n"
    "Content-Length: 2\r\n"
    "Connection: close\r\n"
    "\r\n"
    "OK";

int PollTimeout(std::chrono::steady_clock::duration duration)
{
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

int CreateListenSocket(const std::string& ip, std::uint16_t port)
{
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* addresses = nullptr;
    const auto service = std::to_string(port);
    if (::getaddrinfo(ip.c_str(), service.c_str(), &hints, &addresses) != 0) { return -1; }

    int listenSocket = -1;
    for (auto* address = addresses; address != nullptr; address = address->ai_next) {
        const int socket = ::socket(address->ai_family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK,
                                    address->ai_protocol);
        if (socket < 0) { continue; }

        int reuseAddress = 1;
        ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuseAddress, sizeof(reuseAddress));
        if (::bind(socket, address->ai_addr, address->ai_addrlen) == 0 &&
            ::listen(socket, kListenBacklog) == 0) {
            listenSocket = socket;
            break;
        }
        ::close(socket);
    }

    ::freeaddrinfo(addresses);
    return listenSocket;
}

bool IsHealthRequest(std::string_view request)
{
    const auto lineEnd = request.find("\r\n");
    const auto requestLine = request.substr(0, lineEnd);
    return requestLine == "GET /health HTTP/1.1" || requestLine == "GET /health HTTP/1.0";
}

bool WaitUntilWritable(int socket, std::chrono::steady_clock::time_point deadline,
                       const std::atomic_bool& stopping)
{
    while (!stopping.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) { return false; }

        pollfd event{socket, POLLOUT, 0};
        const auto wait = std::min(kIoPollInterval, deadline - now);
        const int ready = ::poll(&event, 1, PollTimeout(wait));
        if (ready > 0) { return (event.revents & POLLOUT) != 0; }
        if (ready < 0 && errno != EINTR) { return false; }
    }
    return false;
}

void SendHealthResponse(int socket, const std::atomic_bool& stopping)
{
    std::string_view remaining = kHealthResponse;
    const auto deadline = std::chrono::steady_clock::now() + kClientRequestTimeout;
    while (!remaining.empty()) {
        const auto sent = ::send(socket, remaining.data(), remaining.size(), MSG_NOSIGNAL);
        if (sent > 0) {
            remaining.remove_prefix(static_cast<std::size_t>(sent));
            continue;
        }
        if (sent < 0 && errno == EINTR) { continue; }
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) &&
            WaitUntilWritable(socket, deadline, stopping)) {
            continue;
        }
        return;
    }
}

}  // namespace

HealthServer::~HealthServer() { Stop(); }

Status HealthServer::Start()
{
    if (g_config.healthPort == 0) {
        UC_INFO_UNLIMITED("DramPool health server is disabled");
        return Status::OK();
    }
    const int socket = CreateListenSocket(g_config.addr.host, g_config.healthPort);
    if (socket < 0) { return Status::OsApiError("failed to listen on DramPool health endpoint"); }

    listenSocket_ = socket;
    stopping_.store(false, std::memory_order_release);
    try {
        worker_ = std::thread(&HealthServer::Run, this);
    } catch (const std::exception& error) {
        stopping_.store(true, std::memory_order_release);
        ::close(listenSocket_);
        listenSocket_ = -1;
        return Status::Error(std::string{"failed to start HealthServer: "} + error.what());
    }
    UC_INFO_UNLIMITED("DramPool health server started, addr={}:{}", g_config.addr.host,
                      g_config.healthPort);
    return Status::OK();
}

void HealthServer::Stop() noexcept
{
    if (!worker_.joinable()) { return; }

    stopping_.store(true, std::memory_order_release);
    worker_.join();
    ::close(listenSocket_);
    listenSocket_ = -1;
}

void HealthServer::Run() noexcept
{
    while (!stopping_.load(std::memory_order_acquire)) {
        pollfd event{listenSocket_, POLLIN, 0};
        const int ready = ::poll(&event, 1, PollTimeout(kIoPollInterval));
        if (ready < 0) {
            if (errno == EINTR) { continue; }
            UC_ERROR_UNLIMITED("HealthServer poll failed, errno={}", errno);
            break;
        }
        if (ready == 0) { continue; }
        if ((event.revents & POLLIN) == 0) {
            UC_ERROR_UNLIMITED("HealthServer listen socket failed, events={}", event.revents);
            break;
        }

        while (!stopping_.load(std::memory_order_acquire)) {
            const int client =
                ::accept4(listenSocket_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (client < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) { break; }
                if (errno == EINTR) { continue; }
                UC_WARN("HealthServer accept failed, errno={}", errno);
                break;
            }
            HandleClient(client);
            ::close(client);
        }
    }
}

void HealthServer::HandleClient(int clientSocket) noexcept
{
    std::string request;
    request.reserve(kMaxRequestBytes);
    const auto deadline = std::chrono::steady_clock::now() + kClientRequestTimeout;

    while (request.find(kHeaderTerminator) == std::string::npos) {
        if (stopping_.load(std::memory_order_acquire) || request.size() == kMaxRequestBytes) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) { return; }

        pollfd event{clientSocket, POLLIN, 0};
        const auto wait = std::min(kIoPollInterval, deadline - now);
        const int ready = ::poll(&event, 1, PollTimeout(wait));
        if (ready < 0) {
            if (errno == EINTR) { continue; }
            return;
        }
        if (ready == 0) { continue; }
        if ((event.revents & POLLIN) == 0) { return; }

        char buffer[1024];
        const auto capacity = std::min(sizeof(buffer), kMaxRequestBytes - request.size());
        const auto received = ::recv(clientSocket, buffer, capacity, 0);
        if (received > 0) {
            request.append(buffer, static_cast<std::size_t>(received));
            continue;
        }
        if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        return;
    }

    if (IsHealthRequest(request)) { SendHealthResponse(clientSocket, stopping_); }
}

}  // namespace UC::DramPool
