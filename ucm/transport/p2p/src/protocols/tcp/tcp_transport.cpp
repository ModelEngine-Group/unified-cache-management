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
 */

#include "protocols/tcp/tcp_transport.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include "logger/logger.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace transport {
namespace {

constexpr int kListenBacklog = 128;

bool WaitConnected(int socket)
{
    pollfd descriptor{socket, POLLOUT, 0};
    int ready = 0;
    do {
        ready = poll(&descriptor, 1, -1);
    } while (ready < 0 && errno == EINTR);
    if (ready <= 0) { return false; }
    int error = 0;
    socklen_t length = sizeof(error);
    return getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &length) == 0 && error == 0;
}

}  // namespace

TcpTransport::~TcpTransport() { Stop(); }

Status TcpTransport::StartAccepting(const Endpoint& endpoint, AcceptHandler handler)
{
    if (endpoint.port == 0 || !handler) { return Status::InvalidParam(); }
    Stop();

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    const auto* host = endpoint.host.empty() ? nullptr : endpoint.host.c_str();
    if (getaddrinfo(host, port.c_str(), &hints, &results) != 0) { return Status::Error(); }
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        const auto socket = ::socket(
            item->ai_family, item->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, item->ai_protocol);
        if (socket == kInvalidSocket) { continue; }
        int enabled = 1;
        if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) == 0 &&
            bind(socket, item->ai_addr, item->ai_addrlen) == 0 &&
            listen(socket, kListenBacklog) == 0) {
            listen_socket_ = socket;
            break;
        }
        Close(socket);
    }
    freeaddrinfo(results);
    if (listen_socket_ == kInvalidSocket) { return Status::Error(); }

    accept_handler_ = std::move(handler);
    stopping_.store(false, std::memory_order_release);
    accept_thread_ = std::thread(&TcpTransport::AcceptLoop, this);
    return Status::OK();
}

Status TcpTransport::Connect(const Endpoint& endpoint, Socket& socket) const
{
    socket = kInvalidSocket;
    if (endpoint.host.empty() || endpoint.port == 0) { return Status::InvalidParam(); }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* results = nullptr;
    const auto port = std::to_string(endpoint.port);
    if (getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &results) != 0) {
        return Status::Error();
    }
    for (auto* item = results; item != nullptr; item = item->ai_next) {
        const auto candidate = ::socket(
            item->ai_family, item->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, item->ai_protocol);
        if (candidate == kInvalidSocket) { continue; }
        const auto connected = ::connect(candidate, item->ai_addr, item->ai_addrlen);
        if (connected == 0 || (connected < 0 && errno == EINPROGRESS && WaitConnected(candidate))) {
            socket = candidate;
            break;
        }
        Close(candidate);
    }
    freeaddrinfo(results);
    return socket == kInvalidSocket ? Status::Error() : Status::OK();
}

Status TcpTransport::Send(Socket socket, const void* data, std::size_t length) const
{
    const auto* cursor = static_cast<const char*>(data);
    while (length != 0) {
        const auto sent = ::send(socket, cursor, length, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) { continue; }
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd descriptor{socket, POLLOUT, 0};
            if (poll(&descriptor, 1, -1) > 0) { continue; }
        }
        if (sent <= 0) { return Status::Error(); }
        cursor += sent;
        length -= static_cast<std::size_t>(sent);
    }
    return Status::OK();
}

Status TcpTransport::Receive(Socket socket, void* data, std::size_t capacity,
                             std::size_t& received) const
{
    received = 0;
    const auto result = recv(socket, data, capacity, MSG_DONTWAIT);
    if (result > 0) {
        received = static_cast<std::size_t>(result);
        return Status::OK();
    }
    if (result < 0 && errno == EINTR) { return Receive(socket, data, capacity, received); }
    if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { return Status::Retry(); }
    return Status::Error();
}

void TcpTransport::AcceptLoop()
{
    const auto listen_socket = listen_socket_;
    const auto epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd < 0) {
        UC_ERROR("transport manager TCP listener epoll creation failed errno={} error={}", errno,
                 std::strerror(errno));
        return;
    }
    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = listen_socket;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_socket, &event) != 0) {
        UC_ERROR("transport manager TCP listener epoll add failed errno={} error={}", errno,
                 std::strerror(errno));
        (void)close(epoll_fd);
        return;
    }
    while (!stopping_.load(std::memory_order_acquire)) {
        const auto ready = epoll_wait(epoll_fd, &event, 1, 100);
        if (ready == 0 || (ready < 0 && errno == EINTR)) { continue; }
        if (ready < 0 || (event.events & (EPOLLERR | EPOLLHUP)) != 0) {
            if (!stopping_.load(std::memory_order_acquire)) {
                UC_ERROR("transport manager TCP listener failed errno={} error={}", errno,
                         std::strerror(errno));
            }
            continue;
        }
        const auto socket = accept4(listen_socket, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (socket == kInvalidSocket) {
            if (!stopping_.load(std::memory_order_acquire) && errno != EINTR && errno != EAGAIN &&
                errno != EWOULDBLOCK) {
                UC_ERROR("transport manager TCP accept failed errno={} error={}", errno,
                         std::strerror(errno));
            }
            continue;
        }
        accept_handler_(socket);
    }
    (void)close(epoll_fd);
}

void TcpTransport::Stop()
{
    stopping_.store(true, std::memory_order_release);
    const auto socket = listen_socket_;
    listen_socket_ = kInvalidSocket;
    Close(socket);
    if (accept_thread_.joinable()) { accept_thread_.join(); }
    accept_handler_ = {};
}

void TcpTransport::Close(Socket socket)
{
    if (socket == kInvalidSocket) { return; }
    (void)shutdown(socket, SHUT_RDWR);
    (void)close(socket);
}

}  // namespace transport
