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
#include "roce_tcp_channel.h"
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

// TCP_USER_TIMEOUT is available since Linux 2.6.37; keep the numeric value as a
// fallback so the provider still builds against older headers.
#ifndef TCP_USER_TIMEOUT
#define TCP_USER_TIMEOUT 18
#endif

namespace kv {
namespace roce {
namespace {

std::string ErrnoText(int code) { return std::string(std::strerror(code)); }

bool SetNonBlocking(int fd, bool enable)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) { return false; }
    const int wanted = enable ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, wanted) == 0;
}

void TuneConnectedSocket(int fd)
{
    int on = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
}

bool IsIpv4Literal(const std::string& address)
{
    in_addr parsed{};
    return inet_pton(AF_INET, address.c_str(), &parsed) == 1;
}

bool FillSockAddr(const std::string& address, std::uint16_t port, sockaddr_in& out,
                  std::string& error)
{
    out = sockaddr_in{};
    out.sin_family = AF_INET;
    out.sin_port = htons(port);
    if (inet_pton(AF_INET, address.c_str(), &out.sin_addr) != 1) {
        error = "invalid IPv4 address '" + address + "'";
        return false;
    }
    return true;
}

// poll() takes its timeout as an int, and a negative timeout means "wait
// forever".  A millisecond count above INT_MAX would wrap into exactly that,
// so it is clamped instead: the caller still gets its deadline, only measured
// in several waits rather than one.
constexpr std::uint32_t kMaxPollTimeoutMs = 0x7FFFFFFFU;

int PollTimeoutMs(std::uint32_t timeoutMs)
{
    return static_cast<int>(timeoutMs > kMaxPollTimeoutMs ? kMaxPollTimeoutMs : timeoutMs);
}

Status WaitForFd(int fd, short events, std::uint32_t timeoutMs, std::string& error)
{
    pollfd entry{};
    entry.fd = fd;
    entry.events = events;
    const int timeout = PollTimeoutMs(timeoutMs);
    int ready = 0;
    do {
        ready = poll(&entry, 1, timeout);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
        error = "poll() failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    if (ready == 0) {
        error = "timed out after " + std::to_string(timeoutMs) + " ms";
        return Status::Error(StatusCode::TIMEOUT, error);
    }
    return Status::OK();
}

}  // namespace

Status TcpChannel::ApplyTuning(int fd, bool includeUserTimeout, std::string& error)
{
    const int buffer = static_cast<int>(kSocketBufferBytes);
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, sizeof(buffer)) != 0) {
        error = "setsockopt(SO_SNDBUF) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, sizeof(buffer)) != 0) {
        error = "setsockopt(SO_RCVBUF) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    if (!includeUserTimeout) { return Status::OK(); }
    const unsigned int userTimeout = kUserTimeoutMs;
    if (setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &userTimeout, sizeof(userTimeout)) != 0) {
        error = "setsockopt(TCP_USER_TIMEOUT) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    return Status::OK();
}

TcpChannel::TcpChannel(int fd) : fd_(fd) {}

TcpChannel::~TcpChannel() { Close(); }

TcpChannel::TcpChannel(TcpChannel&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }

TcpChannel& TcpChannel::operator=(TcpChannel&& other) noexcept
{
    if (this != &other) {
        Close();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

Status TcpChannel::Connect(const std::string& localIp, const std::string& remoteIp,
                           std::uint16_t port, std::uint32_t timeoutMs, TcpChannel& channel,
                           std::string& error)
{
    error.clear();
    if (port == 0) {
        error = "remote port must not be zero";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    if (!IsIpv4Literal(remoteIp)) {
        error = "invalid remote IPv4 address '" + remoteIp + "'";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    sockaddr_in remote{};
    if (!FillSockAddr(remoteIp, port, remote, error)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        error = "socket() failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    TcpChannel socket(fd);

    if (Status tuned = ApplyTuning(fd, true, error); !tuned.ok()) { return tuned; }

    if (!localIp.empty()) {
        sockaddr_in local{};
        if (!FillSockAddr(localIp, 0, local, error)) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, error);
        }
        if (::bind(fd, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
            error = "bind(" + localIp + ") failed: " + ErrnoText(errno);
            return Status::Error(StatusCode::IO_ERROR, error);
        }
    }

    if (!SetNonBlocking(fd, true)) {
        error = "fcntl(O_NONBLOCK) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }

    const int rc = ::connect(fd, reinterpret_cast<const sockaddr*>(&remote), sizeof(remote));
    if (rc != 0 && errno != EINPROGRESS) {
        error =
            "connect(" + remoteIp + ":" + std::to_string(port) + ") failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::CONNECTION_ERROR, error);
    }

    if (rc != 0) {
        std::string waitError;
        Status waited = WaitForFd(fd, POLLOUT, timeoutMs, waitError);
        if (!waited.ok()) {
            error = "connect(" + remoteIp + ":" + std::to_string(port) + ") " + waitError;
            const StatusCode code = waited.code == StatusCode::TIMEOUT
                                        ? StatusCode::TIMEOUT
                                        : StatusCode::CONNECTION_ERROR;
            return Status::Error(code, error);
        }
        int soError = 0;
        socklen_t length = sizeof(soError);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soError, &length) != 0) {
            error = "getsockopt(SO_ERROR) failed: " + ErrnoText(errno);
            return Status::Error(StatusCode::CONNECTION_ERROR, error);
        }
        if (soError != 0) {
            error = "connect(" + remoteIp + ":" + std::to_string(port) +
                    ") failed: " + ErrnoText(soError);
            return Status::Error(StatusCode::CONNECTION_ERROR, error);
        }
    }

    TuneConnectedSocket(fd);
    channel = std::move(socket);
    return Status::OK();
}

Status TcpChannel::Listen(const std::string& bindIp, std::uint16_t port, std::uint16_t backlog,
                          TcpChannel& listener, std::string& error)
{
    error.clear();
    if (port == 0) {
        error = "listen port must not be zero";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    sockaddr_in local{};
    if (bindIp.empty()) {
        local = sockaddr_in{};
        local.sin_family = AF_INET;
        local.sin_port = htons(port);
        local.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (!FillSockAddr(bindIp, port, local, error)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        error = "socket() failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    TcpChannel socket(fd);

    // A listening socket only needs the buffers; the accepted connection gets
    // the full tuning including TCP_USER_TIMEOUT.
    if (Status tuned = ApplyTuning(fd, false, error); !tuned.ok()) { return tuned; }

    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
        error = "bind(port " + std::to_string(port) + ") failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    if (::listen(fd, static_cast<int>(backlog)) != 0) {
        error = "listen() failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    // A non-blocking listener turns the race between poll() and accept4() -- the
    // connection is reset before accept4() runs -- into EAGAIN instead of a call
    // that blocks past its timeout.
    if (!SetNonBlocking(fd, true)) {
        error = "fcntl(O_NONBLOCK) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }

    listener = std::move(socket);
    return Status::OK();
}

namespace {

// Milliseconds left until the deadline, zero once it has passed.  The result is
// clamped to the ceiling poll() accepts, so a deadline beyond that is waited out
// in steps instead of overflowing into an immediate or endless wait.
int RemainingMs(const std::chrono::steady_clock::time_point& deadline)
{
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) { return 0; }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    const auto ceiling = static_cast<decltype(remaining)>(kMaxPollTimeoutMs);
    return static_cast<int>(remaining > ceiling ? ceiling : remaining);
}

}  // namespace

Status TcpChannel::Accept(const TcpChannel& listener, std::uint32_t timeoutMs, TcpChannel& accepted,
                          std::string& error)
{
    error.clear();
    if (!listener.Valid()) {
        error = "accept() on an invalid listening socket";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    int fd = -1;
    while (fd < 0) {
        std::string waitError;
        Status waited = WaitForFd(listener.fd_, POLLIN, RemainingMs(deadline), waitError);
        if (!waited.ok()) {
            error = "accept() " + waitError;
            const StatusCode code = waited.code == StatusCode::TIMEOUT
                                        ? StatusCode::TIMEOUT
                                        : StatusCode::CONNECTION_ERROR;
            return Status::Error(code, error);
        }
        fd = ::accept4(listener.fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd >= 0) { break; }
        if (errno == EINTR) { continue; }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // The connection that woke poll() was reset before it could be
            // accepted; wait for the rest of the budget instead of failing.
            if (RemainingMs(deadline) == 0) {
                error = "accept() timed out after " + std::to_string(timeoutMs) + " ms";
                return Status::Error(StatusCode::TIMEOUT, error);
            }
            continue;
        }
        error = "accept4() failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::CONNECTION_ERROR, error);
    }
    TcpChannel connection(fd);
    if (Status tuned = ApplyTuning(fd, true, error); !tuned.ok()) { return tuned; }
    if (!SetNonBlocking(fd, true)) {
        error = "fcntl(O_NONBLOCK) failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::IO_ERROR, error);
    }
    TuneConnectedSocket(fd);
    accepted = std::move(connection);
    return Status::OK();
}

namespace {

// One poll-until-done loop serves both directions: the socket is
// non-blocking, so a short count means "wait for readiness and retry".
Status Transfer(bool sending, int fd, const std::uint8_t* data, std::size_t size,
                std::uint32_t timeoutMs, std::string& error)
{
    const std::string verb = sending ? "sending" : "receiving";
    const std::string action = sending ? "send" : "receive";
    const std::string readiness = sending ? "writable" : "readable";
    const std::string polling = sending ? "writability" : "readability";
    auto* buffer = const_cast<std::uint8_t*>(data);  // only written when receiving

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t step = sending ? ::send(fd, buffer + done, size - done, MSG_NOSIGNAL)
                                     : ::recv(fd, buffer + done, size - done, 0);
        if (step > 0) {
            done += static_cast<std::size_t>(step);
            continue;
        }
        if (!sending && step == 0) {
            error = "peer closed the connection after " + std::to_string(done) + " of " +
                    std::to_string(size) + " bytes";
            return Status::Error(StatusCode::CONNECTION_ERROR, error);
        }
        if (step < 0 && errno == EINTR) { continue; }
        if (step < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const int budget = RemainingMs(deadline);
            if (budget <= 0) {
                error = "timed out after " + verb + " " + std::to_string(done) + " of " +
                        std::to_string(size) + " bytes";
                return Status::Error(StatusCode::TIMEOUT, error);
            }
            pollfd entry{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
            const int ready = ::poll(&entry, 1, budget);
            if (ready == 0) {
                error = "timed out waiting for the socket to become " + readiness;
                return Status::Error(StatusCode::TIMEOUT, error);
            }
            if (ready < 0 && errno != EINTR) {
                error = "poll for " + polling + " failed: " + ErrnoText(errno);
                return Status::Error(StatusCode::IO_ERROR, error);
            }
            continue;
        }
        error = action + " failed: " + ErrnoText(errno);
        return Status::Error(StatusCode::CONNECTION_ERROR, error);
    }
    return Status::OK();
}

}  // namespace

Status TcpChannel::SendAll(const std::uint8_t* data, std::size_t size, std::uint32_t timeoutMs,
                           std::string& error) const
{
    error.clear();
    if (fd_ < 0) {
        error = "socket is not open";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    return Transfer(true, fd_, data, size, timeoutMs, error);
}

Status TcpChannel::RecvAll(std::uint8_t* data, std::size_t size, std::uint32_t timeoutMs,
                           std::string& error) const
{
    error.clear();
    if (fd_ < 0) {
        error = "socket is not open";
        return Status::Error(StatusCode::INVALID_ARGUMENT, error);
    }
    return Transfer(false, fd_, data, size, timeoutMs, error);
}

bool TcpChannel::Valid() const noexcept { return fd_ >= 0; }

int TcpChannel::Fd() const noexcept { return fd_; }

std::string TcpChannel::PeerAddress() const
{
    if (fd_ < 0) { return std::string(); }
    sockaddr_in peer{};
    socklen_t length = sizeof(peer);
    if (getpeername(fd_, reinterpret_cast<sockaddr*>(&peer), &length) != 0) {
        return std::string();
    }
    char text[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &peer.sin_addr, text, sizeof(text)) == nullptr) { return std::string(); }
    return std::string(text) + ":" + std::to_string(ntohs(peer.sin_port));
}

void TcpChannel::Close() noexcept
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

std::size_t TcpChannel::SendBufferBytes() const
{
    if (fd_ < 0) { return 0; }
    int value = 0;
    socklen_t length = sizeof(value);
    if (getsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &value, &length) != 0 || value <= 0) { return 0; }
    return static_cast<std::size_t>(value);
}

std::size_t TcpChannel::ReceiveBufferBytes() const
{
    if (fd_ < 0) { return 0; }
    int value = 0;
    socklen_t length = sizeof(value);
    if (getsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &value, &length) != 0 || value <= 0) { return 0; }
    return static_cast<std::size_t>(value);
}

std::uint32_t TcpChannel::UserTimeoutMs() const
{
    if (fd_ < 0) { return 0; }
    unsigned int value = 0;
    socklen_t length = sizeof(value);
    if (getsockopt(fd_, IPPROTO_TCP, TCP_USER_TIMEOUT, &value, &length) != 0) { return 0; }
    return value;
}

}  // namespace roce
}  // namespace kv
