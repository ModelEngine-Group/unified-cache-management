/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Minimal read-only HTTP server for M800 views.
 */

#include "view/view_http_server.h"

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>

#include "view/m800_view_common.h"

namespace kv {
namespace {

constexpr std::size_t kMaxRequestLineBytes = 8 * 1024;
constexpr std::size_t kMaxRequestHeaderBytes = 16 * 1024;
constexpr std::size_t kMaxResponseBodyBytes = 4 * 1024 * 1024;
constexpr int kListenBacklog = 16;

void CloseFd(int fd)
{
    if (fd >= 0) { close(fd); }
}

std::string ErrnoMessage(int error)
{
    return std::error_code(error, std::generic_category()).message();
}

bool SplitListenAddress(const std::string& address, std::string& host, std::string& port)
{
    host.clear();
    port.clear();
    if (address.empty()) { return false; }
    if (address.front() == '[') {
        const auto closeBracket = address.find(']');
        if (closeBracket == std::string::npos || closeBracket == 1 ||
            closeBracket + 1 >= address.size() || address[closeBracket + 1] != ':') {
            return false;
        }
        host = address.substr(1, closeBracket - 1);
        port = address.substr(closeBracket + 2);
    } else {
        const auto firstColon = address.find(':');
        const auto lastColon = address.rfind(':');
        if (firstColon == std::string::npos || firstColon == 0 || firstColon != lastColon ||
            lastColon + 1 >= address.size()) {
            return false;
        }
        host = address.substr(0, lastColon);
        port = address.substr(lastColon + 1);
    }
    if (port.empty()) { return false; }
    unsigned int numericPort = 0;
    const auto result = std::from_chars(port.data(), port.data() + port.size(), numericPort, 10);
    return result.ec == std::errc{} && result.ptr == port.data() + port.size() &&
           numericPort <= 65535;
}

std::uint64_t ParseSinceEpoch(const std::string& target)
{
    const auto queryPos = target.find('?');
    if (queryPos == std::string::npos || queryPos + 1 >= target.size()) { return 0; }
    const auto query = target.substr(queryPos + 1);
    bool found = false;
    std::uint64_t result = 0;
    std::size_t begin = 0;
    while (begin <= query.size()) {
        const auto end = query.find('&', begin);
        const auto item = query.substr(begin, end == std::string::npos ? std::string::npos
                                                                       : end - begin);
        const auto equals = item.find('=');
        if (equals != std::string::npos && item.substr(0, equals) == "since_epoch") {
            if (found) { return 0; }
            found = true;
            const auto value = item.substr(equals + 1);
            std::uint64_t parsed = 0;
            const auto parseResult =
                std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
            if (value.empty() || parseResult.ec != std::errc{} ||
                parseResult.ptr != value.data() + value.size()) {
                result = 0;
            } else {
                result = parsed;
            }
        }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return result;
}

bool ParseRequest(const std::string& request, ViewHttpRequest& parsed)
{
    parsed.sinceEpoch = 0;
    const auto lineEnd = request.find("\r\n");
    if (lineEnd == std::string::npos || lineEnd > kMaxRequestLineBytes) { return false; }
    const auto line = request.substr(0, lineEnd);
    const auto firstSpace = line.find(' ');
    const auto secondSpace = firstSpace == std::string::npos
                                 ? std::string::npos
                                 : line.find(' ', firstSpace + 1);
    if (firstSpace == std::string::npos || secondSpace == std::string::npos ||
        firstSpace == 0 || secondSpace == firstSpace + 1 || secondSpace + 1 >= line.size() ||
        line.find(' ', secondSpace + 1) != std::string::npos) {
        return false;
    }
    if (line.substr(0, firstSpace) != "GET") { return false; }
    const auto version = line.substr(secondSpace + 1);
    if (version != "HTTP/1.1" && version != "HTTP/1.0") { return false; }

    const auto target = line.substr(firstSpace + 1, secondSpace - firstSpace - 1);
    const auto queryPos = target.find('?');
    const auto path = target.substr(0, queryPos);
    if (path != M800ViewHTTPContract::kViewsPath) { return false; }
    parsed.sinceEpoch = ParseSinceEpoch(target);
    return true;
}

std::string BuildResponse(ViewHttpReplyKind kind, const std::string& requestedBody)
{
    int status = M800ViewHTTPContract::kStatusServerError;
    const char* reason = "Internal Server Error";
    const std::string* body = nullptr;
    if (kind == ViewHttpReplyKind::FULL_VIEW && requestedBody.size() <= kMaxResponseBodyBytes) {
        status = M800ViewHTTPContract::kStatusOk;
        reason = "OK";
        body = &requestedBody;
    } else if (kind == ViewHttpReplyKind::NOT_MODIFIED) {
        status = M800ViewHTTPContract::kStatusNotModified;
        reason = "Not Modified";
    }
    static const std::string emptyBody;
    if (body == nullptr) { body = &emptyBody; }

    std::string response = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
    response += "Content-Type: ";
    response += M800ViewHTTPContract::kContentType;
    response += "\r\nContent-Length: " + std::to_string(body->size());
    response += "\r\nConnection: ";
    response += M800ViewHTTPContract::kConnection;
    response += "\r\n\r\n";
    response += *body;
    return response;
}

bool SendAll(int fd, const std::string& data)
{
    std::size_t sent = 0;
    while (sent < data.size()) {
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const auto result = send(fd, data.data() + sent, data.size() - sent, flags);
        if (result > 0) {
            sent += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) { continue; }
        return false;
    }
    return true;
}

class ViewHttpServerImpl final : public ViewHttpServer {
public:
    explicit ViewHttpServerImpl(std::uint64_t timeoutMs)
        : timeoutMs_(timeoutMs == 0 ? 2000 : timeoutMs)
    {
    }

    ~ViewHttpServerImpl() override { Stop(); }

    Status Start(const std::string& listenAddr, const ViewHttpHandler& handler) override
    {
        std::lock_guard<std::mutex> lock{lifecycleMutex_};
        if (!handler) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "HTTP handler must not be empty");
        }
        if (running_.load() || listenFd_.load() >= 0 || thread_.joinable()) {
            return Status::Error(StatusCode::RESOURCE_BUSY, "HTTP server is already running");
        }

        std::string host;
        std::string port;
        if (!SplitListenAddress(listenAddr, host, port)) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "listen address must be host:port or [ipv6]:port");
        }

        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        hints.ai_flags = AI_PASSIVE;
        addrinfo* addresses = nullptr;
        const auto gaiResult = getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses);
        if (gaiResult != 0) {
            return Status::Error(StatusCode::CONNECTION_ERROR,
                                 "failed to resolve HTTP listen address: " +
                                     std::string{gai_strerror(gaiResult)});
        }

        int listenFd = -1;
        int lastError = 0;
        for (auto* address = addresses; address != nullptr; address = address->ai_next) {
            const int candidate = socket(address->ai_family, address->ai_socktype,
                                         address->ai_protocol);
            if (candidate < 0) {
                lastError = errno;
                continue;
            }
            int reuse = 1;
            if (setsockopt(candidate, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
                lastError = errno;
                CloseFd(candidate);
                continue;
            }
            if (bind(candidate, address->ai_addr, address->ai_addrlen) != 0) {
                lastError = errno;
                CloseFd(candidate);
                continue;
            }
            if (listen(candidate, kListenBacklog) != 0) {
                lastError = errno;
                CloseFd(candidate);
                continue;
            }
            listenFd = candidate;
            break;
        }
        freeaddrinfo(addresses);
        if (listenFd < 0) {
            std::string message = "failed to bind HTTP listen address " + listenAddr;
            if (lastError != 0) { message += ": " + ErrnoMessage(lastError); }
            return Status::Error(StatusCode::CONNECTION_ERROR, message);
        }

        try {
            handler_ = handler;
        } catch (...) {
            CloseFd(listenFd);
            return Status::Error(StatusCode::INTERNAL_ERROR, "failed to store HTTP handler");
        }
        listenFd_.store(listenFd);
        running_.store(true);
        try {
            thread_ = std::thread(&ViewHttpServerImpl::AcceptLoop, this);
        } catch (...) {
            running_.store(false);
            listenFd_.store(-1);
            handler_ = nullptr;
            CloseFd(listenFd);
            return Status::Error(StatusCode::INTERNAL_ERROR, "failed to start HTTP accept thread");
        }
        return Status::OK();
    }

    void Stop() override
    {
        std::lock_guard<std::mutex> lock{lifecycleMutex_};
        running_.exchange(false);
        const int fd = listenFd_.exchange(-1);
        if (fd >= 0) {
            shutdown(fd, SHUT_RDWR);
            CloseFd(fd);
        }
        if (thread_.joinable()) { thread_.join(); }
        handler_ = nullptr;
    }

private:
    bool ConfigureSocket(int fd) const
    {
        timeval timeout{};
        const auto seconds = timeoutMs_ / 1000;
        const auto maxSeconds = static_cast<std::uint64_t>(std::numeric_limits<time_t>::max());
        timeout.tv_sec = static_cast<time_t>(seconds > maxSeconds ? maxSeconds : seconds);
        timeout.tv_usec = static_cast<suseconds_t>((timeoutMs_ % 1000) * 1000);
        return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
               setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0;
    }

    bool ReceiveRequest(int fd, std::string& request) const
    {
        request.clear();
        char buffer[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            const auto received = recv(fd, buffer, sizeof(buffer), 0);
            if (received > 0) {
                request.append(buffer, static_cast<std::size_t>(received));
                const auto lineEnd = request.find("\r\n");
                if ((lineEnd == std::string::npos && request.size() > kMaxRequestLineBytes) ||
                    (lineEnd != std::string::npos && lineEnd > kMaxRequestLineBytes) ||
                    request.size() > kMaxRequestHeaderBytes) {
                    return false;
                }
                continue;
            }
            if (received < 0 && errno == EINTR) { continue; }
            return false;
        }
        return true;
    }

    void HandleConnection(int fd)
    {
        ViewHttpReply reply;
        try {
            if (!ConfigureSocket(fd)) { return; }
            std::string request;
            ViewHttpRequest parsed;
            if (!ReceiveRequest(fd, request) || !ParseRequest(request, parsed)) {
                reply.kind = ViewHttpReplyKind::INTERNAL_ERROR;
            } else {
                try {
                    reply = handler_(parsed);
                } catch (...) {
                    reply = ViewHttpReply{};
                }
            }
            const auto response = BuildResponse(reply.kind, reply.body);
            SendAll(fd, response);
        } catch (...) {
            try {
                const auto response =
                    BuildResponse(ViewHttpReplyKind::INTERNAL_ERROR, std::string{});
                SendAll(fd, response);
            } catch (...) {
                // The connection is closed by the accept loop.
            }
        }
    }

    void AcceptLoop()
    {
        while (running_.load()) {
            const int fd = listenFd_.load();
            if (fd < 0) { break; }
            const int client = accept(fd, nullptr, nullptr);
            if (client < 0) {
                if (!running_.load()) { break; }
                if (errno == EINTR) { continue; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (!running_.load()) {
                CloseFd(client);
                break;
            }
            HandleConnection(client);
            CloseFd(client);
        }
    }

    const std::uint64_t timeoutMs_;
    std::atomic_bool running_{false};
    std::atomic_int listenFd_{-1};
    std::thread thread_;
    ViewHttpHandler handler_;
    std::mutex lifecycleMutex_;
};

}  // namespace

std::unique_ptr<ViewHttpServer> CreateViewHttpServer(std::uint64_t timeoutMs)
{
    return std::unique_ptr<ViewHttpServer>{new (std::nothrow) ViewHttpServerImpl(timeoutMs)};
}

}  // namespace kv
