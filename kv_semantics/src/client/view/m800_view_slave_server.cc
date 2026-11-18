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
#include "view/m800_view_slave_server.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netdb.h>
#include <poll.h>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include "logger.h"
#include "view/m800_view_codec.h"
#include "view/m800_view_common.h"

namespace kv {
namespace {

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;
constexpr std::size_t kMaxBodyBytes = 4 * 1024 * 1024;

struct MasterAddress {
    std::string host;
    std::uint16_t port{9100};
    bool ipv6{false};
};

struct HttpResponse {
    int status{500};
    std::string body;
};

Status SlaveError(StatusCode code, const std::string& message)
{
    return Status::Error(code, message);
}

bool ParsePort(std::string_view text, std::uint16_t& port)
{
    if (text.empty()) { return false; }
    unsigned int value{0};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);
    if (result.ec != std::errc{} || result.ptr != end || value == 0 || value > 65535) { return false; }
    port = static_cast<std::uint16_t>(value);
    return true;
}

bool ParseMasterAddress(const std::string& text, MasterAddress& address)
{
    if (text.empty() || text.find_first_of("\r\n\0 ") != std::string::npos) { return false; }
    if (text.front() == '[') {
        const auto close = text.find(']');
        if (close == std::string::npos || close == 1) { return false; }
        address.host = text.substr(1, close - 1);
        address.ipv6 = true;
        if (close + 1 < text.size()) {
            if (text[close + 1] != ':' ||
                !ParsePort(std::string_view{text}.substr(close + 2), address.port)) {
                return false;
            }
        }
        return true;
    }
    const auto colonCount = static_cast<std::size_t>(std::count(text.begin(), text.end(), ':'));
    if (colonCount >= 2) {
        address.host = text;
        address.ipv6 = true;
        return true;
    }
    if (colonCount == 1) {
        const auto colon = text.find(':');
        address.host = text.substr(0, colon);
        if (!ParsePort(std::string_view{text}.substr(colon + 1), address.port)) { return false; }
    } else {
        address.host = text;
    }
    return !address.host.empty();
}

std::uint32_t RemainingMs(std::chrono::steady_clock::time_point deadline)
{
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) { return 0; }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    return static_cast<std::uint32_t>(std::max<std::int64_t>(1, remaining.count()));
}

Status WaitWritable(int fd, std::chrono::steady_clock::time_point deadline,
                    const std::atomic<bool>& stopRequested)
{
    while (true) {
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "socket operation canceled"); }
        const auto remaining = RemainingMs(deadline);
        if (remaining == 0) { return SlaveError(StatusCode::TIMEOUT, "socket operation timed out"); }
        pollfd descriptor{fd, POLLOUT, 0};
        const int result = poll(&descriptor, 1, static_cast<int>(remaining));
        if (result > 0) { return Status::OK(); }
        if (result == 0) { return SlaveError(StatusCode::TIMEOUT, "socket operation timed out"); }
        if (errno != EINTR) {
            return SlaveError(StatusCode::CONNECTION_ERROR,
                              "poll failed: " + std::string(std::strerror(errno)));
        }
    }
}

Status ConnectSocket(const addrinfo& address, std::chrono::steady_clock::time_point deadline,
                     std::atomic_int& activeFd, const std::atomic<bool>& stopRequested,
                     int& connectedFd)
{
    const int fd = socket(address.ai_family, address.ai_socktype, address.ai_protocol);
    if (fd < 0) { return SlaveError(StatusCode::CONNECTION_ERROR, "failed to create socket"); }
    activeFd.store(fd);
    auto fail = [&](Status status) {
        close(fd);
        activeFd.store(-1);
        return status;
    };
    if (stopRequested.load()) { return fail(SlaveError(StatusCode::CANCELED, "connect canceled")); }
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return fail(SlaveError(StatusCode::CONNECTION_ERROR, "failed to set socket nonblocking"));
    }

    const int result = connect(fd, address.ai_addr, address.ai_addrlen);
    if (result < 0 && errno != EINPROGRESS && errno != EWOULDBLOCK) {
        return fail(SlaveError(StatusCode::CONNECTION_ERROR, "connect failed"));
    }
    if (result < 0) {
        auto status = WaitWritable(fd, deadline, stopRequested);
        if (!status.ok()) { return fail(std::move(status)); }
        int socketError{0};
        socklen_t errorSize = sizeof(socketError);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &errorSize) < 0 || socketError != 0) {
            return fail(SlaveError(StatusCode::CONNECTION_ERROR, "connect failed"));
        }
    }
    if (stopRequested.load()) { return fail(SlaveError(StatusCode::CANCELED, "connect canceled")); }
    if (fcntl(fd, F_SETFL, flags) < 0) {
        return fail(SlaveError(StatusCode::CONNECTION_ERROR, "failed to restore socket flags"));
    }
    connectedFd = fd;
    return Status::OK();
}

Status SetSocketTimeout(int fd, int option, std::uint32_t timeoutMs)
{
    timeval timeout{};
    timeout.tv_sec = static_cast<time_t>(timeoutMs / 1000);
    timeout.tv_usec = static_cast<suseconds_t>((timeoutMs % 1000) * 1000);
    if (setsockopt(fd, SOL_SOCKET, option, &timeout, sizeof(timeout)) < 0) {
        return SlaveError(StatusCode::CONNECTION_ERROR, "failed to set socket timeout");
    }
    return Status::OK();
}

Status SendAll(int fd, std::string_view data, const std::atomic<bool>& stopRequested)
{
    std::size_t sent{0};
    while (sent < data.size()) {
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "HTTP write canceled"); }
        const auto result = send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (result > 0) {
            sent += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) { continue; }
        if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return SlaveError(StatusCode::TIMEOUT, "HTTP write timed out");
        }
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "HTTP write canceled"); }
        return SlaveError(StatusCode::CONNECTION_ERROR, "HTTP write failed");
    }
    return Status::OK();
}

std::string TrimHeader(std::string value)
{
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) { return {}; }
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
}

Status ParseHttpResponse(std::string& wire, int fd, std::chrono::steady_clock::time_point deadline,
                         const std::atomic<bool>& stopRequested, HttpResponse& response)
{
    auto receiveMore = [&](void) -> Status {
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "HTTP read canceled"); }
        if (wire.size() >= kMaxHeaderBytes + kMaxBodyBytes) {
            return SlaveError(StatusCode::INTERNAL_ERROR, "HTTP response exceeds size limit");
        }
        char buffer[4096];
        const auto remaining = RemainingMs(deadline);
        if (remaining == 0) { return SlaveError(StatusCode::TIMEOUT, "HTTP read timed out"); }
        const auto status = SetSocketTimeout(fd, SO_RCVTIMEO, remaining);
        if (!status.ok()) { return status; }
        const auto received = recv(fd, buffer, sizeof(buffer), 0);
        if (received > 0) {
            wire.append(buffer, static_cast<std::size_t>(received));
            return Status::OK();
        }
        if (received == 0) { return SlaveError(StatusCode::CONNECTION_ERROR, "HTTP response closed early"); }
        if (errno == EINTR) { return Status::OK(); }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return SlaveError(StatusCode::TIMEOUT, "HTTP read timed out");
        }
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "HTTP read canceled"); }
        return SlaveError(StatusCode::CONNECTION_ERROR, "HTTP read failed");
    };

    std::size_t headerEnd{0};
    while ((headerEnd = wire.find("\r\n\r\n")) == std::string::npos) {
        if (wire.size() > kMaxHeaderBytes) {
            return SlaveError(StatusCode::INTERNAL_ERROR, "HTTP response header is too large");
        }
        auto status = receiveMore();
        if (!status.ok()) { return status; }
    }
    if (headerEnd > kMaxHeaderBytes) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "HTTP response header is too large");
    }

    const auto firstLineEnd = wire.find("\r\n");
    if (firstLineEnd == std::string::npos || firstLineEnd < 12) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "invalid HTTP status line");
    }
    const std::string_view statusLine{wire.data(), firstLineEnd};
    if (statusLine.rfind("HTTP/1.1 ", 0) != 0 && statusLine.rfind("HTTP/1.0 ", 0) != 0) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "invalid HTTP version");
    }
    const auto statusSpace = statusLine.find(' ');
    if (statusSpace == std::string_view::npos || statusSpace + 4 > statusLine.size() ||
        (statusSpace + 4 < statusLine.size() && statusLine[statusSpace + 4] != ' ')) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "invalid HTTP status");
    }
    unsigned int statusCode{0};
    const auto codeText = statusLine.substr(statusSpace + 1, 3);
    const auto codeResult = std::from_chars(codeText.data(), codeText.data() + codeText.size(), statusCode, 10);
    if (codeResult.ec != std::errc{} || codeResult.ptr != codeText.data() + codeText.size()) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "invalid HTTP status code");
    }
    response.status = static_cast<int>(statusCode);

    bool hasContentLength{false};
    std::size_t contentLength{0};
    std::size_t lineBegin = firstLineEnd + 2;
    while (lineBegin < headerEnd) {
        const auto lineEnd = wire.find("\r\n", lineBegin);
        if (lineEnd == std::string::npos || lineEnd > headerEnd) { break; }
        const std::string line = wire.substr(lineBegin, lineEnd - lineBegin);
        lineBegin = lineEnd + 2;
        const auto colon = line.find(':');
        if (colon == std::string::npos || colon == 0) {
            return SlaveError(StatusCode::INTERNAL_ERROR, "invalid HTTP header");
        }
        auto name = line.substr(0, colon);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const auto value = TrimHeader(line.substr(colon + 1));
        if (name == "content-length") {
            std::size_t parsed{0};
            const char* begin = value.data();
            const char* end = begin + value.size();
            const auto result = std::from_chars(begin, end, parsed, 10);
            if (result.ec != std::errc{} || result.ptr != end || parsed > kMaxBodyBytes ||
                (hasContentLength && parsed != contentLength)) {
                return SlaveError(StatusCode::INTERNAL_ERROR, "invalid Content-Length");
            }
            hasContentLength = true;
            contentLength = parsed;
        } else if (name == "transfer-encoding" && value != "identity") {
            return SlaveError(StatusCode::INTERNAL_ERROR, "chunked response is unsupported");
        }
    }

    const std::size_t bodyBegin = headerEnd + 4;
    if (response.status != 200) {
        if (response.status == 304 &&
            (!hasContentLength || contentLength != 0 || wire.size() > bodyBegin)) {
            return SlaveError(StatusCode::INTERNAL_ERROR, "304 response has a body or no Content-Length");
        }
        return Status::OK();
    }
    if (!hasContentLength) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "200 response misses Content-Length");
    }
    if (wire.size() > bodyBegin + contentLength) {
        return SlaveError(StatusCode::INTERNAL_ERROR, "HTTP response exceeds Content-Length");
    }
    while (wire.size() < bodyBegin + contentLength) {
        auto status = receiveMore();
        if (!status.ok()) { return status; }
    }
    response.body = wire.substr(bodyBegin, contentLength);
    return Status::OK();
}

Status FetchFromMaster(const std::string& addressText, std::uint64_t sinceEpoch,
                       std::chrono::steady_clock::time_point deadline,
                       std::atomic_int& activeFd, const std::atomic<bool>& stopRequested,
                       HttpResponse& response)
{
    if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "master fetch canceled"); }
    if (std::chrono::steady_clock::now() >= deadline) {
        return SlaveError(StatusCode::TIMEOUT, "master fetch deadline expired");
    }
    MasterAddress address;
    if (!ParseMasterAddress(addressText, address)) {
        return SlaveError(StatusCode::INVALID_ARGUMENT, "invalid master address: " + addressText);
    }
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* rawAddresses{nullptr};
    const auto service = std::to_string(address.port);
    const int gaiResult = getaddrinfo(address.host.c_str(), service.c_str(), &hints, &rawAddresses);
    if (gaiResult != 0) {
        return SlaveError(StatusCode::CONNECTION_ERROR,
                          "failed to resolve master: " + std::string(gai_strerror(gaiResult)));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(rawAddresses, freeaddrinfo);
    if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "master fetch canceled"); }
    if (std::chrono::steady_clock::now() >= deadline) {
        return SlaveError(StatusCode::TIMEOUT, "master fetch deadline expired");
    }

    Status lastFailure = SlaveError(StatusCode::CONNECTION_ERROR, "master is unreachable");
    for (const addrinfo* current = addresses.get(); current != nullptr; current = current->ai_next) {
        int fd{-1};
        auto status = ConnectSocket(*current, deadline, activeFd, stopRequested, fd);
        if (!status.ok()) {
            lastFailure = std::move(status);
            if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "master fetch canceled"); }
            continue;
        }
        const auto request = "GET /v1/views?since_epoch=" + std::to_string(sinceEpoch) +
                             " HTTP/1.1\r\nHost: " + addressText +
                             "\r\nConnection: close\r\n\r\n";
        const auto remaining = RemainingMs(deadline);
        status = remaining == 0 ? SlaveError(StatusCode::TIMEOUT, "HTTP write timed out")
                                : SetSocketTimeout(fd, SO_SNDTIMEO, remaining);
        if (status.ok()) { status = SendAll(fd, request, stopRequested); }
        std::string wire;
        if (status.ok()) { status = ParseHttpResponse(wire, fd, deadline, stopRequested, response); }
        shutdown(fd, SHUT_RDWR);
        close(fd);
        activeFd.store(-1);
        if (status.ok()) { return Status::OK(); }
        if (stopRequested.load()) { return SlaveError(StatusCode::CANCELED, "master fetch canceled"); }
        lastFailure = std::move(status);
    }
    return lastFailure;
}

class M800ViewSlaveServerImpl final : public M800ViewSlaveServer {
public:
    explicit M800ViewSlaveServerImpl(M800ViewServerConfig config) : config_(std::move(config)) {}
    ~M800ViewSlaveServerImpl() override { Stop(); }

    Status GetAllViews(ViewUniverse& universe) override
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
        std::vector<std::string> addresses = config_.masterAddrs;
        if (addresses.empty() && !config_.masterAddr.empty()) {
            addresses.emplace_back(config_.masterAddr);
        }
        if (addresses.empty()) {
            return SlaveError(StatusCode::INVALID_ARGUMENT,
                              "view.master_addrs or view.master_addr is required for worker");
        }

        Status lastFailure = SlaveError(StatusCode::CONNECTION_ERROR, "no master is available");
        const auto effectiveEpoch = std::max(cachedEpoch_, universe.GetGlobalEpoch());
        const auto timeout = config_.httpTimeoutMs == 0 ? std::uint64_t{2000} : config_.httpTimeoutMs;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
        std::size_t staleMasterCount{0};
        for (const auto& address : addresses) {
            if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
            if (std::chrono::steady_clock::now() >= deadline) {
                return SlaveError(StatusCode::TIMEOUT, "worker fetch deadline expired");
            }
            HttpResponse response;
            auto status = FetchFromMaster(address, effectiveEpoch, deadline, activeFd_, stopRequested_,
                                          response);
            if (!status.ok()) {
                if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
                lastFailure = std::move(status);
                continue;
            }
            if (response.status == 304) {
                if (effectiveEpoch == 0) {
                    lastFailure = SlaveError(StatusCode::INTERNAL_ERROR,
                                             "master returned 304 without a known local epoch");
                    continue;
                }
                return Status::OK();
            }
            if (response.status != 200) {
                lastFailure = SlaveError(StatusCode::INTERNAL_ERROR,
                                         "master returned HTTP " + std::to_string(response.status));
                continue;
            }

            auto decoded = CreateViewUniverse();
            if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
            status = M800ViewCodec::Decode(response.body, *decoded);
            if (!status.ok()) {
                lastFailure = std::move(status);
                continue;
            }
            if (decoded->Empty()) {
                lastFailure = SlaveError(StatusCode::INTERNAL_ERROR, "master returned an empty universe");
                continue;
            }
            const auto nextEpoch = decoded->GetGlobalEpoch();
            if (effectiveEpoch != 0 && nextEpoch < effectiveEpoch) {
                ++staleMasterCount;
                KV_WARN("Master {} returned stale epoch {} < effective epoch {}, trying next master",
                        address, nextEpoch, effectiveEpoch);
                continue;
            }
            status = universe.PublishAll(*decoded);
            if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
            if (std::chrono::steady_clock::now() >= deadline) {
                return SlaveError(StatusCode::TIMEOUT, "worker fetch deadline expired");
            }
            if (!status.ok()) {
                lastFailure = std::move(status);
                continue;
            }
            cachedEpoch_ = std::max(cachedEpoch_, universe.GetGlobalEpoch());
            return Status::OK();
        }
        if (stopRequested_.load()) { return SlaveError(StatusCode::CANCELED, "worker is stopping"); }
        if (staleMasterCount == addresses.size()) {
            // Every address returned a valid, non-empty but stale view. This is
            // an intentional no-op, not a new publication or an error.
            return Status::OK();
        }
        return lastFailure;
    }

    void Stop() override
    {
        stopRequested_.store(true);
        const int fd = activeFd_.load();
        if (fd >= 0) { shutdown(fd, SHUT_RDWR); }
        std::lock_guard<std::mutex> guard(mutex_);
        cachedEpoch_ = 0;
        stopRequested_.store(false);
    }

    Status GetGlobalView(GlobalView& view) override
    {
        (void)view;
        return SlaveError(StatusCode::UNSUPPORTED,
                          "GetGlobalView is unsupported for a multi-namespace M800 worker");
    }

private:
    M800ViewServerConfig config_;
    std::mutex mutex_;
    std::atomic_int activeFd_{-1};
    std::atomic<bool> stopRequested_{false};
    std::uint64_t cachedEpoch_{0};
};

}  // namespace

std::shared_ptr<M800ViewSlaveServer> CreateM800ViewSlaveServer(const M800ViewServerConfig& config)
{
    try {
        return std::make_shared<M800ViewSlaveServerImpl>(config);
    } catch (...) {
        return nullptr;
    }
}

}  // namespace kv
