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
#include "ipc/uds.h"
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace UC::Transfer {
namespace {

constexpr std::size_t kHeaderSize = 4;

constexpr std::size_t kControlBufferSize = CMSG_SPACE(kUdsMaxFdsPerFrame * sizeof(int));

void PutLe32(std::byte* dst, std::uint32_t value)
{
    dst[0] = static_cast<std::byte>((value >> 0) & 0xFF);
    dst[1] = static_cast<std::byte>((value >> 8) & 0xFF);
    dst[2] = static_cast<std::byte>((value >> 16) & 0xFF);
    dst[3] = static_cast<std::byte>((value >> 24) & 0xFF);
}

std::uint32_t GetLe32(const std::byte* src)
{
    return static_cast<std::uint32_t>(std::to_integer<std::uint32_t>(src[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint32_t>(src[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint32_t>(src[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint32_t>(src[3])) << 24);
}

/**
 * @brief Collects the descriptors of one recvmsg call. Every received
 *        descriptor is an owned duplicate; ownership passes to the caller's
 *        vector.
 */
Status CollectFds(msghdr& msg, std::vector<int>& fds)
{
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) { continue; }
        const std::size_t count = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        const auto* data = reinterpret_cast<const int*>(CMSG_DATA(cmsg));
        fds.insert(fds.end(), data, data + count);
    }
    if ((msg.msg_flags & MSG_CTRUNC) != 0) {
        return Status::InvalidParam(
            "UDS frame carries more ancillary data than the receiver allows");
    }
    return Status::Ok();
}

struct ChunkResult {
    std::size_t received{0};
    bool peerClosed{false};
    Status status;
};

/**
 * @brief Performs one recvmsg into the given buffer, retrying interrupts,
 *        and folds whatever ancillary descriptors arrived into fds.
 */
ChunkResult ReceiveChunk(int fd, std::byte* buf, std::size_t len, std::vector<int>& fds)
{
    alignas(cmsghdr) std::array<char, kControlBufferSize> control{};
    iovec iov{};
    iov.iov_base = buf;
    iov.iov_len = len;
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.data();
    msg.msg_controllen = control.size();

    ssize_t n = 0;
    do {
        n = ::recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
    } while (n < 0 && errno == EINTR);
    if (n == 0) { return ChunkResult{0, true, Status::Ok()}; }
    if (n < 0) {
        return ChunkResult{0, false,
                           Status::OsApiError("UDS recvmsg failed: {}", std::strerror(errno))};
    }
    if (Status st = CollectFds(msg, fds); st.Failure()) {
        return ChunkResult{0, false, std::move(st)};
    }
    return ChunkResult{static_cast<std::size_t>(n), false, Status::Ok()};
}

constexpr std::size_t kSunPathLimit = sizeof(sockaddr_un::sun_path) - 1;

Status CheckSocketPath(const std::string& path)
{
    if (path.empty() || path.size() >= kSunPathLimit) {
        return Status::InvalidParam("UDS socket path is empty or longer than {} bytes: '{}'",
                                    kSunPathLimit, path);
    }
    return Status::Ok();
}

sockaddr_un MakeAddress(const std::string& path)
{
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    return addr;
}

}  // namespace

UdsConnection::~UdsConnection() { Close(); }

UdsConnection::UdsConnection(UdsConnection&& other) noexcept : fd_{std::exchange(other.fd_, -1)} {}

UdsConnection& UdsConnection::operator=(UdsConnection&& other) noexcept
{
    if (this != &other) {
        Close();
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

Expected<UdsConnection> UdsConnection::Connect(const std::string& path)
{
    if (Status st = CheckSocketPath(path); st.Failure()) { return st; }
    const sockaddr_un addr = MakeAddress(path);

    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return Status::OsApiError("UDS socket creation failed: {}", std::strerror(errno));
    }
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
        Status st =
            Status::OsApiError("UDS connect to '{}' failed: {}", path, std::strerror(errno));
        ::close(fd);
        return st;
    }
    return UdsConnection{fd};
}

Status UdsConnection::SendFrame(const std::vector<std::byte>& payload)
{
    static const std::vector<int> kNoFds{};
    return SendFrameWithFds(payload, kNoFds);
}

Status UdsConnection::SendFrameWithFds(const std::vector<std::byte>& payload,
                                       const std::vector<int>& fds)
{
    if (fd_ < 0) { return Status::InvalidParam("UdsConnection: connection is not open"); }
    if (payload.size() > kUdsMaxFrameSize) {
        return Status::InvalidParam("UDS frame payload of {} bytes exceeds the {} byte limit",
                                    payload.size(), kUdsMaxFrameSize);
    }
    if (fds.size() > kUdsMaxFdsPerFrame) {
        return Status::InvalidParam("UDS frame carries {} descriptors, more than the limit of {}",
                                    fds.size(), kUdsMaxFdsPerFrame);
    }

    std::array<std::byte, kHeaderSize> header{};
    PutLe32(header.data(), static_cast<std::uint32_t>(payload.size()));

    alignas(cmsghdr) std::array<char, kControlBufferSize> control{};
    std::array<iovec, 2> iov{};
    iov[0].iov_base = header.data();
    iov[0].iov_len = header.size();
    iov[1].iov_base = const_cast<std::byte*>(payload.data());
    iov[1].iov_len = payload.size();
    msghdr msg{};
    msg.msg_iov = iov.data();
    msg.msg_iovlen = iov.size();
    if (!fds.empty()) {
        /*
         * msg_controllen must cover exactly the cmsg chain written: the kernel
         * walks every header inside the declared window and rejects a
         * half-filled control buffer (trailing zeros read as a zero cmsg_len)
         * with EINVAL.
         */
        cmsghdr* cmsg = reinterpret_cast<cmsghdr*>(control.data());
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
        std::memcpy(CMSG_DATA(cmsg), fds.data(), fds.size() * sizeof(int));
        msg.msg_control = control.data();
        msg.msg_controllen = cmsg->cmsg_len;
    }

    const std::size_t total = header.size() + payload.size();
    ssize_t sent = 0;
    do {
        sent = ::sendmsg(fd_, &msg, MSG_NOSIGNAL);
    } while (sent < 0 && errno == EINTR);
    if (sent < 0) { return Status::OsApiError("UDS sendmsg failed: {}", std::strerror(errno)); }

    /*
     * The kernel consumes the ancillary descriptors together with the first
     * byte, so they are delivered exactly once even on a partial write; the
     * remainder goes out as plain data.
     */
    std::size_t done = static_cast<std::size_t>(sent);
    while (done < total) {
        std::array<iovec, 2> rest{};
        std::size_t iovCount = 0;
        if (done < header.size()) {
            rest[iovCount].iov_base = header.data() + done;
            rest[iovCount].iov_len = header.size() - done;
            ++iovCount;
            if (!payload.empty()) {
                rest[iovCount].iov_base = const_cast<std::byte*>(payload.data());
                rest[iovCount].iov_len = payload.size();
                ++iovCount;
            }
        } else {
            const std::size_t offset = done - header.size();
            rest[iovCount].iov_base = const_cast<std::byte*>(payload.data()) + offset;
            rest[iovCount].iov_len = payload.size() - offset;
            ++iovCount;
        }
        msghdr restMsg{};
        restMsg.msg_iov = rest.data();
        restMsg.msg_iovlen = iovCount;
        ssize_t moved = 0;
        do {
            moved = ::sendmsg(fd_, &restMsg, MSG_NOSIGNAL);
        } while (moved < 0 && errno == EINTR);
        if (moved < 0) {
            return Status::OsApiError("UDS sendmsg failed: {}", std::strerror(errno));
        }
        done += static_cast<std::size_t>(moved);
    }
    return Status::Ok();
}

Status UdsConnection::ReceiveFrame(std::vector<std::byte>& payload, std::vector<int>& receivedFds)
{
    if (fd_ < 0) { return Status::InvalidParam("UdsConnection: connection is not open"); }
    payload.clear();
    receivedFds.clear();

    std::array<std::byte, kHeaderSize> header{};
    std::size_t headerGot = 0;
    while (headerGot < header.size()) {
        ChunkResult chunk =
            ReceiveChunk(fd_, header.data() + headerGot, header.size() - headerGot, receivedFds);
        if (chunk.status.Failure()) { return std::move(chunk.status); }
        if (chunk.peerClosed) {
            return Status::General("UDS connection closed by peer while receiving a frame header");
        }
        headerGot += chunk.received;
    }

    const std::uint32_t length = GetLe32(header.data());
    if (length > kUdsMaxFrameSize) {
        return Status::InvalidParam("UDS frame length of {} bytes exceeds the {} byte limit",
                                    length, kUdsMaxFrameSize);
    }

    payload.resize(length);
    std::size_t payloadGot = 0;
    while (payloadGot < payload.size()) {
        ChunkResult chunk = ReceiveChunk(fd_, payload.data() + payloadGot,
                                         payload.size() - payloadGot, receivedFds);
        if (chunk.status.Failure()) { return std::move(chunk.status); }
        if (chunk.peerClosed) {
            return Status::General("UDS connection closed by peer mid-frame at byte {} of {}",
                                   payloadGot, payload.size());
        }
        payloadGot += chunk.received;
    }
    return Status::Ok();
}

void UdsConnection::Close() noexcept
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

UdsListener::~UdsListener() { Close(); }

UdsListener::UdsListener(UdsListener&& other) noexcept
    : fd_{std::exchange(other.fd_, -1)}, path_{std::move(other.path_)}
{
    other.path_.clear();
}

UdsListener& UdsListener::operator=(UdsListener&& other) noexcept
{
    if (this != &other) {
        Close();
        fd_ = std::exchange(other.fd_, -1);
        path_ = std::move(other.path_);
        other.path_.clear();
    }
    return *this;
}

Status UdsListener::Listen(const std::string& path)
{
    Close();
    if (Status st = CheckSocketPath(path); st.Failure()) { return st; }
    const sockaddr_un addr = MakeAddress(path);

    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return Status::OsApiError("UDS socket creation failed: {}", std::strerror(errno));
    }
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
        Status st = Status::OsApiError("UDS bind to '{}' failed: {}", path, std::strerror(errno));
        ::close(fd);
        return st;
    }
    if (::listen(fd, SOMAXCONN) < 0) {
        Status st = Status::OsApiError("UDS listen on '{}' failed: {}", path, std::strerror(errno));
        ::close(fd);
        return st;
    }
    fd_ = fd;
    path_ = path;
    return Status::Ok();
}

Expected<UdsConnection> UdsListener::Accept()
{
    if (fd_ < 0) { return Status::InvalidParam("UdsListener: listener is not open"); }
    int conn = -1;
    do {
        conn = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
    } while (conn < 0 && errno == EINTR);
    if (conn < 0) {
        return Status::OsApiError("UDS accept on '{}' failed: {}", path_, std::strerror(errno));
    }
    return UdsConnection{conn};
}

void UdsListener::Close() noexcept
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    path_.clear();
}

}  // namespace UC::Transfer
