/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "status/status.h"

namespace UC {

class FdSocket {
    std::atomic<int32_t> sock_{-1};

    static void FillAbstractAddr(sockaddr_un& addr, const std::string& name)
    {
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        addr.sun_path[0] = '\0';
        auto len = std::min(name.size(), sizeof(addr.sun_path) - 1);
        std::memcpy(addr.sun_path + 1, name.data(), len);
    }

public:
    FdSocket() = default;
    ~FdSocket() { Close(); }
    FdSocket(const FdSocket&) = delete;
    FdSocket& operator=(const FdSocket&) = delete;

    Status Listen(const std::string& name)
    {
        Close();
        if (name.empty() || name.size() >= sizeof(sockaddr_un{}.sun_path)) {
            return Status::InvalidParam("invalid abstract socket name");
        }
        auto fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) { return Status::OsApiError("socket failed"); }
        sock_.store(fd, std::memory_order_release);
        sockaddr_un addr{};
        FillAbstractAddr(addr, name);
        auto len = static_cast<socklen_t>(sizeof(sa_family_t) + 1 + name.size());
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), len) != 0) {
            auto error = errno;
            Close();
            if (error == EADDRINUSE) { return Status::DuplicateKey(); }
            return Status::OsApiError("bind failed");
        }
        if (::listen(fd, 256) != 0) {
            Close();
            return Status::OsApiError("listen failed");
        }
        return Status::OK();
    }

    Status Connect(const std::string& name)
    {
        Close();
        if (name.empty() || name.size() >= sizeof(sockaddr_un{}.sun_path)) {
            return Status::InvalidParam("invalid abstract socket name");
        }
        auto fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) { return Status::OsApiError("socket failed"); }
        sock_.store(fd, std::memory_order_release);
        sockaddr_un addr{};
        FillAbstractAddr(addr, name);
        auto len = static_cast<socklen_t>(sizeof(sa_family_t) + 1 + name.size());
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), len) != 0) {
            Close();
            return Status::OsApiError("connect failed");
        }
        return Status::OK();
    }

    Status AcceptAndSend(int32_t fdToSend)
    {
        auto listenFd = sock_.load(std::memory_order_acquire);
        auto conn = static_cast<int32_t>(::accept(listenFd, nullptr, nullptr));
        if (conn < 0) { return Status::OsApiError("accept failed"); }

        char byte{'x'};
        iovec iov{&byte, 1};
        char control[CMSG_SPACE(sizeof(int32_t))]{};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        auto* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int32_t));
        std::memcpy(CMSG_DATA(cmsg), &fdToSend, sizeof(fdToSend));
        auto sent = ::sendmsg(conn, &msg, MSG_NOSIGNAL);
        ::close(conn);
        if (sent < 0) { return Status::OsApiError("sendmsg failed"); }
        return Status::OK();
    }

    Status RecvFd(int32_t& fdOut)
    {
        fdOut = -1;
        char byte{};
        iovec iov{&byte, 1};
        char control[CMSG_SPACE(sizeof(int32_t))]{};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
        auto fd = sock_.load(std::memory_order_acquire);
        auto received = ::recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
        if (received <= 0) { return Status::OsApiError("recvmsg failed"); }
        auto* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS ||
            cmsg->cmsg_len != CMSG_LEN(sizeof(int32_t)) || (msg.msg_flags & MSG_CTRUNC) != 0) {
            return Status::OsApiError("missing or truncated descriptor");
        }
        std::memcpy(&fdOut, CMSG_DATA(cmsg), sizeof(fdOut));
        if (fdOut < 0) { return Status::OsApiError("invalid descriptor"); }
        return Status::OK();
    }

    void Close()
    {
        auto fd = sock_.exchange(-1, std::memory_order_acq_rel);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
            ::close(fd);
        }
    }
};

}  // namespace UC
