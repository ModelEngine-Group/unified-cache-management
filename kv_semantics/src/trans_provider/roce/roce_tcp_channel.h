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
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "kv_types.h"

namespace kv {
namespace roce {

// Out-of-band TCP channel used to exchange queue pair information before any
// verbs object exists.  A queue pair cannot send or receive while it is in the
// INIT state, so the qpn/psn/gid/mtu handshake has to travel over a medium that
// does not depend on it.
//
// Only IPv4 literals are accepted for now; host name resolution can be added
// once the peer addressing scheme is fixed.
class TcpChannel {
public:
    TcpChannel() = default;

    ~TcpChannel();

    TcpChannel(TcpChannel&& other) noexcept;

    TcpChannel& operator=(TcpChannel&& other) noexcept;

    TcpChannel(const TcpChannel&) = delete;

    TcpChannel& operator=(const TcpChannel&) = delete;

    // Connects to remoteIp:port, optionally binding to localIp first.  The call
    // is non-blocking under the hood and gives up after timeoutMs.
    static Status Connect(const std::string& localIp, const std::string& remoteIp,
                          std::uint16_t port, std::uint32_t timeoutMs, TcpChannel& channel,
                          std::string& error);

    // Creates a listening socket on bindIp:port.  An empty bindIp listens on all
    // local addresses.
    static Status Listen(const std::string& bindIp, std::uint16_t port, std::uint16_t backlog,
                         TcpChannel& listener, std::string& error);

    // Accepts one connection from a listening channel, waiting at most timeoutMs.
    static Status Accept(const TcpChannel& listener, std::uint32_t timeoutMs, TcpChannel& accepted,
                         std::string& error);

    bool Valid() const noexcept;

    int Fd() const noexcept;

    // "ip:port" of the peer, or an empty string when it cannot be resolved.
    std::string PeerAddress() const;

    // Full-duplex framing helpers used by the handshake state machine.  Both
    // loop until the whole buffer is transferred or the deadline expires.
    Status SendAll(const std::uint8_t* data, std::size_t size, std::uint32_t timeoutMs,
                   std::string& error) const;

    Status RecvAll(std::uint8_t* data, std::size_t size, std::uint32_t timeoutMs,
                   std::string& error) const;

    // Tuning applied to every socket this class produces.  The buffers are
    // requested as 1 MiB; the kernel doubles the request and clamps it to
    // net.core.{r,w}mem_max, so the effective value can be smaller.
    static constexpr std::size_t kSocketBufferBytes = 1U << 20U;

    // Maximum time unacknowledged transmitted data may stay in flight (ms).
    static constexpr std::uint32_t kUserTimeoutMs = 5000U;

    // Effective values read back from the socket; zero when it is not open.
    std::size_t SendBufferBytes() const;

    std::size_t ReceiveBufferBytes() const;

    std::uint32_t UserTimeoutMs() const;

    void Close() noexcept;

private:
    explicit TcpChannel(int fd);

    // SO_SNDBUF/SO_RCVBUF = 1 MiB and, for connected sockets, TCP_USER_TIMEOUT
    // = 5 s.  Applied before connect() so the peer never sees an untuned
    // connection.
    static Status ApplyTuning(int fd, bool includeUserTimeout, std::string& error);

    int fd_{-1};
};

}  // namespace roce
}  // namespace kv
