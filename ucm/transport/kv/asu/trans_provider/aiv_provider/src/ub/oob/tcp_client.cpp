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

#include "src/ub/oob/tcp_client.h"
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/protocol/ub_protocol.h"
#include "src/protocol/xrt_rdma_neg_msg.h"
#include "src/ub/log.h"
#include "src/ub/oob/codec/cm_codec.h"
#include "src/ub/oob/codec/xrt_neg_codec.h"

namespace umc::comm::oob {

namespace nc = ::umc::comm::oob::negcodec;

namespace {

bool SendAll(int fd, const void* buf, std::size_t len)
{
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    std::size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::send(fd, p, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool RecvAll(int fd, void* buf, std::size_t len)
{
    uint8_t* p = static_cast<uint8_t*>(buf);
    std::size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::recv(fd, p, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  // EOF
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool SetTimeout(int fd, uint32_t ms)
{
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) return false;
    if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) return false;
    return true;
}

int ConnectTo(const std::string& host, uint16_t port, uint32_t timeoutMs)
{
    const uint32_t effectiveTimeoutMs = timeoutMs == 0 ? 3000 : timeoutMs;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* res = nullptr;
    std::string portStr = std::to_string(port);
    int rc = ::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        UB_LOG_ERROR("getaddrinfo({}:{}) failed: {}", host.c_str(), port, gai_strerror(rc));
        return -1;
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(effectiveTimeoutMs);
    int fd = -1;
    int lastError = ETIMEDOUT;
    for (const addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            lastError = errno;
            continue;
        }
        const int oldFlags = ::fcntl(fd, F_GETFL, 0);
        if (oldFlags < 0 || ::fcntl(fd, F_SETFL, oldFlags | O_NONBLOCK) != 0) {
            lastError = errno;
            ::close(fd);
            fd = -1;
            continue;
        }

        int connectRc = ::connect(fd, ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen));
        if (connectRc != 0 && errno != EINPROGRESS && errno != EWOULDBLOCK) {
            lastError = errno;
            ::close(fd);
            fd = -1;
            continue;
        }
        while (connectRc != 0) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                lastError = ETIMEDOUT;
                break;
            }
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            pollfd pfd{};
            pfd.fd = fd;
            pfd.events = POLLOUT;
            const int pollMs = static_cast<int>(std::min<int64_t>(
                std::numeric_limits<int>::max(), std::max<int64_t>(1, remaining.count())));
            const int pollRc = ::poll(&pfd, 1, pollMs);
            if (pollRc < 0 && errno == EINTR) continue;
            if (pollRc <= 0) {
                lastError = pollRc == 0 ? ETIMEDOUT : errno;
                break;
            }
            int socketError = 0;
            socklen_t socketErrorLen = sizeof(socketError);
            if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &socketErrorLen) != 0) {
                lastError = errno;
                break;
            }
            if (socketError != 0) {
                lastError = socketError;
                break;
            }
            connectRc = 0;
        }
        if (connectRc == 0) {
            (void)::fcntl(fd, F_SETFL, oldFlags);
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        UB_LOG_ERROR("connect({}:{}) failed: {}", host.c_str(), port, std::strerror(lastError));
        return -1;
    }
    SetTimeout(fd, effectiveTimeoutMs);
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

void LogCapReq(const char* tag, const xrt_rdma_neg_msg_t& msg, const xrt_neg_cap_req& cap)
{
    UB_LOG_DEBUG(
        "NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}} "
        "cap_req{{major={} minor={} kato={} private_len={}}}",
        tag, msg.head.crc, msg.head.ver, msg.head.cmd, msg.head.len, cap.major_version,
        cap.minor_version, cap.kato, msg.body.conn_auth.private_len);
}

void LogCapRsp(const char* tag, const xrt_rdma_neg_msg_t& msg, const xrt_neg_cap_rsp& cap)
{
    UB_LOG_DEBUG(
        "NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}} "
        "cap_rsp{{major={} minor={} controller_id={} queue_num={} ioq_depth={} "
        "sr_mdts={}KB key_length={} controller_cap=0x{:x} private_len={}}}",
        tag, msg.head.crc, msg.head.ver, msg.head.cmd, msg.head.len, cap.major_version,
        cap.minor_version, cap.controller_id, cap.queue_num, cap.ioq_depth, cap.sr_mdts,
        cap.key_length, cap.controller_cap, msg.body.conn_auth.private_len);
}

void LogUbInfo(const char* tag, const xrt_rdma_neg_msg_t& msg, const kv_cm_conn_ub_info& info)
{
    UB_LOG_DEBUG(
        "NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}} "
        "ub_info{{eid0=0x{:02x} eid15=0x{:02x} qp_key_size={} mem_key_size={} "
        "token_id={} token_value={} remote_addr=0x{:x} remote_size={} "
        "net_addr_kind={} vlan={} rm_uasid={} rm_jetty_id={} conn_mode={}}}",
        tag, msg.head.crc, msg.head.ver, msg.head.cmd, msg.head.len,
        static_cast<unsigned>(info.hccp_eid_raw[0]), static_cast<unsigned>(info.hccp_eid_raw[15]),
        info.qp_key_size, info.mem_key_size, info.token_id, info.token_value,
        static_cast<unsigned long long>(info.remote_addr),
        static_cast<unsigned long long>(info.remote_size), info.net_addr_kind,
        static_cast<unsigned long long>(info.uboe_vlan), info.rm_uasid, info.rm_jetty_id,
        info.conn_mode);
}

void LogHeadOnly(const char* tag, const xrt_rdma_neg_msg_t& msg)
{
    UB_LOG_DEBUG("NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}}", tag, msg.head.crc,
                 msg.head.ver, msg.head.cmd, msg.head.len);
}

}  // namespace

TcpClientOobTransport::TcpClientOobTransport(Config cfg) : cfg_(std::move(cfg)) {}
TcpClientOobTransport::~TcpClientOobTransport() = default;

UbStatus TcpClientOobTransport::CheckKvLimits(uint32_t keyLen, uint64_t valueBytes) const
{
    if (!hasNegCaps_) return UbStatus::Ok();
    const auto& cap = remoteCap_.neg;
    if (cap.key_length != 0 && keyLen > cap.key_length) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "key length " + std::to_string(keyLen) +
                                                                " exceeds negotiated key_length " +
                                                                std::to_string(cap.key_length));
    }
    const uint64_t maxBytes = static_cast<uint64_t>(cap.sr_mdts) * 1024ull;
    if (cap.sr_mdts != 0 && valueBytes > maxBytes) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch,
                        "value bytes " + std::to_string(valueBytes) +
                            " exceeds negotiated sr_mdts " + std::to_string(maxBytes) + "B");
    }
    return UbStatus::Ok();
}

UbStatus TcpClientOobTransport::Negotiate(RemoteEndpointSet* out)
{
    if (out == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "out null");
    *out = RemoteEndpointSet{};
    hasNegCaps_ = false;

    int fd = ConnectTo(cfg_.host, cfg_.port, cfg_.connectTimeoutMs);
    if (fd < 0) {
        return UbStatus(UbErrorCode::OobConnectFailed,
                        "connect " + cfg_.host + ":" + std::to_string(cfg_.port) + " failed");
    }
    auto closer = [&]() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    };

    SetTimeout(fd, cfg_.ioTimeoutMs);

    xrt_neg_cap_req clientNeg{};
    clientNeg.major_version = 1;
    clientNeg.minor_version = 0;
    clientNeg.kato = static_cast<uint16_t>(cfg_.kato);

    kv_cm_conn_ub_info clientUbInfo{};
    cm::FillUbInfo(&clientUbInfo, cfg_.local);

    xrt_rdma_neg_msg_t msg{};

    nc::FillConnAuthReq(&msg, clientNeg);
    LogCapReq("TCP C->S CONN_AUTH", msg, clientNeg);
    if (!SendAll(fd, &msg, sizeof(msg))) {
        closer();
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_AUTH failed");
    }
    std::memset(&msg, 0, sizeof(msg));
    if (!RecvAll(fd, &msg, sizeof(msg))) {
        closer();
        return UbStatus(UbErrorCode::OobTransportClosed, "recv CONN_AUTH resp failed");
    }
    if (!nc::VerifyNegCrc(msg)) {
        closer();
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "CONN_AUTH resp crc mismatch");
    }
    xrt_neg_cap_rsp serverNeg{};
    if (msg.head.cmd != XRT_RDMA_NEG_CONN_AUTH || !nc::ParseConnAuthRsp(msg, &serverNeg)) {
        closer();
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "expect CONN_AUTH resp");
    }
    LogCapRsp("TCP S->C CONN_AUTH", msg, serverNeg);
    localCap_.neg = clientNeg;
    remoteCap_.neg = serverNeg;
    if (serverNeg.major_version != clientNeg.major_version) {
        closer();
        return UbStatus(
            UbErrorCode::OobCmVersionRejected,
            "server neg major_version mismatch (client=" + std::to_string(clientNeg.major_version) +
                " server=" + std::to_string(serverNeg.major_version) + ")");
    }
    UB_LOG_DEBUG(
        "TcpClientOobTransport(negV2): CONN_AUTH ok, server controller_id={} "
        "(major={} minor={} ioq_depth={})",
        serverNeg.controller_id, serverNeg.major_version, serverNeg.minor_version,
        serverNeg.ioq_depth);

    nc::FillConnUb(&msg, XRT_RDMA_NEG_CONN_REQ, clientUbInfo);
    LogUbInfo("TCP C->S CONN_REQ", msg, clientUbInfo);
    if (!SendAll(fd, &msg, sizeof(msg))) {
        closer();
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_REQ failed");
    }
    std::memset(&msg, 0, sizeof(msg));
    if (!RecvAll(fd, &msg, sizeof(msg))) {
        closer();
        return UbStatus(UbErrorCode::OobTransportClosed, "recv CONN_RSP failed");
    }
    if (!nc::VerifyNegCrc(msg)) {
        closer();
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "CONN_RSP crc mismatch");
    }
    if (msg.head.cmd != XRT_RDMA_NEG_CONN_RSP) {
        closer();
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "expect CONN_RSP");
    }
    const kv_cm_conn_ub_info serverUbInfo = msg.body.conn_ub.ub_info;
    LogUbInfo("TCP S->C CONN_RSP", msg, serverUbInfo);
    localCap_.ub_info = clientUbInfo;
    remoteCap_.ub_info = serverUbInfo;

    nc::FillConnDone(&msg);
    LogHeadOnly("TCP C->S CONN_DONE", msg);
    if (!SendAll(fd, &msg, sizeof(msg))) {
        closer();
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_DONE failed");
    }

    closer();

    auto st =
        cm::UbInfoToRemoteSet(serverUbInfo, kKvCmMinorVersionV1_3, cfg_.host, /*jettyId=*/0, out);
    if (st.IsError()) return st;

    hasNegCaps_ = true;

    UB_LOG_DEBUG(
        "TcpClientOobTransport(negV2): 3-stage handshake done, controller_id={} "
        "peer eid[0]=0x{:02x} peer.net_addr_kind={}",
        remoteCap_.neg.controller_id, static_cast<unsigned>(serverUbInfo.hccp_eid_raw[0]),
        static_cast<unsigned>(serverUbInfo.net_addr_kind));
    return UbStatus::Ok();
}

}  // namespace umc::comm::oob
