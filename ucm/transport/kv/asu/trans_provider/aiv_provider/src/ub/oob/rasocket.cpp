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

#include "src/ub/oob/rasocket.h"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <vector>
#include "src/protocol/ub_protocol.h"
#include "src/protocol/xrt_rdma_neg_msg.h"
#include "src/runtime/hccp_v2_loader.h"
#include "src/ub/log.h"
#include "src/ub/oob/codec/cm_codec.h"
#include "src/ub/oob/codec/xrt_neg_codec.h"

namespace umc::comm::oob {

namespace v2 = ::umc::comm::v2;
namespace nc = ::umc::comm::oob::negcodec;

namespace {

using Clock = std::chrono::steady_clock;

bool Expired(Clock::time_point start, uint32_t timeoutMs)
{
    if (timeoutMs == 0) return false;
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    return elapsed >= static_cast<long>(timeoutMs);
}

bool FillRdev(v2::Rdev* rdev, uint32_t phyId, const std::string& localIp)
{
    std::memset(rdev, 0, sizeof(*rdev));
    rdev->phyId = phyId;
    if (::inet_pton(AF_INET, localIp.c_str(), &rdev->localIp.addr) == 1) {
        rdev->family = AF_INET;
        return true;
    }
    if (::inet_pton(AF_INET6, localIp.c_str(), &rdev->localIp.addr6) == 1) {
        rdev->family = AF_INET6;
        return true;
    }
    return false;
}

bool RaSendAll(const void* fd, const void* buf, std::size_t len, uint32_t timeoutMs)
{
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    std::size_t sent = 0;
    auto start = Clock::now();
    while (sent < len && !Expired(start, timeoutMs)) {
        unsigned long long done = 0;
        int rc = v2::DlHccpV2Api::RaSocketSend(fd, p + sent,
                                               static_cast<unsigned long long>(len - sent), &done);
        if (rc == 0 && done > 0) {
            sent += static_cast<std::size_t>(done);
            continue;
        }
        if (rc == v2::SOCK_EAGAIN || (rc == 0 && done == 0)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        UB_LOG_ERROR("RaSocketSend failed rc={} sent={}/{}", rc, sent, len);
        return false;
    }
    return sent == len;
}

bool RaRecvAll(const void* fd, void* buf, std::size_t len, uint32_t timeoutMs)
{
    uint8_t* p = static_cast<uint8_t*>(buf);
    std::size_t recvd = 0;
    auto start = Clock::now();
    while (recvd < len && !Expired(start, timeoutMs)) {
        unsigned long long done = 0;
        int rc = v2::DlHccpV2Api::RaSocketRecv(fd, p + recvd,
                                               static_cast<unsigned long long>(len - recvd), &done);
        if (rc == 0 && done > 0) {
            recvd += static_cast<std::size_t>(done);
            continue;
        }
        if (rc == v2::SOCK_EAGAIN) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (rc == v2::SOCK_CLOSE || rc == v2::SOCK_ESOCKCLOSED || (rc == 0 && done == 0)) {
            UB_LOG_ERROR("RaSocketRecv connection closed recvd={}/{}", recvd, len);
            return false;
        }
        UB_LOG_ERROR("RaSocketRecv failed rc={} recvd={}/{}", rc, recvd, len);
        return false;
    }
    return recvd == len;
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

RaSocketOobTransport::RaSocketOobTransport(Config cfg) : cfg_(std::move(cfg)) {}
RaSocketOobTransport::~RaSocketOobTransport() = default;

bool RaSocketOobTransport::Available()
{
    if (!v2::DlHccpV2Api::IsLoaded()) { (void)v2::DlHccpV2Api::LoadLibrary(); }
    return v2::DlHccpV2Api::SocketApiAvailable();
}

bool RaSocketOobTransport::QueryLocalNicIp(uint32_t phyId, std::string* outIp)
{
    if (outIp == nullptr) return false;
    outIp->clear();
    if (!v2::DlHccpV2Api::IsLoaded()) {
        if (v2::DlHccpV2Api::LoadLibrary() != UbErrorCode::Ok) return false;
    }

    if (v2::DlHccpV2Api::SocketVnicIpApiAvailable()) {
        unsigned int id = phyId;
        v2::IpInfo info{};
        const int rc =
            v2::DlHccpV2Api::RaSocketGetVnicIpInfos(phyId, v2::PHY_ID_VNIC_IP, &id, 1, &info);
        char vnicIp[INET_ADDRSTRLEN] = {0};
        if (rc == 0 && info.family == AF_INET && info.ip.addr.s_addr != htonl(INADDR_ANY) &&
            ::inet_ntop(AF_INET, &info.ip.addr, vnicIp, sizeof(vnicIp)) != nullptr) {
            *outIp = vnicIp;
            UB_LOG_INFO(
                "QueryLocalNicIp: selected phyId={} VNIC ip={} "
                "(RaSocketGetVnicIpInfos)",
                phyId, outIp->c_str());
            return true;
        }
        UB_LOG_WARN(
            "QueryLocalNicIp: RaSocketGetVnicIpInfos phyId={} rc={} "
            "family={} returned no valid IPv4 address; falling back to RaGetIfaddrs",
            phyId, rc, info.family);
    } else {
        UB_LOG_WARN(
            "QueryLocalNicIp: RaSocketGetVnicIpInfos is unavailable for phyId={}; "
            "falling back to RaGetIfaddrs",
            phyId);
    }

    v2::RaGetIfAttr attr{};
    attr.phyId = phyId;
    attr.nicPosition = v2::NETWORK_OFFLINE;
    std::vector<v2::InterfaceInfo> infos;
    auto queryInterfaces = [&](bool isAll) {
        attr.isAll = isAll;
        unsigned int count = 0;
        const int numRc = v2::DlHccpV2Api::RaGetIfNum(&attr, &count);
        if (numRc != 0 || count == 0) {
            UB_LOG_DEBUG("QueryLocalNicIp: RaGetIfNum phyId={} isAll={} rc={} count={}", phyId,
                         isAll ? 1u : 0u, numRc, count);
            return false;
        }
        count = std::min(count, static_cast<unsigned int>(v2::MAX_INTERFACE_NUM));
        infos.assign(count, v2::InterfaceInfo{});
        unsigned int actual = count;
        const int addrRc = v2::DlHccpV2Api::RaGetIfAddrs(&attr, infos.data(), &actual);
        if (addrRc != 0 || actual == 0) {
            UB_LOG_DEBUG("QueryLocalNicIp: RaGetIfAddrs phyId={} isAll={} rc={} actual={}", phyId,
                         isAll ? 1u : 0u, addrRc, actual);
            infos.clear();
            return false;
        }
        if (actual < infos.size()) { infos.resize(actual); }
        return true;
    };

    bool usedAllFallback = false;
    if (!queryInterfaces(/*isAll=*/false)) {
        UB_LOG_WARN("QueryLocalNicIp: phyId={} exact query was empty; trying all eth/bond devices",
                    phyId);
        usedAllFallback = true;
        if (!queryInterfaces(/*isAll=*/true)) {
            UB_LOG_WARN("QueryLocalNicIp: phyId={} has no queryable NIC or all-IP is unsupported",
                        phyId);
            return false;
        }
    }

    char buf[INET_ADDRSTRLEN] = {0};
    char maskBuf[INET_ADDRSTRLEN] = {0};
    const std::string expectedEth = "eth" + std::to_string(phyId);
    const std::string expectedBond = "bond" + std::to_string(phyId);

    for (const auto& info : infos) {
        const std::size_t nameLen = ::strnlen(info.ifName, sizeof(info.ifName));
        const std::string ifName(info.ifName, nameLen);
        if (info.family == AF_INET &&
            ::inet_ntop(AF_INET, &info.ifAddr.ip.addr, buf, sizeof(buf)) != nullptr) {
            const char* mask = ::inet_ntop(AF_INET, &info.ifAddr.mask, maskBuf, sizeof(maskBuf));
            UB_LOG_DEBUG("QueryLocalNicIp: candidate phyId={} ifName={} ip={} mask={}", phyId,
                         ifName.c_str(), buf, mask != nullptr ? mask : "<invalid>");
        }
    }

    enum class Ipv4Preference {
        Any,
        ExactEth,
        AnyEth,
        ExactBond,
    };
    auto selectIpv4 = [&](Ipv4Preference preference) {
        for (const auto& info : infos) {
            const std::size_t nameLen = ::strnlen(info.ifName, sizeof(info.ifName));
            const std::string ifName(info.ifName, nameLen);
            const bool isEth = ifName.rfind("eth", 0) == 0;
            if ((preference == Ipv4Preference::ExactEth && ifName != expectedEth) ||
                (preference == Ipv4Preference::AnyEth && !isEth) ||
                (preference == Ipv4Preference::ExactBond && ifName != expectedBond)) {
                continue;
            }
            if (info.family == AF_INET && info.ifAddr.ip.addr.s_addr != htonl(INADDR_ANY) &&
                ::inet_ntop(AF_INET, &info.ifAddr.ip.addr, buf, sizeof(buf)) != nullptr) {
                UB_LOG_INFO(
                    "QueryLocalNicIp: selected phyId={} ifName={} ip={} "
                    "(IPv4{})",
                    phyId, ifName.c_str(), buf, usedAllFallback ? ", all-IP fallback" : "");
                *outIp = buf;
                return true;
            }
        }
        return false;
    };
    if (!usedAllFallback && selectIpv4(Ipv4Preference::Any)) { return true; }
    if (usedAllFallback &&
        (selectIpv4(Ipv4Preference::ExactEth) || selectIpv4(Ipv4Preference::AnyEth))) {
        return true;
    }
    if (usedAllFallback) {
        UB_LOG_WARN(
            "QueryLocalNicIp: phyId={} found no usable eth management IPv4 address; "
            "falling back to bond or the first IPv4 candidate",
            phyId);
    }
    if (selectIpv4(Ipv4Preference::ExactBond) || selectIpv4(Ipv4Preference::Any)) { return true; }
    UB_LOG_WARN("QueryLocalNicIp: phyId={} has {} interfaces but no valid IPv4 address", phyId,
                infos.size());
    return false;
}

UbStatus RaSocketOobTransport::CheckKvLimits(uint32_t keyLen, uint64_t valueBytes) const
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

UbStatus RaSocketOobTransport::Negotiate(RemoteEndpointSet* out)
{
    if (out == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "out null");
    *out = RemoteEndpointSet{};
    hasNegCaps_ = false;

    if (!Available()) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "RaSocket API unavailable (need CANN 9.0 libra/hccl_v2; "
                        "fallback to TcpClient)");
    }

    v2::Rdev rdev{};
    if (!FillRdev(&rdev, cfg_.phyId, cfg_.localIp)) {
        return UbStatus(UbErrorCode::InvalidArgument, "RaSocket localIp invalid: " + cfg_.localIp);
    }
    void* sockHandle = nullptr;
    int rc = v2::DlHccpV2Api::RaSocketInit(v2::NETWORK_OFFLINE, rdev, &sockHandle);
    if (rc != 0 || sockHandle == nullptr) {
        return UbStatus(UbErrorCode::OobConnectFailed, "RaSocketInit rc=" + std::to_string(rc));
    }
    auto deinit = [&]() -> UbStatus {
        if (!sockHandle) { return UbStatus::Ok(); }
        const int deinitRc = v2::DlHccpV2Api::RaSocketDeinit(sockHandle);
        sockHandle = nullptr;
        return deinitRc == 0 ? UbStatus::Ok()
                             : UbStatus(UbErrorCode::OobTransportClosed,
                                        "RaSocketDeinit rc=" + std::to_string(deinitRc));
    };
    auto logCleanupFailure = [](const UbStatus& status) {
        if (status.IsError()) {
            UB_LOG_WARN("RaSocketOobTransport cleanup failed: {}", status.Message().c_str());
        }
    };

    v2::SocketConnectInfoT conn{};
    conn.socketHandle = sockHandle;
    if (::inet_pton(AF_INET, cfg_.host.c_str(), &conn.remoteIp.addr) != 1 &&
        ::inet_pton(AF_INET6, cfg_.host.c_str(), &conn.remoteIp.addr6) != 1) {
        logCleanupFailure(deinit());
        return UbStatus(UbErrorCode::InvalidArgument,
                        "RaSocket remoteIp invalid (v4/v6): " + cfg_.host);
    }
    conn.port = cfg_.port;
    nc::BuildHccpTag(conn.tag, sizeof(conn.tag));

    rc = v2::DlHccpV2Api::RaSocketBatchConnect(&conn, 1);
    if (rc != 0 && rc != v2::SOCK_EAGAIN) {
        logCleanupFailure(deinit());
        return UbStatus(UbErrorCode::OobConnectFailed,
                        "RaSocketBatchConnect rc=" + std::to_string(rc));
    }
    UB_LOG_DEBUG("RaSocketOobTransport: BatchConnect issued (3-stage) waiting whitelist handshake");

    void* fdHandle = nullptr;
    {
        v2::SocketInfoT info{};
        info.socketHandle = sockHandle;
        info.remoteIp = conn.remoteIp;
        std::memcpy(info.tag, conn.tag,
                    sizeof(info.tag) < sizeof(conn.tag) ? sizeof(info.tag) : sizeof(conn.tag));
        auto start = Clock::now();
        while (!Expired(start, cfg_.connectTimeoutMs)) {
            unsigned int connectedNum = 0;
            rc = v2::DlHccpV2Api::RaGetSockets(/*role=client*/ 1, &info, 1, &connectedNum);
            if (rc == 0 && connectedNum == 1 && info.fdHandle != nullptr &&
                info.status == v2::HCCP_SOCKET_CONNECTED) {
                fdHandle = info.fdHandle;
                break;
            }
            if (rc != 0 && rc != v2::SOCK_EAGAIN) {
                logCleanupFailure(deinit());
                return UbStatus(UbErrorCode::OobConnectFailed,
                                "RaGetSockets rc=" + std::to_string(rc));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (fdHandle == nullptr) {
        logCleanupFailure(deinit());
        return UbStatus(UbErrorCode::OobTimeout,
                        "RaSocket connect timeout (server may not consume tag / reply a5a5a)");
    }
    UB_LOG_DEBUG("RaSocketOobTransport: connected to {}:{} fdHandle={}", cfg_.host.c_str(),
                 cfg_.port, fdHandle);

    auto closeAll = [&]() -> UbStatus {
        UbStatus closeStatus = UbStatus::Ok();
        if (sockHandle && fdHandle) {
            v2::SocketCloseInfoT ci{};
            ci.socketHandle = sockHandle;
            ci.fdHandle = fdHandle;
            ci.disuseLinger = 0;
            const int closeRc = v2::DlHccpV2Api::RaSocketBatchClose(&ci, 1);
            if (closeRc != 0) {
                closeStatus = UbStatus(UbErrorCode::OobTransportClosed,
                                       "RaSocketBatchClose rc=" + std::to_string(closeRc));
            }
            fdHandle = nullptr;
        }
        const auto deinitStatus = deinit();
        return closeStatus.IsError() ? closeStatus : deinitStatus;
    };

    UbStatus st = NegotiateV2(fdHandle, out);
    const auto closeStatus = closeAll();
    return st.IsError() ? st : closeStatus;
}

UbStatus RaSocketOobTransport::NegotiateV2(void* fdHandle, RemoteEndpointSet* out)
{
    xrt_neg_cap_req clientNeg{};
    clientNeg.major_version = 1;
    clientNeg.minor_version = 0;
    clientNeg.kato = static_cast<uint16_t>(cfg_.kato);

    kv_cm_conn_ub_info clientUbInfo{};
    cm::FillUbInfo(&clientUbInfo, cfg_.local);

    xrt_rdma_neg_msg_t msg{};

    nc::FillConnAuthReq(&msg, clientNeg);
    LogCapReq("RASOCKET C->S CONN_AUTH", msg, clientNeg);
    if (!RaSendAll(fdHandle, &msg, sizeof(msg), cfg_.ioTimeoutMs)) {
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_AUTH failed");
    }
    std::memset(&msg, 0, sizeof(msg));
    if (!RaRecvAll(fdHandle, &msg, sizeof(msg), cfg_.ioTimeoutMs)) {
        return UbStatus(UbErrorCode::OobTransportClosed, "recv CONN_AUTH resp failed");
    }
    if (!nc::VerifyNegCrc(msg)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "CONN_AUTH resp crc mismatch");
    }
    xrt_neg_cap_rsp serverNeg{};
    if (msg.head.cmd != XRT_RDMA_NEG_CONN_AUTH || !nc::ParseConnAuthRsp(msg, &serverNeg)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "expect CONN_AUTH resp");
    }
    LogCapRsp("RASOCKET S->C CONN_AUTH", msg, serverNeg);
    localCap_.neg = clientNeg;
    remoteCap_.neg = serverNeg;
    if (serverNeg.major_version != clientNeg.major_version) {
        return UbStatus(
            UbErrorCode::OobCmVersionRejected,
            "server neg major_version mismatch (client=" + std::to_string(clientNeg.major_version) +
                " server=" + std::to_string(serverNeg.major_version) + ")");
    }
    UB_LOG_DEBUG(
        "RaSocketOobTransport(negV2): CONN_AUTH ok, server controller_id={} "
        "(major={} minor={} ioq_depth={})",
        serverNeg.controller_id, serverNeg.major_version, serverNeg.minor_version,
        serverNeg.ioq_depth);

    nc::FillConnUb(&msg, XRT_RDMA_NEG_CONN_REQ, clientUbInfo);
    LogUbInfo("RASOCKET C->S CONN_REQ", msg, clientUbInfo);
    if (!RaSendAll(fdHandle, &msg, sizeof(msg), cfg_.ioTimeoutMs)) {
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_REQ failed");
    }
    std::memset(&msg, 0, sizeof(msg));
    if (!RaRecvAll(fdHandle, &msg, sizeof(msg), cfg_.ioTimeoutMs)) {
        return UbStatus(UbErrorCode::OobTransportClosed, "recv CONN_RSP failed");
    }
    if (!nc::VerifyNegCrc(msg)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "CONN_RSP crc mismatch");
    }
    if (msg.head.cmd != XRT_RDMA_NEG_CONN_RSP) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "expect CONN_RSP");
    }
    const kv_cm_conn_ub_info serverUbInfo = msg.body.conn_ub.ub_info;
    LogUbInfo("RASOCKET S->C CONN_RSP", msg, serverUbInfo);
    localCap_.ub_info = clientUbInfo;
    remoteCap_.ub_info = serverUbInfo;

    nc::FillConnDone(&msg);
    LogHeadOnly("RASOCKET C->S CONN_DONE", msg);
    if (!RaSendAll(fdHandle, &msg, sizeof(msg), cfg_.ioTimeoutMs)) {
        return UbStatus(UbErrorCode::OobTransportClosed, "send CONN_DONE failed");
    }

    auto st =
        cm::UbInfoToRemoteSet(serverUbInfo, kKvCmMinorVersionV1_3, cfg_.host, /*jettyId=*/0, out);
    if (st.IsError()) return st;

    hasNegCaps_ = true;

    UB_LOG_DEBUG(
        "RaSocketOobTransport(negV2): 3-stage handshake done, controller_id={} "
        "peer eid[0]=0x{:02x}",
        remoteCap_.neg.controller_id, static_cast<unsigned>(serverUbInfo.hccp_eid_raw[0]));
    return UbStatus::Ok();
}

}  // namespace umc::comm::oob
