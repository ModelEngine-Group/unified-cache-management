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

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include "src/protocol/ub_protocol.h"
#include "src/ub/oob/tcp_client.h"  // TcpClientLocalEndpoint
#include "src/ub/oob/transport.h"
#include "src/ub/status.h"

namespace umc::comm::oob::cm {

inline uint8_t NetAddrKindToWire(NetAddrKind k)
{
    switch (k) {
        case NetAddrKind::None: return static_cast<uint8_t>(KV_NET_ADDR_NONE);
        case NetAddrKind::Uboe4: return static_cast<uint8_t>(KV_NET_ADDR_UBOE4);
        case NetAddrKind::Uboe6: return static_cast<uint8_t>(KV_NET_ADDR_UBOE6);
    }
    return static_cast<uint8_t>(KV_NET_ADDR_NONE);
}

inline NetAddrKind NetAddrKindFromWire(uint8_t w)
{
    switch (w) {
        case static_cast<uint8_t>(KV_NET_ADDR_NONE): return NetAddrKind::None;
        case static_cast<uint8_t>(KV_NET_ADDR_UBOE4): return NetAddrKind::Uboe4;
        case static_cast<uint8_t>(KV_NET_ADDR_UBOE6): return NetAddrKind::Uboe6;
        default: return NetAddrKind::None;
    }
}

inline void FillUbInfo(kv_cm_conn_ub_info* info, const TcpClientLocalEndpoint& local)
{
    std::memset(info, 0, sizeof(*info));
    std::memcpy(info->hccp_eid_raw, local.hccpEidRaw.data(), 16);
    std::memcpy(info->qp_key_raw, local.qpKeyRaw.data(), 64);
    info->qp_key_size = local.qpKeySize;
    std::memcpy(info->mem_key_raw, local.memKeyRaw.data(), 128);
    info->mem_key_size = local.memKeySize;
    info->token_id = local.tokenId;
    info->token_value = local.tokenValue;
    info->remote_addr = local.remoteAddr;
    info->remote_size = local.remoteSize;
    info->net_addr_kind = NetAddrKindToWire(local.localNetAddr.kind);
    info->prefix_len = local.localNetAddr.prefixLen;
    info->uboe_vlan = local.localNetAddr.vlan;
    std::memcpy(info->uboe_mac, local.localNetAddr.mac.data(), 6);
    if (local.localNetAddr.IsUboe()) {
        std::memcpy(info->hccp_eid_raw, local.localNetAddr.ipRaw.data(), 16);
    }
    info->mami_tp_handle = 0;
    info->mami_tag = local.tag;
    info->mami_tx_psn = local.txPsn;
    info->mami_rx_psn = 0;
    info->rm_uasid = local.uasid;
    info->rm_jetty_id = local.jettyId;
    info->rm_jetty_type = 1;  // TARGET_TYPE_JETTY
    info->rm_jetty_num = 1;
    info->conn_mode = local.connMode;
}

inline UbStatus UbInfoToRemoteSet(const kv_cm_conn_ub_info& info, uint8_t minorVersion,
                                  const std::string& peerName, uint32_t jettyId,
                                  RemoteEndpointSet* out)
{
    if (out == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "remote endpoint output is null");
    }
    if (info.qp_key_size > sizeof(info.qp_key_raw)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "remote qp key exceeds wire capacity");
    }
    if (info.mem_key_size > sizeof(info.mem_key_raw)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch,
                        "remote memory key exceeds wire capacity");
    }
    if (info.conn_mode > 1) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "remote connection mode is invalid");
    }
    if (info.remote_addr == 0 || info.remote_size == 0 ||
        info.remote_addr > std::numeric_limits<uint64_t>::max() - info.remote_size) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch, "remote memory range is invalid");
    }
    if (info.net_addr_kind != static_cast<uint8_t>(KV_NET_ADDR_NONE) &&
        info.net_addr_kind != static_cast<uint8_t>(KV_NET_ADDR_UBOE4) &&
        info.net_addr_kind != static_cast<uint8_t>(KV_NET_ADDR_UBOE6)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch,
                        "remote network address kind is invalid");
    }
    if ((info.net_addr_kind == static_cast<uint8_t>(KV_NET_ADDR_UBOE4) && info.prefix_len > 32) ||
        (info.net_addr_kind == static_cast<uint8_t>(KV_NET_ADDR_UBOE6) && info.prefix_len > 128)) {
        return UbStatus(UbErrorCode::OobCmProtocolMismatch,
                        "remote network prefix length is invalid");
    }
    out->schemaVersion = 1;
    out->clusterName.clear();
    out->protocol = "UB_PROTO";
    out->tokenPolicy = "PLAIN_TEXT";
    RemoteJettyDescriptor d;
    d.peerName = peerName;
    d.jettyId = jettyId;
    std::memcpy(d.hccpEidRaw.data(), info.hccp_eid_raw, 16);
    std::memcpy(d.qpKeyRaw.data(), info.qp_key_raw, 64);
    d.qpKeySize = info.qp_key_size;
    std::memcpy(d.memKeyRaw.data(), info.mem_key_raw, 128);
    d.memKeySize = info.mem_key_size;
    d.tokenId = info.token_id;
    d.tokenValue = info.token_value;
    d.remoteAddr = info.remote_addr;
    d.remoteSize = info.remote_size;
    d.netAddr.kind = NetAddrKindFromWire(info.net_addr_kind);
    d.netAddr.prefixLen = info.prefix_len;
    d.netAddr.vlan = info.uboe_vlan;
    std::memcpy(d.netAddr.mac.data(), info.uboe_mac, 6);
    if (d.netAddr.kind == NetAddrKind::Uboe4) {
        d.netAddr.family = 2;  // AF_INET
        std::memcpy(d.netAddr.ipRaw.data(), info.hccp_eid_raw, 16);
    } else if (d.netAddr.kind == NetAddrKind::Uboe6) {
        d.netAddr.family = 10;  // AF_INET6
        std::memcpy(d.netAddr.ipRaw.data(), info.hccp_eid_raw, 16);
    }
    if (minorVersion >= kKvCmMinorVersionV1_2) {
        d.peerTpHandle = info.mami_tp_handle;
        d.peerTxPsn = info.mami_tx_psn;
        d.tag = info.mami_tag;
    }
    if (minorVersion >= kKvCmMinorVersionV1_3) {
        d.connMode = info.conn_mode;
        d.uasid = info.rm_uasid;
        if (info.conn_mode != 0) { d.jettyId = info.rm_jetty_id; }
    }
    out->jetties.push_back(std::move(d));
    return UbStatus::Ok();
}

}  // namespace umc::comm::oob::cm
