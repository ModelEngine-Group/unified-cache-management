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
#include <cstring>
#include "src/protocol/xrt_neg_crc.h"
#include "src/protocol/xrt_rdma_neg_msg.h"

namespace umc::comm::oob::negcodec {

inline uint32_t ComputeCrc(const xrt_rdma_neg_msg_t& msg, uint32_t bodyLen)
{
    xrt_rdma_neg_msg_t tmp = msg;
    tmp.head.crc = 0;
    const std::size_t span = sizeof(tmp.head) + bodyLen;
    return ::umc::neg::Crc32Ieee(&tmp, span);
}

inline void FillNegHead(xrt_rdma_neg_msg_t* msg, uint8_t cmd, uint32_t bodyLen)
{
    msg->head.ver = static_cast<uint8_t>(XRT_NEG_MSG_VERSION);
    msg->head.cmd = cmd;
    std::memset(msg->head.pad, 0, sizeof(msg->head.pad));
    msg->head.len = bodyLen;
    msg->head.crc = 0;
    msg->head.crc = ComputeCrc(*msg, bodyLen);
}

inline bool ExpectedBodyLen(uint8_t cmd, uint32_t* bodyLen)
{
    switch (cmd) {
        case XRT_RDMA_NEG_CONN_AUTH:
            *bodyLen = static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_auth_t));
            return true;
        case XRT_RDMA_NEG_CONN_REQ:
        case XRT_RDMA_NEG_CONN_RSP:
            *bodyLen = static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_ub_t));
            return true;
        case XRT_RDMA_NEG_CONN_DONE: *bodyLen = 0; return true;
        case XRT_RDMA_NEG_CONN_FIN:
            *bodyLen = static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_fin_t));
            return true;
        case XRT_RDMA_NEG_CONN_HCCP: *bodyLen = kXrtHccpHeadLen; return true;
        default: return false;
    }
}

inline bool VerifyNegCrc(const xrt_rdma_neg_msg_t& msg)
{
    if (msg.head.ver != static_cast<uint8_t>(XRT_NEG_MSG_VERSION)) return false;
    for (uint8_t byte : msg.head.pad) {
        if (byte != 0) return false;
    }
    uint32_t expectedBodyLen = 0;
    if (!ExpectedBodyLen(msg.head.cmd, &expectedBodyLen) || msg.head.len != expectedBodyLen ||
        msg.head.len > XRT_MAX_NEG_BODY_SIZE) {
        return false;
    }
    return ComputeCrc(msg, msg.head.len) == msg.head.crc;
}

inline void BuildHccpTag(char* tag, std::size_t tagSize)
{
    std::memset(tag, 0, tagSize);
    if (tagSize < sizeof(xrt_rdma_neg_head_t)) return;
    xrt_rdma_neg_head_t head{};
    head.crc = 0;
    head.ver = static_cast<uint8_t>(XRT_NEG_MSG_VERSION);
    head.cmd = static_cast<uint8_t>(XRT_RDMA_NEG_CONN_HCCP);
    head.len = kXrtHccpHeadLen;  // 240
    std::memcpy(tag, &head, sizeof(head));
}

inline void FillConnAuthReq(xrt_rdma_neg_msg_t* msg, const xrt_neg_cap_req& cap)
{
    std::memset(&msg->body, 0, sizeof(msg->body));
    msg->body.conn_auth.cap = 0;
    msg->body.conn_auth.private_len = XRT_NEG_CAP_REQ_PRIVATE_LEN;
    std::memcpy(msg->body.conn_auth.private_data, &cap, sizeof(cap));
    FillNegHead(msg, XRT_RDMA_NEG_CONN_AUTH,
                static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_auth_t)));
}

inline void FillConnAuthRsp(xrt_rdma_neg_msg_t* msg, const xrt_neg_cap_rsp& cap)
{
    std::memset(&msg->body, 0, sizeof(msg->body));
    msg->body.conn_auth.cap = 0;
    msg->body.conn_auth.private_len = XRT_NEG_CAP_RSP_PRIVATE_LEN;
    std::memcpy(msg->body.conn_auth.private_data, &cap, sizeof(cap));
    FillNegHead(msg, XRT_RDMA_NEG_CONN_AUTH,
                static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_auth_t)));
}

inline bool ParseConnAuthReq(const xrt_rdma_neg_msg_t& msg, xrt_neg_cap_req* cap)
{
    if (cap == nullptr || msg.head.cmd != XRT_RDMA_NEG_CONN_AUTH || !VerifyNegCrc(msg) ||
        msg.body.conn_auth.private_len != XRT_NEG_CAP_REQ_PRIVATE_LEN ||
        msg.body.conn_auth.private_len != sizeof(*cap)) {
        return false;
    }
    std::memcpy(cap, msg.body.conn_auth.private_data, sizeof(*cap));
    return true;
}

inline bool ParseConnAuthRsp(const xrt_rdma_neg_msg_t& msg, xrt_neg_cap_rsp* cap)
{
    if (cap == nullptr || msg.head.cmd != XRT_RDMA_NEG_CONN_AUTH || !VerifyNegCrc(msg) ||
        msg.body.conn_auth.private_len != XRT_NEG_CAP_RSP_PRIVATE_LEN ||
        msg.body.conn_auth.private_len != sizeof(*cap)) {
        return false;
    }
    std::memcpy(cap, msg.body.conn_auth.private_data, sizeof(*cap));
    return true;
}

inline void FillConnUb(xrt_rdma_neg_msg_t* msg, uint8_t cmd, const kv_cm_conn_ub_info& info)
{
    std::memset(&msg->body, 0, sizeof(msg->body));
    msg->body.conn_ub.ub_info = info;
    FillNegHead(msg, cmd, static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_ub_t)));
}

inline void FillConnDone(xrt_rdma_neg_msg_t* msg)
{
    std::memset(&msg->body, 0, sizeof(msg->body));
    FillNegHead(msg, XRT_RDMA_NEG_CONN_DONE, 0);
}

inline void FillConnFin(xrt_rdma_neg_msg_t* msg, uint32_t localQpn, uint32_t remoteQpn)
{
    std::memset(&msg->body, 0, sizeof(msg->body));
    msg->body.conn_fin.local_qpn = localQpn;
    msg->body.conn_fin.remote_qpn = remoteQpn;
    FillNegHead(msg, XRT_RDMA_NEG_CONN_FIN, static_cast<uint32_t>(sizeof(xrt_rdma_neg_conn_fin_t)));
}

}  // namespace umc::comm::oob::negcodec
