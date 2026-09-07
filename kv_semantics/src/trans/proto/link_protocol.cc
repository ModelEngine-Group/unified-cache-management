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
#include "link_protocol.h"
#include <cstring>

namespace kv {

Status NegotiateSqe::Pack(const NegotiateRequest& req)
{
    buffer.assign(kMsgHeaderSize + kNegotiatePayloadSize, 0);

    MsgHeader header;
    header.cmd = LinkProtocolCmd::Negotiate;
    header.len = kNegotiatePayloadSize;
    header.Pack(buffer);

    // Offset 16: cap[31:0]
    std::memcpy(&buffer[16], &req.cap, 4);

    // Offset 20: rsv[23:0] (zero)

    // Offset 44: private_len[31:0]
    std::memcpy(&buffer[44], &req.private_len, 4);

    // Offset 48: major_version
    buffer[48] = req.major_version;

    // Offset 49: minor_version
    buffer[49] = req.minor_version;

    // Offset 50: kato[15:0]
    std::memcpy(&buffer[50], &req.kato, 2);

    // Offset 52: reserved[123:0] (zero)
    return Status::OK();
}

Status NegotiateSqe::Validate() const
{
    if (buffer.size() < kMsgHeaderSize + kNegotiatePayloadSize) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer too small");
    }
    return Status::OK();
}

Status HandshakeSqe::Pack(const HandshakeRequest& req)
{
    buffer.assign(kMsgHeaderSize + kHandshakePayloadSize, 0);

    MsgHeader header;
    header.cmd = LinkProtocolCmd::Handshake;
    header.len = kHandshakePayloadSize;
    header.Pack(buffer);

    // Offset 16: gid[15:0]
    std::memcpy(&buffer[16], req.gid, 16);

    // Offset 32: lid[15:0]
    std::memcpy(&buffer[32], &req.lid, 2);

    // Offset 34: mtu
    buffer[34] = req.mtu;

    // Offset 35: total_qp_num
    buffer[35] = req.total_qp_num;

    // Offset 36: sl
    buffer[36] = req.sl;

    // Offset 37: traffic_class
    buffer[37] = req.traffic_class;

    // Offset 38: rnr_timer
    buffer[38] = req.rnr_timer;

    // Offset 39: rnr_retry_cnt
    buffer[39] = req.rnr_retry_cnt;

    // Offset 40: timeout
    buffer[40] = req.timeout;

    // Offset 41: retry_cnt
    buffer[41] = req.retry_cnt;

    // Offset 42: qp_rd_atom
    buffer[42] = req.qp_rd_atom;

    // Offset 43: rsv
    buffer[43] = req.rsv;

    // Offset 44: start_psn[31:0]
    std::memcpy(&buffer[44], &req.start_psn, 4);

    // Offset 48: qpn[32] (128 bytes)
    std::memcpy(&buffer[48], req.qpn, 128);
    return Status::OK();
}

Status HandshakeSqe::Validate() const
{
    if (buffer.size() < kMsgHeaderSize + kHandshakePayloadSize) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "buffer too small");
    }
    return Status::OK();
}

Status HandshakeDoneSqe::Pack()
{
    buffer.assign(kMsgHeaderSize, 0);

    MsgHeader header;
    header.cmd = LinkProtocolCmd::HandshakeDone;
    header.len = 0;
    header.Pack(buffer);
    return Status::OK();
}

Status DisconnectSqe::Pack(const DisconnectRequest& req)
{
    buffer.assign(kMsgHeaderSize + kDisconnectPayloadSize, 0);

    MsgHeader header;
    header.cmd = LinkProtocolCmd::Disconnect;
    header.len = kDisconnectPayloadSize;
    header.Pack(buffer);

    // Offset 16: local_qpn[31:0]
    std::memcpy(&buffer[16], &req.local_qpn, 4);

    // Offset 20: remote_qpn[31:0]
    std::memcpy(&buffer[20], &req.remote_qpn, 4);
    return Status::OK();
}

}  // namespace kv
