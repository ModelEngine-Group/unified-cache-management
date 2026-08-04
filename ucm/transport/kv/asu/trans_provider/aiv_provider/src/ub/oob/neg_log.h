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

#include "src/protocol/ub_protocol.h"
#include "src/protocol/xrt_rdma_neg_msg.h"
#include "src/ub/log.h"

namespace umc::comm::oob {

inline void LogCapReq(const char* tag, const xrt_rdma_neg_msg_t& msg, const xrt_neg_cap_req& cap)
{
    UB_LOG_DEBUG(
        "NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}} "
        "cap_req{{major={} minor={} kato={} private_len={}}}",
        tag, msg.head.crc, msg.head.ver, msg.head.cmd, msg.head.len, cap.major_version,
        cap.minor_version, cap.kato, msg.body.conn_auth.private_len);
}

inline void LogCapRsp(const char* tag, const xrt_rdma_neg_msg_t& msg, const xrt_neg_cap_rsp& cap)
{
    UB_LOG_DEBUG(
        "NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}} "
        "cap_rsp{{major={} minor={} controller_id={} queue_num={} ioq_depth={} "
        "sr_mdts={}KB key_length={} controller_cap=0x{:x} private_len={}}}",
        tag, msg.head.crc, msg.head.ver, msg.head.cmd, msg.head.len, cap.major_version,
        cap.minor_version, cap.controller_id, cap.queue_num, cap.ioq_depth, cap.sr_mdts,
        cap.key_length, cap.controller_cap, msg.body.conn_auth.private_len);
}

inline void LogUbInfo(const char* tag, const xrt_rdma_neg_msg_t& msg,
                      const kv_cm_conn_ub_info& info)
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

inline void LogHeadOnly(const char* tag, const xrt_rdma_neg_msg_t& msg)
{
    UB_LOG_DEBUG("NEG_PROTO {} head{{crc=0x{:08x} ver={} cmd={} len={}}}", tag, msg.head.crc,
                 msg.head.ver, msg.head.cmd, msg.head.len);
}

}  // namespace umc::comm::oob
