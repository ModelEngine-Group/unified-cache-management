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
#include "src/protocol/ub_protocol.h"

namespace umc::comm {

#define XRT_NEG_MSG_VERSION 1
#define XRT_MAX_NEG_PRI_SIZE 128
#define XRT_NEG_CAP_REQ_PRIVATE_LEN 4
#define XRT_NEG_CAP_RSP_PRIVATE_LEN 36

enum xrt_rdma_neg_cmd_t : uint8_t {
    XRT_RDMA_NEG_CONN_AUTH = 0,
    XRT_RDMA_NEG_CONN_REQ = 1,
    XRT_RDMA_NEG_CONN_RSP = 2,
    XRT_RDMA_NEG_CONN_DONE = 3,
    XRT_RDMA_NEG_CONN_FIN = 4,
    XRT_RDMA_NEG_CONN_HCCP = 128,
};

#pragma pack(push, 1)

struct xrt_rdma_neg_head_t {
    uint32_t crc;    // [ 0, 4)
    uint8_t ver;     // [ 4, 5)  = XRT_NEG_MSG_VERSION (=1)
    uint8_t cmd;     // [ 5, 6)  xrt_rdma_neg_cmd_t
    uint8_t pad[6];  // [ 6,12)  = 0
    uint32_t len;    // [12,16)  body 长度
};
#pragma pack(pop)
static_assert(sizeof(xrt_rdma_neg_head_t) == 16, "xrt_rdma_neg_head_t must be 16 bytes");
static_assert(offsetof(xrt_rdma_neg_head_t, ver) == 4, "");
static_assert(offsetof(xrt_rdma_neg_head_t, cmd) == 5, "");
static_assert(offsetof(xrt_rdma_neg_head_t, len) == 12, "");

#pragma pack(push, 1)
struct xrt_neg_cap_req {
    uint8_t major_version;  // [ 0, 1)  KV 协议大版本 = 1
    uint8_t minor_version;  // [ 1, 2)  KV 协议小版本 = 0
    uint16_t kato;          // [ 2, 4)  心跳/连接老化时间(s)，0=关闭
};
#pragma pack(pop)
static_assert(sizeof(xrt_neg_cap_req) == 4, "xrt_neg_cap_req must be 4 bytes");
static_assert(offsetof(xrt_neg_cap_req, kato) == 2, "");

#pragma pack(push, 1)
struct xrt_neg_cap_rsp {
    uint8_t major_version;        // [ 0, 1)  = 1
    uint8_t minor_version;        // [ 1, 2)  = 0
    uint16_t queue_num;           // [ 2, 4)  单 KV 连接支持的队列数量
    uint16_t adminq_depth;        // [ 4, 6)  adminQ 深度（0=未实现，只有 ioQ）
    uint16_t ioq_depth;           // [ 6, 8)  io 队列深度（命令并发）
    uint16_t ioq_key_batch;       // [ 8,10)  单队列 key 并发数(store/retrieve)
    uint16_t total_key_batch;     // [10,12)  单连接 key 并发数(store/retrieve)
    uint32_t sr_mdts;             // [12,16)  store/retrieve 单命令数据大小(KB)
    uint32_t bsr_mdts;            // [16,20)  batch s/r 单 key 数据大小(KB)
    uint16_t store_key_batch;     // [20,22)  单 batch store key 并发数
    uint16_t retrieve_key_batch;  // [22,24)  单 batch retrieve key 并发数
    uint16_t delete_key_batch;    // [24,26)  单 delete key 并发数
    uint16_t exist_key_batch;     // [26,28)  单 exist key 并发数
    uint8_t key_length;           // [28,29)  key 长度(B)，最大 16
    uint8_t pad;                  // [29,30)  字节对齐 = 0
    uint16_t controller_id;       // [30,32)  UB 场景 controller id（IO 下发携带）
    uint32_t controller_cap;      // [32,36)  BIT0：delete 非立即数携带 key；其余预留
};
#pragma pack(pop)
static_assert(sizeof(xrt_neg_cap_rsp) == 36, "xrt_neg_cap_rsp must be 36 bytes");
static_assert(offsetof(xrt_neg_cap_rsp, queue_num) == 2, "");
static_assert(offsetof(xrt_neg_cap_rsp, sr_mdts) == 12, "");
static_assert(offsetof(xrt_neg_cap_rsp, bsr_mdts) == 16, "");
static_assert(offsetof(xrt_neg_cap_rsp, key_length) == 28, "");
static_assert(offsetof(xrt_neg_cap_rsp, controller_id) == 30, "");
static_assert(offsetof(xrt_neg_cap_rsp, controller_cap) == 32, "");

#pragma pack(push, 1)
struct xrt_rdma_neg_conn_auth_t {
    uint32_t cap;                                // [  0,  4)  传输层能力协商（预留 = 0）
    uint8_t rsv[24];                             // [  4, 28)
    uint32_t private_len;                        // [ 28, 32)  有效 cap 长度(req=4/rsp=36)
    uint8_t private_data[XRT_MAX_NEG_PRI_SIZE];  // [ 32,160)  承载 xrt_neg_cap_req/rsp
};
#pragma pack(pop)
static_assert(sizeof(xrt_rdma_neg_conn_auth_t) == 160,
              "xrt_rdma_neg_conn_auth_t must be 160 bytes");
static_assert(offsetof(xrt_rdma_neg_conn_auth_t, private_len) == 28, "");
static_assert(offsetof(xrt_rdma_neg_conn_auth_t, private_data) == 32, "");

#pragma pack(push, 1)
struct xrt_rdma_neg_conn_ub_t {
    kv_cm_conn_ub_info ub_info;  // 本端/对端 UB jetty 4 元组（EID/QpKey/MemKey/token/UBoE/RM）
};
#pragma pack(pop)
static_assert(sizeof(xrt_rdma_neg_conn_ub_t) == 296, "xrt_rdma_neg_conn_ub_t must be 296 bytes");

#pragma pack(push, 1)
struct xrt_rdma_neg_conn_fin_t {
    uint32_t local_qpn;   // [0,4)  发起端本地 qpn（UB 下可填 jetty_id）
    uint32_t remote_qpn;  // [4,8)  对端 qpn（UB 下可填对端 jetty_id）
};
#pragma pack(pop)
static_assert(sizeof(xrt_rdma_neg_conn_fin_t) == 8, "xrt_rdma_neg_conn_fin_t must be 8 bytes");

#define XRT_MAX_NEG_BODY_SIZE 304

#pragma pack(push, 1)
struct xrt_rdma_neg_msg_t {
    xrt_rdma_neg_head_t head;
    union {
        uint8_t raw[XRT_MAX_NEG_BODY_SIZE];
        xrt_rdma_neg_conn_auth_t conn_auth;  // CONN_AUTH（双向，private_data 承载 cap_req/cap_rsp）
        xrt_rdma_neg_conn_ub_t conn_ub;      // CONN_REQ / CONN_RSP
        xrt_rdma_neg_conn_fin_t conn_fin;    // CONN_FIN
    } body;
};
#pragma pack(pop)
static_assert(sizeof(xrt_rdma_neg_msg_t) == 16 + XRT_MAX_NEG_BODY_SIZE,
              "xrt_rdma_neg_msg_t must be 320 bytes");
static_assert(offsetof(xrt_rdma_neg_msg_t, body) == 16, "");

constexpr uint32_t kXrtHccpTagSize = 192;
constexpr uint32_t kXrtHccpHeadLen = 240;

}  // namespace umc::comm
