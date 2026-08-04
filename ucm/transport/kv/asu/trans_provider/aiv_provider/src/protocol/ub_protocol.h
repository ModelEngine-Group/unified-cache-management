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

namespace umc::comm {

enum kv_protocol_opcode_t : uint8_t {
    UB_PROTO = 0,
};

enum kv_cm_op_t : uint8_t {
    KV_CM_OP_INVALID = 0,
    KV_CM_OP_DISC = 3,
    KV_CM_OP_KEEPALIVE = 4,
    KV_CM_OP_MR_REGISTER = 5,
    KV_CM_OP_MR_REG_ACK = 6,
};

constexpr uint8_t kKvCmMajorVersionV1 = 1;
constexpr uint8_t kKvCmMinorVersionV1 = 0;
constexpr uint8_t kKvCmMinorVersionV1_1 = 1;
constexpr uint8_t kKvCmMinorVersionV1_2 = 2;
constexpr uint8_t kKvCmMinorVersionV1_3 = 3;
constexpr uint8_t kKvCmMinorVersionV1_4 = 4;
constexpr uint8_t kKvCmMinorVersionCur = kKvCmMinorVersionV1_4;

enum kv_cm_net_addr_kind : uint8_t {
    KV_NET_ADDR_NONE = 0,
    KV_NET_ADDR_UBOE4 = 1,
    KV_NET_ADDR_UBOE6 = 2,
};

#pragma pack(push, 1)
struct kv_cm_hdr {
    uint8_t major_version;  // [0,1)   == kKvCmMajorVersionV1
    uint8_t minor_version;  // [1,2)   kKvCmMinorVersionV1..kKvCmMinorVersionCur
    uint8_t op;             // [2,3)   kv_cm_op_t
    uint8_t protocol;       // [3,4)   kv_protocol_opcode_t == UB_PROTO
    uint32_t msg_len;       // [4,8)
    uint32_t request_id;    // [8,12)
    uint8_t rsv[4];         // [12,16)
};
#pragma pack(pop)
static_assert(sizeof(kv_cm_hdr) == 16, "kv_cm_hdr must be 16 bytes");
static_assert(offsetof(kv_cm_hdr, msg_len) == 4, "");
static_assert(offsetof(kv_cm_hdr, request_id) == 8, "");

#pragma pack(push, 1)
struct kv_cm_conn_ub_info {
    uint8_t hccp_eid_raw[16];  // [  0, 16)
    uint8_t qp_key_raw[64];    // [ 16, 80)
    uint8_t qp_key_size;       // [ 80, 81)
    uint8_t mem_key_raw[128];  // [ 81,209)
    uint8_t mem_key_size;      // [209,210)
    uint32_t token_id;         // [210,214)
    uint32_t token_value;      // [214,218)
    uint64_t remote_addr;      // [218,226)
    uint64_t remote_size;      // [226,234)
    uint8_t net_addr_kind;     // [234,235)
    uint8_t prefix_len;        // [235,236)
    uint8_t net_addr_rsv1[2];  // [236,238)
    uint64_t uboe_vlan;        // [238,246)
    uint8_t uboe_mac[6];       // [246,252)
    uint8_t net_addr_rsv2[4];  // [252,256)
    uint64_t mami_tp_handle;   // [256,264)
    uint64_t mami_tag;         // [264,272)
    uint32_t mami_tx_psn;      // [272,276)
    uint32_t mami_rx_psn;      // [276,280)
    uint32_t rm_uasid;         // [280,284)
    uint32_t rm_jetty_id;      // [284,288)  QpCreateInfo.ub.id（= urma_jetty_id_t.id）
    uint8_t rm_jetty_type;     // [288,289)  TARGET_TYPE_JETTY(1)
    uint8_t rm_jetty_num;      // [289,290)  must be 1
    uint8_t conn_mode;         // [290,291)
    uint8_t rm_rsv[5];         // [291,296)
};
#pragma pack(pop)
static_assert(sizeof(kv_cm_conn_ub_info) == 296, "kv_cm_conn_ub_info must be 296 bytes");
static_assert(offsetof(kv_cm_conn_ub_info, qp_key_raw) == 16, "");
static_assert(offsetof(kv_cm_conn_ub_info, mem_key_raw) == 81, "");
static_assert(offsetof(kv_cm_conn_ub_info, token_id) == 210, "");
static_assert(offsetof(kv_cm_conn_ub_info, token_value) == 214, "");
static_assert(offsetof(kv_cm_conn_ub_info, remote_addr) == 218, "");
static_assert(offsetof(kv_cm_conn_ub_info, remote_size) == 226, "");
static_assert(offsetof(kv_cm_conn_ub_info, net_addr_kind) == 234, "");
static_assert(offsetof(kv_cm_conn_ub_info, prefix_len) == 235, "");
static_assert(offsetof(kv_cm_conn_ub_info, uboe_vlan) == 238, "");
static_assert(offsetof(kv_cm_conn_ub_info, uboe_mac) == 246, "");
static_assert(offsetof(kv_cm_conn_ub_info, net_addr_rsv2) == 252, "");
static_assert(offsetof(kv_cm_conn_ub_info, mami_tp_handle) == 256, "");
static_assert(offsetof(kv_cm_conn_ub_info, mami_tag) == 264, "");
static_assert(offsetof(kv_cm_conn_ub_info, mami_tx_psn) == 272, "");
static_assert(offsetof(kv_cm_conn_ub_info, mami_rx_psn) == 276, "");
static_assert(offsetof(kv_cm_conn_ub_info, rm_uasid) == 280, "");
static_assert(offsetof(kv_cm_conn_ub_info, rm_jetty_id) == 284, "");
static_assert(offsetof(kv_cm_conn_ub_info, rm_jetty_type) == 288, "");
static_assert(offsetof(kv_cm_conn_ub_info, rm_jetty_num) == 289, "");
static_assert(offsetof(kv_cm_conn_ub_info, conn_mode) == 290, "");

#pragma pack(push, 1)
struct kv_cm_mr_seg {
    uint32_t mr_id;         // [  0,  4)
    uint32_t token_value;   // [  4,  8)
    uint64_t addr;          // [  8, 16)
    uint64_t size;          // [ 16, 24)
    uint8_t seg_blob[128];  // [ 24,152)
    uint8_t seg_blob_size;  // [152,153)
    uint8_t rsv[7];         // [153,160)
};
#pragma pack(pop)
static_assert(sizeof(kv_cm_mr_seg) == 160, "kv_cm_mr_seg must be 160 bytes");
static_assert(offsetof(kv_cm_mr_seg, token_value) == 4, "");
static_assert(offsetof(kv_cm_mr_seg, addr) == 8, "");
static_assert(offsetof(kv_cm_mr_seg, size) == 16, "");
static_assert(offsetof(kv_cm_mr_seg, seg_blob) == 24, "");
static_assert(offsetof(kv_cm_mr_seg, seg_blob_size) == 152, "");

constexpr uint32_t kKvCmMaxMrPerRegister = 8;

#pragma pack(push, 1)
struct kv_cm_mr_register {
    kv_cm_hdr hdr;                             // [   0,  16)
    uint32_t client_id;                        // [  16,  20)  == TaskDesc.client_id
    uint32_t mr_count;                         // [  20,  24)
    kv_cm_mr_seg segs[kKvCmMaxMrPerRegister];  // [  24,1304)
};
#pragma pack(pop)
static_assert(sizeof(kv_cm_mr_register) == 24 + 160 * 8, "kv_cm_mr_register must be 1304 bytes");
static_assert(offsetof(kv_cm_mr_register, client_id) == 16, "");
static_assert(offsetof(kv_cm_mr_register, mr_count) == 20, "");
static_assert(offsetof(kv_cm_mr_register, segs) == 24, "");

using kv_cm_mr_reg_ack = kv_cm_hdr;

using kv_cm_disc = kv_cm_hdr;

using kv_cm_keepalive = kv_cm_hdr;

constexpr uint32_t kProtocolSignalValueVersion = 1;
constexpr uint32_t kProtocolCmFrameMajor = 1;
constexpr uint32_t kProtocolCmFrameMinor = 1;

}  // namespace umc::comm
