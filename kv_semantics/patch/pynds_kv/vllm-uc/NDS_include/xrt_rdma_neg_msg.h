// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: XRT RDMA SOCKET建连协商消息定义
 * Author: zhangjian&xiapeiwei
 * Create: 2026/01/09
 * Notes: (1)结构遵守网络字节序；(2)该文件的任何修改都会影响升级,请谨慎修改和严格审核；
 */

#ifndef XRT_RDMA_NEG_MSG_H
#define XRT_RDMA_NEG_MSG_H

#include <infiniband/verbs.h>

#define XRT_NEG_MSG_VERSION 0      // 协商消息版本号
#define XRT_MAX_NEG_BODY_SIZE 240  // 协商消息体最大长度
#define XRT_MAX_NEG_PRI_SIZE 128   // 用户私有信息最大长度
#define XRT_MAX_NEG_QP_NUM 32      // 单次协商最大批量建立连接数
#define XRT_GID_RAW_LEN 16

#ifdef __cplusplus
extern "C" {
#endif

/*
 * [Client]               [Server]
 *     |   ---CONN_AUTH-->   |
 *     |   <--CONN_AUTH---   |
 *     |   ---CONN_REQ--->   |
 *     |   <--CONN_RSP----   |
 *     |   ---CONN_DONE-->   |
 */

#pragma pack(1)

// 协商cmd命令字
typedef enum {
    XRT_RDMA_NEG_CONN_AUTH = 0,
    XRT_RDMA_NEG_CONN_REQ = 1,
    XRT_RDMA_NEG_CONN_RSP = 2,
    XRT_RDMA_NEG_CONN_DONE = 3,
    XRT_RDMA_NEG_CONN_FIN = 4,
    XRT_RDMA_NEG_CMD_BUTT
} xrt_rdma_neg_cmd_t;

// 协商消息头（16字节）
typedef struct {
    uint32_t crc;  // CRC32
    uint8_t ver;
    uint8_t cmd;
    uint8_t pad[6];
    uint32_t len;
} xrt_rdma_neg_head_t;

// CONN_AUTH建连认证
typedef struct {
    uint32_t cap;
    uint8_t rsv[24];

    uint32_t private_len;
    uint8_t private_data[XRT_MAX_NEG_PRI_SIZE];
} xrt_rdma_neg_conn_auth_t;

// QP协商信息（32字节）
typedef struct {
    // uint8_t gid[XRT_GID_RAW_LEN];
    union ibv_gid gid;
    uint16_t lid;
    uint8_t mtu;  // ibv_mtu
    uint8_t total_qp_num;

    uint8_t sl;
    uint8_t traffic_class;
    uint8_t rnr_timer;
    uint8_t rnr_retry_cnt;
    uint8_t timeout;  // ack timeout
    uint8_t retry_cnt;
    uint8_t qp_rd_atom;
    uint8_t rsv[1];

    uint32_t start_psn;
    uint32_t qpn[XRT_MAX_NEG_QP_NUM];
} xrt_rdma_neg_qp_attr_t;

// CONN_REQ建连请求
typedef struct {
    xrt_rdma_neg_qp_attr_t qp_attr;  // 包含零长QPN数组，确保放在结构体的最后
} xrt_rdma_neg_conn_req_t;

// CONN_RSP建连响应
typedef struct {
    xrt_rdma_neg_qp_attr_t qp_attr;  // 包含零长QPN数组，确保放在结构体的最后
} xrt_rdma_neg_conn_rsp_t;

// CONN_FIN关闭连接
typedef struct {
    uint32_t local_qpn;
    uint32_t remote_qpn;
} xrt_rdma_neg_conn_fin_t;

// 协商消息（256B = 16B + 240B）
typedef struct {
    xrt_rdma_neg_head_t head;
    union {
        uint8_t raw[XRT_MAX_NEG_BODY_SIZE];
        xrt_rdma_neg_conn_auth_t conn_auth;
        xrt_rdma_neg_conn_req_t conn_req;
        xrt_rdma_neg_conn_rsp_t conn_rsp;
        xrt_rdma_neg_conn_fin_t conn_fin;
    } body;
} xrt_rdma_neg_msg_t;

#if defined(__ARM64__) && !defined(__clang__)
#pragma pack(0)
#else
#pragma pack()
#endif

#ifdef __cplusplus
}
#endif
#endif
