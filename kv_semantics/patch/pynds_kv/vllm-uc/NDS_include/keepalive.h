// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "NDS_client.h"

#pragma pack(push, 1)
typedef struct {
    uint8_t opcode;
    uint8_t rsvd1;
    uint16_t cid;
    uint32_t rsvd2[2];
    uint64_t res_buf_addr;
    uint64_t res_buf_mrkey;
    uint32_t rsvd3[10];
} data_heart_beat;
#pragma pack(pop)

extern void *keepalive_thread(void *arg);
extern NDS_ERROR_T send_heart_beat(struct rdma_connection *conn);
extern NDS_ERROR_T transfer_data_heartbeat(struct rdma_connection *conn,
                                           data_heart_beat *keepalive_msg, int qpID);