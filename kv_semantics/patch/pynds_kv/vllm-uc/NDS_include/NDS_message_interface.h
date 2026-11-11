// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * 本文件为存储计算对接的数据接口数据类型、共享队列及对应post/pop的实现
 */

#ifndef __NPU_STORAGE_STRUCTURES_H__
#define __NPU_STORAGE_STRUCTURES_H__

#include <stdint.h>
#include <stdlib.h>
#include "common.h"

#pragma pack(push, 1)
typedef struct {
    uint32_t offset;  // 存储侧offset
    block_key block_id;
    uint64_t hbm_addr;    // HBM地址
    uint32_t length;      // 实际只用了前24bit，高8bit为mr_key[0]
    uint32_t remote_key;  // 前24bit:mr_key[3:1], 高8bit为dptrtype
} addr_pair_storge;
#pragma pack(pop)

/* batch 操作包，包含batch get/put */
#pragma pack(push, 1)
typedef struct {
    // 对应到Dword 0-1
    uint8_t opcode;
    uint8_t rsvd1;  // rsvd1
    uint16_t cid;   // 不能重复
    uint32_t nsid;  // 不需要管

    // 对应到Dword 2-5
    uint8_t DSPEC_DTYPE;
    uint8_t rsvd6[3];
    uint64_t task_id;   // 请求id
    uint16_t layer_id;  // layer_id 不填
    uint16_t addr_num;  // 包内block_id 到HBM有效对数--> batch num
    // uint32_t rsvd2;

    // 对应到Dword 6-9
    uint64_t dptr_buf;  // 协议固定填0
    uint32_t dptr_len;  // batch_num * 32
    uint8_t rsvd3[3];   // 协议固定填0
    uint8_t dptr_type;  // 固定1

    // 对应到Dword 10-15
    uint16_t batch_num;  // 子IO个数
    uint16_t rsvd4;
    uint32_t rsvd5[5];

    addr_pair_storge addr_pair_list[MAX_ADDR_NUM];
} data_block;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    // 对应到Dword 0-1
    uint8_t opcode;
    uint8_t rsvd1;  // rsvd1
    uint16_t cid;   // 不能重复
    uint32_t nsid;  // 不需要管

    // 对应到Dword 2-5
    uint8_t DSPEC_DTYPE;
    uint8_t rsvd5[3];
    uint64_t task_id;   // 请求id
    uint16_t layer_id;  // layer_id 不填
    uint16_t addr_num;  // 包内block_id 到HBM有效对数--> batch num

    // 对应到Dword 2-5
    uint64_t dptr_buf;       // 协议固定填0
    uint32_t dptr_len_key;   // 0-23 bit length | 24-31bit mr_key[0]
    uint32_t dptr_key_type;  // key[3:1], type 0x40
    uint32_t offset;         //
    uint32_t length;         // 0-23 bit length | 24-30bit 预留 | 31 bit LR

    block_key block_id;
    // uint32_t rsvd;
} data_block_single;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    OP_CODE opcode;
    uint8_t rsvd1;
    uint16_t cid;   // 不能重复
    uint32_t nsid;  // 不需要管

    // 对应到Dword 2-5
    uint64_t task_id;   // 请求id
    uint16_t layer_id;  // layer_id
    uint16_t addr_num;  // 包内block_id 到HBM有效对数--> batch num
    uint32_t rsvd2;
    // 对应到Dword 2-5

    uint64_t dptr_buf;   // 协议固定填0
    uint32_t dptr_len;   // batch_num * 12
    uint8_t rsvd3[3];    // 协议固定填0
    uint8_t dptr_type;   // 固定1
    uint16_t batch_num;  // 子IO个数
    uint8_t access_control;
    uint8_t rsvd4;

    uint32_t rsvd5[5];

    block_key key[MAX_CTL_ADDR_NUM];
} ctl_block;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint16_t result;
    uint16_t rsvd1;
    uint64_t rsvd2;
    uint16_t cid;
    uint16_t rsvd : 1;
    uint16_t status : 15;
    uint32_t res_buf[16];  // 异常情况检查，每4bit为具体错误码
} recv_data_block;
#pragma pack(pop)

#endif  // __NPU_STORAGE_STRUCTURES_H__