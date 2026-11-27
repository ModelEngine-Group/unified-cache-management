// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_MQ_IPC__H__
#define __NDS_MQ_IPC__H__

#include <stdint.h>
#include "common.h"

typedef struct {
    uint64_t task_id;  // 免拷贝方案无需任务ID   wh:需要增加rid标志
    uint16_t mr_id;    // mr_seg id
    uint32_t nsid;
    uint64_t notify_addr;    // 免拷贝方案需notify地址
    OP_TYPE op_type;         // 操作类型
    uint16_t addr_num;       // 包内block_id_list中有效块数 batch_num
    uint16_t reserved_data;  // 用于对齐的保留字段
    uint64_t
        data_lenth;  // 块的大小 每个块大小相同，第一层写时，该长度为该block的总kvcache大小（vLLM）
    addr_pair addr_pair_list[MAX_ADDR_NUM_IPC];  // 地址对数组
    uint8_t access_control;
} IPC_data_block;

#endif  // END __NDS_MQ_IPC__H__
