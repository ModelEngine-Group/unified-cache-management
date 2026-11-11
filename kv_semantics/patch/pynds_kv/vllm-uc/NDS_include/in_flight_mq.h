// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_IN_FLIGHT__H__
#define __NDS_IN_FLIGHT__H__

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "IPC_message_interface.h"

#define MAX_INFLIGHT_TASK_NUM 20

// 与内核模块匹配的IOCTL定义
#define PHYMAP_IOCTL_MAGIC 'P'
#define PHYMAP_SET_PAGE_ARRAY _IOW(PHYMAP_IOCTL_MAGIC, 4, struct phymap_page_array_info)

struct phymap_page_array_info {
    size_t page_count;
    uint64_t *pages;
};

NDS_ERROR_T add_to_in_flight_message_list(IPC_data_block *ipcDataBlock);
NDS_ERROR_T remove_from_in_flight_message_list(
    uint64_t notify_addr,
    bool
        cq_status);  // 根据一个任务的完成情况，执行对应的taskt消息回复，并从list里面移除，任务的cqe触发
NDS_ERROR_T clear_in_flight_message_list();  // 遍历所有的list item将里面的每个task任务回复失败信息
                                             // ，心跳检测失败会触发
NDS_ERROR_T init_in_flight_message_list();
NDS_ERROR_T destory_in_flight_list();

#endif