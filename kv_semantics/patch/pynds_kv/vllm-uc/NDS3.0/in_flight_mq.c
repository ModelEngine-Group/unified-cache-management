// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/in_flight_mq.h"

task_cq_info *in_flight_message_list = NULL;
uint16_t in_flight_message_lenth = 0;

NDS_ERROR_T init_in_flight_message_list()
{
    in_flight_message_list = (task_cq_info *)malloc(MAX_INFLIGHT_TASK_NUM * sizeof(task_cq_info));
    if (in_flight_message_list == NULL) { return IN_FLIGHT_MESSAGE_LIST_NULL; }

    for (uint16_t i = 0; i < MAX_INFLIGHT_TASK_NUM; i++) {
        in_flight_message_list[i].notify_addr = UINT64_MAX;
    }

    in_flight_message_lenth = 0;

    return NDS_SUCCESS;
}

NDS_ERROR_T add_to_in_flight_message_list(IPC_data_block *ipcDataBlock)
{
    // printf("add_to_in_flight_message_list start\n");
    if (ipcDataBlock->addr_num > MAX_ADDR_NUM_IPC) { return IN_FLIGHT_LIST_LEN_MAX; }

    if (in_flight_message_lenth >= MAX_INFLIGHT_TASK_NUM) { return IN_FLIGHT_LIST_TASK_MAX; }

    task_cq_info new_cq_info;
    new_cq_info.notify_addr = ipcDataBlock->notify_addr;
    new_cq_info.cq_type = false;

    in_flight_message_list[in_flight_message_lenth] = new_cq_info;

    in_flight_message_lenth++;
    // printf("add_to_in_flight_message_list end\n");
    return NDS_SUCCESS;
}

NDS_ERROR_T remove_from_in_flight_message_list(uint64_t notify_addr, bool cq_status)
{
    // printf("in remove_from_in_flight_message, in_flight_message_lenth
    // %lu\n",in_flight_message_lenth);

    for (uint16_t i = 0; i < in_flight_message_lenth; i++) {
        // printf("in_flight list, notify addr(%llu)\n", in_flight_message_list[i].notify_addr);
        if (in_flight_message_list[i].notify_addr == notify_addr) {
            // printf("in_flight_message_list found\n");
            // task_cq_info cq_info = in_flight_message_list[i];
            // cq_info.cq_type = cq_status;
            in_flight_message_list[i].cq_type = cq_status;
            *(volatile uint64_t *)(notify_addr) = 1;

            // printf("notify address end\n");

            in_flight_message_list[i].notify_addr = UINT64_MAX;

            if (i != in_flight_message_lenth - 1) {
                for (uint16_t j = i; j < in_flight_message_lenth - 1; j++) {
                    in_flight_message_list[j] = in_flight_message_list[j + 1];
                }
            }

            in_flight_message_lenth--;
            // printf("remove succ\n");
            return NDS_SUCCESS;
        }
        printf("remove error\n");
    }
    return IN_FLIGHT_LIST_REMOVE_ERROR;
}

NDS_ERROR_T clear_in_flight_message_list()
{
    for (uint16_t i = 0; i < in_flight_message_lenth; i++) {
        if (in_flight_message_list[i].notify_addr != UINT64_MAX) {
            task_cq_info cq_info = in_flight_message_list[i];
            cq_info.cq_type = false;

            char que_name[48];  // uint16_t 的最大值是65535，需要6个字符（包括'\0'）
            snprintf(que_name, sizeof(que_name), "%s%lu", CQ_QUEN_HEAD,
                     in_flight_message_list[i].notify_addr);  // "%hu"表示unsigned short

            //  if (send_mq (TASK_CQ_INFO, que_name, &cq_info) != NDS_SUCCESS) {
            //     return -1;
            // }

            memset((uint32_t *)(cq_info.notify_addr), 0, sizeof(uint32_t));

            in_flight_message_list[i].notify_addr = UINT64_MAX;
        }
    }

    return NDS_SUCCESS;
}

NDS_ERROR_T destory_in_flight_list()
{
    for (uint16_t i = 0; i < in_flight_message_lenth; i++) {
        if (in_flight_message_list[i].notify_addr != UINT64_MAX) {
            char que_name[48];
            snprintf(que_name, sizeof(que_name), "%s%lu", CQ_QUEN_HEAD,
                     in_flight_message_list[i].notify_addr);
            aicpu_mq_unlink(que_name);
        }
    }

    if (in_flight_message_list != NULL) {
        free(in_flight_message_list);
        in_flight_message_list = NULL;
        // printf("destroy in flight list success!\n");
    }

    return NDS_SUCCESS;
}