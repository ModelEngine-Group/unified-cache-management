// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/IPC_message_interface.h"
#include <limits.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 性能瓶颈分析，优化版本，减少一次拷贝
NDS_ERROR_T receive_mq(MQ_TYPE mq_type, const char *queue_name, void *task)
{
    (void)mq_type;
    mqd_t mqdes;
    struct mq_attr attr;
    char buffer[MQ_BUFFER_SIZE];
    unsigned int prio;
    char chmod_command[256];

    attr.mq_flags = 0;
    attr.mq_maxmsg = 1;
    attr.mq_msgsize = MQ_BUFFER_SIZE;
    attr.mq_curmsgs = 0;
    // printf("before aicpu_mq_open_v2\n");
    mqdes = aicpu_mq_open_v2(queue_name, O_CREAT | O_RDWR, 0666, &attr);
    if (mqdes == (mqd_t)-1) {
        perror("mq_open failed");
        exit(EXIT_FAILURE);
    }
    snprintf(chmod_command, sizeof(chmod_command), "chmod 666 /dev/mqueue%s", queue_name);

    int res = aicpu_mq_receive(mqdes, (char *)task, MQ_BUFFER_SIZE, &prio);

    if (res == -1) {
        printf("%ld %s\n", sizeof(buffer), buffer);
        perror("mq_receive failed");
        aicpu_mq_close(mqdes);
        aicpu_mq_unlink(queue_name);
        exit(EXIT_FAILURE);
    }
    aicpu_mq_close(mqdes);

    return NDS_SUCCESS;
}

NDS_ERROR_T send_mq(MQ_TYPE mq_type, const char *queue_name, const void *task)
{
    mqd_t mqdes;
    struct mq_attr attr;
    int task_len = 0;

    if (task == NULL) {
        printf("Invalid task_cq_info pointer\n");
        return TASK_INFO_NULL;
    }

    attr.mq_flags = 0;
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = MQ_BUFFER_SIZE;
    attr.mq_curmsgs = 0;

    mqdes = aicpu_mq_open_v2(queue_name, O_CREAT | O_RDWR, 0666, &attr);

    if (mqdes == (mqd_t)-1) {
        perror("mq_open failed");
        printf(" the quene name is %s\n", queue_name);
        return MQ_OPEN_FAILED;
    }

    switch (mq_type) {
        case IPC_DATA_BLOCK: task_len = sizeof(IPC_data_block); break;
        case TASK_CQ_INFO: task_len = sizeof(task_cq_info); break;
        case PHY_ADDR_TO_FILE: task_len = sizeof(int); break;
        case NOFITY_ADDR: task_len = sizeof(uint64_t); break;
        default: printf("NO MQ_TYPE!\n"); return MQ_TYPE_ERROR;
    }

    // 开销来源：跨态拷贝
    if (aicpu_mq_send(mqdes, (const char *)task, task_len, 1) == -1) {
        perror("mq_send failed");
        aicpu_mq_close(mqdes);
        return MQ_SEND_FAILED;
    }

    aicpu_mq_close(mqdes);
    return NDS_SUCCESS;
}

// TODO：直调posix mqueue接口，后续可以基于此做功能扩展
mqd_t aicpu_mq_open(const char *name, int oflag) { return mq_open(name, oflag); }

mqd_t aicpu_mq_open_v2(const char *name, int oflag, mode_t mode, struct mq_attr *attr)
{
    return mq_open(name, oflag, mode, attr);
}

int aicpu_mq_close(mqd_t mqdes) { return mq_close(mqdes); }

int aicpu_mq_unlink(const char *name) { return mq_unlink(name); }

int aicpu_mq_send(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned int msg_prio)
{
    return mq_send(mqdes, msg_ptr, msg_len, msg_prio);
}

ssize_t aicpu_mq_receive(mqd_t mqdes, char *msg_ptr, size_t msg_len, unsigned int *msg_prio)
{
    return mq_receive(mqdes, msg_ptr, msg_len, msg_prio);
}

int aicpu_mq_getattr(mqd_t mqdes, struct mq_attr *attr) { return mq_getattr(mqdes, attr); }

int aicpu_mq_setattr(mqd_t mqdes, const struct mq_attr *newattr, struct mq_attr *oldattr)
{
    return mq_setattr(mqdes, newattr, oldattr);
}