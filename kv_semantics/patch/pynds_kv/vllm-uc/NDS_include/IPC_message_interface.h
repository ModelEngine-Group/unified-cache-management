// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_MQ_IPC__H__
#define __NDS_MQ_IPC__H__

#include <mqueue.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "common.h"

#define MQ_BUFFER_SIZE 64 * 1024
#define QUEUE_NAME "/nds3.0_posix_queue"

typedef enum {
    IPC_DATA_BLOCK = 0,  // 算子下发通信任务的task
    TASK_CQ_INFO,        // 通信上传通信任务的task
    PHY_ADDR_TO_FILE,    // 写物理地址到文件
    NOFITY_ADDR,         // 上传notify_addr
} MQ_TYPE;

typedef struct {
    uint64_t notify_addr;
    bool cq_type;
} task_cq_info;

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

/********************************** 对外的接口 ***********************************/
/**
 * @brief 创建消息队列并发起recv请求
 *
 * @param[in] mq_type 消息队列的类型
 * @param[in] queue_name 消息队列的名字
 * @param[out] task 收到的任务
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
NDS_ERROR_T receive_mq(MQ_TYPE mq_type, const char *queue_name, void *task);

/**
 * @brief 发起send请求
 *
 * @param[in] mq_type 消息队列的类型
 * @param[in] queue_name 消息队列的名字
 * @param[in] task 发送的任务
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
NDS_ERROR_T send_mq(MQ_TYPE mq_type, const char *queue_name, const void *task);

/********************************** 对内的接口 ***********************************/
/**
 * @brief 创建一个新消息队列或打开一个既有队列
 *
 * @param[in] name 消息队列名字
 * @param[in] oflag O_RDONLY | O_WRONLY | O_RDWR | O_CREAT
 *
 * @return 成功完成后，函数返回消息队列描述符；否则，函数返回(mqd_t)-1，并设置errno
 */
mqd_t aicpu_mq_open(const char *name, int oflag);

/**
 * @brief 创建一个新消息队列或打开一个既有队列
 *
 * @param[in] name 消息队列名字
 * @param[in] oflag O_RDONLY | O_WRONLY | O_RDWR | O_CREAT
 * @param[in] mode 位掩码，指定了施加于新消息队列之上的权限
 * @param[in] attr 制定了新消息队列的特性。如果attr为NULL，那么将使用系统定义的默认特性创建队列
 *                   struct mq_attr {
 *                   　　long mq_flags;   -- 消息队列flag，默认为0
 *                   　　long mq_maxmsg;  --
 * 定义了使用mq_send()向消息队列上添加消息数量上限，必须大于0。mq_open()时确定，后面无法修改
 *                   　　long mq_msgsize; --
 * 定义了加入详细队列的每条消息的大小上限，必须大于0。mq_open()时确定，后面无法修改 　　long
 * mq_curmsgs; -- 当前消息队列中消息数目
 *                   };
 *
 * @return 成功完成后，函数返回消息队列描述符；否则，函数返回(mqd_t)-1，并设置errno
 */
mqd_t aicpu_mq_open_v2(const char *name, int oflag, mode_t mode, struct mq_attr *attr);

/**
 * @brief 关闭消息队列
 *
 * @param[in] mqdes 已打开的消息队列描述符
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
int aicpu_mq_close(mqd_t mqdes);

/**
 * @brief 删除消息队列
 *
 * @param[in] name 消息队列名字
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
int aicpu_mq_unlink(const char *name);

/**
 * @brief 发送消息到消息队列
 *
 * @param[in] mqdes 已打开的消息队列描述符
 * @param[in] msg_ptr 要发送的消息的首地址
 * @param[in] msg_len msg_ptr指向消息的长度，必须小于等于mq_msgsize
 * @param[in] msg_prio
 * 消息的优先级，0表示最低优先级，最大优先级为MQ_PRIO_MAX，一条消息被添加到队列中时，它会被放置在队列中具有相同优先级的所有消息之后
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
int aicpu_mq_send(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned int msg_prio);

/**
 * @brief 从消息队列接收消息
 *
 * @param[in] mqdes 已打开的消息队列描述符
 * @param[out] msg_ptr 要接收的消息的首地址
 * @param[in] msg_len 要接收的消息的长度，必须小于等于mq_msgsize
 * @param[out] msg_prio 不为NULL，那么接收到消息的优先级就会被复制到msg_prio
 *
 * @return
 * 成功完成后，返回选定消息的长度（以字节为单位），并且消息应从队列中删除。否则，不会从队列中删除消息，函数应返回-1的值，并设置errno
 */
ssize_t aicpu_mq_receive(mqd_t mqdes, char *msg_ptr, size_t msg_len, unsigned int *msg_prio);

/**
 * @brief 获取消息队列属性
 *
 * @param[in] mqdes 已打开的消息队列描述符
 * @param[out] attr 指向mq_attr结构的指针，用于返回消息队列的当前属性
 *                   struct mq_attr {
 *                   　　long mq_flags;   -- 消息队列flag，默认为0
 *                   　　long mq_maxmsg;  --
 * 定义了使用mq_send()向消息队列上添加消息数量上限，必须大于0。mq_open()时确定，后面无法修改
 *                   　　long mq_msgsize; --
 * 定义了加入详细队列的每条消息的大小上限，必须大于0。mq_open()时确定，后面无法修改 　　long
 * mq_curmsgs; -- 当前消息队列中消息数目
 *                   };
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
int aicpu_mq_getattr(mqd_t mqdes, struct mq_attr *attr);

/**
 * @brief 设置消息队列属性
 *
 * @param[in] newattr 已打开的消息队列描述符
 * @param[in] newattr 新的消息队列描述符
 * @param[out] oldattr 如果oldattr不为NULL，那么就返回一个之前消息特性的mq_attr结构
 *
 * @return 成功完成后，函数返回0；否则，函数返回(mqd_t)-1，并设置errno
 */
int aicpu_mq_setattr(mqd_t mqdes, const struct mq_attr *newattr, struct mq_attr *oldattr);

#endif  // END __NDS_MQ_IPC__H__
