/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PRIMITIVES_H
#define HCOMM_PRIMITIVES_H

#include <stdint.h>
#include <stddef.h>
#include "securec.h"
#include <arpa/inet.h>
#include "acl/acl_rt.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifndef CHANNEL_HANDLE_DEFINED
#define CHANNEL_HANDLE_DEFINED
/**
 * @brief 通道句柄类型
 */
typedef uint64_t ChannelHandle;
#endif

#ifndef THREAD_HANDLE_DEFINED
#define THREAD_HANDLE_DEFINED
/**
 * @brief 线程句柄类型
 */
typedef uint64_t ThreadHandle;
#endif

typedef enum {
    HCOMM_REDUCE_SUM = 0,    /**< sum */
    HCOMM_REDUCE_PROD = 1,   /**< prod */
    HCOMM_REDUCE_MAX = 2,    /**< max */
    HCOMM_REDUCE_MIN = 3,    /**< min */
    HCOMM_REDUCE_RESERVED = 255 /**< reserved */
} HcommReduceOp;

typedef enum {
    HCOMM_DATA_TYPE_INT8 = 0,    /**< int8 */
    HCOMM_DATA_TYPE_INT16 = 1,   /**< int16 */
    HCOMM_DATA_TYPE_INT32 = 2,   /**< int32 */
    HCOMM_DATA_TYPE_FP16 = 3,    /**< fp16 */
    HCOMM_DATA_TYPE_FP32 = 4,    /**< fp32 */
    HCOMM_DATA_TYPE_INT64 = 5,    /**< int64 */
    HCOMM_DATA_TYPE_UINT64 = 6,    /**< uint64 */
    HCOMM_DATA_TYPE_UINT8 = 7,    /**< uint8 */
    HCOMM_DATA_TYPE_UINT16 = 8,   /**< uint16 */
    HCOMM_DATA_TYPE_UINT32 = 9,   /**< uint32 */
    HCOMM_DATA_TYPE_FP64 = 10,    /**< fp64 */
    HCOMM_DATA_TYPE_BFP16 = 11,    /**< bfp16 */
    HCOMM_DATA_TYPE_INT128 = 12,   /**< int128 */
    HCOMM_DATA_TYPE_HIF8 = 14,     /**< hif8 */
    HCOMM_DATA_TYPE_FP8E4M3 = 15,  /**< fp8e4m3 */
    HCOMM_DATA_TYPE_FP8E5M2 = 16,  /**< fp8e5m2 */
    HCOMM_DATA_TYPE_FP8E8M0 = 17,  /**< fp8e8m0 */
    HCOMM_DATA_TYPE_RESERVED = 255 /**< reserved */
} HcommDataType;

/**
 * @brief 传输类型枚举
 */
typedef enum {
    HCOMM_TRANSFER_TYPE_INVALID = -1,
    HCOMM_TRANSFER_TYPE_WRITE = 0,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,
    HCOMM_TRANSFER_TYPE_READ = 4,
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6,
} HcommTransferType;

/**
 * @brief 批量传输描述符
 */
typedef struct {
    HcommTransferType transType;     ///< 传输类型
    uint8_t reserved[4];
    union {
        uint8_t raws[56];            ///< 通用数据
        struct {
            uint64_t len;
            void *dst;
            void *src;
        } write;
        struct {
            uint64_t len;
            void *dst;
            void *src;
        } read;
        struct {
            uint64_t count;
            void *dst;
            void *src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
        } reduce;
        struct {
            uint32_t notifyIdx;
        } notifyRecord;
        struct {
            uint64_t len;
            void *dst;
            void *src;
            uint32_t notifyIdx;
        } writeWithNotify;
        struct {
            uint64_t count;
            void *dst;
            void *src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
            uint32_t notifyIdx;
        } writeReduceWithNotify;
    } transferInfo;
} HcommBatchTransferDesc;

/**
 * @brief Send/Recv prototype mailbox state.
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API.
 */
typedef enum {
    HCOMM_SEND_RECV_MAILBOX_EMPTY = 0,
    HCOMM_SEND_RECV_MAILBOX_READY = 1,
} HcommSendRecvMailboxState;

/**
 * @brief Send/Recv prototype mailbox header.
 *
 * The selected channel user buffer is interpreted as:
 *   [HcommSendRecvMailboxHeader][payload bytes]
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API.
 */
typedef struct {
    uint32_t magic;
    uint32_t state;
    uint64_t seq;
    uint64_t payloadLen;
    uint8_t reserved[40];
} HcommSendRecvMailboxHeader;

static const uint32_t HCOMM_SEND_RECV_CHANNEL_CONTEXT_MAGIC = 0x48535243; // HSRC
static const uint32_t HCOMM_SEND_RECV_CHANNEL_CONTEXT_VERSION = 2;

/**
 * @brief Send/Recv prototype channel context.
 *
 * This mock-only context lets HcommSendOnThread/HcommRecvOnThread find the
 * mailbox buffers while still issuing transport operations on transportChannel.
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API.
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    ChannelHandle transportChannel;
    uint64_t localMailboxAddr;
    uint64_t localMailboxSize;
    uint64_t remoteMailboxAddr;
    uint64_t remoteMailboxSize;
    uint64_t reserved[4];
} HcommSendRecvChannelContext;

/**
 * @defgroup 数据面编程接口
 * @{
 */

/**
 * @name 本地拷贝和规约
 * @{
 */

/**
 * @brief 本地内存拷贝
 * @param[in] thread 线程句柄
 * @param[out] dst 目标地址
 * @param[in] src 源地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 * @note 源目内存地址要能执行引擎直接访问
 */
extern int32_t HcommLocalCopyOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t len);

/**
 * @brief 本地归约操作
 * @param[in] thread 线程句柄
 * @param[out] dst 目标地址
 * @param[in] src 源地址
 * @param[in] count 元素个数
 * @param[in] dataType 数据类型
 * @param[in] reduceOp 归约操作类型
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void *dst, const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
/** @} */  // 本地拷贝和规约

/**
 * @name 本地线程间同步通知
 * @{
 */

/**
 * @brief 本地记录通知
 * @param[in] thread 线程句柄
 * @param[in] dstThread 目标线程句柄
 * @param[in] dstNotifyIdx 目标通知索引
 * @return int32_t 执行结果状态码
 * @note 配合HcommThreadNotifyWaitOnThread使用
 */
extern int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);

/**
 * @brief 本地等待通知
 * @param[in] thread 线程句柄
 * @param[in] notifyIdx 通知索引
 * @param[in] timeOut 超时时间(毫秒)
 * @return int32_t 执行结果状态码
 * @note 配合HcommThreadNotifyRecordOnThread使用
 */
extern int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut);
/** @} */  // 本地线程间同步通知

/**
 * @name 本地通知
 * @{
 */

/**
 * @brief 记录通知事件（生产者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] dstNotifyId 通知id
 * @return 执行状态码 int32_t
 */
extern int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId);

/**
 * @brief 等待通知事件（消费者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] notifyId 通知id
 * @param[in] timeOut 超时时间
 * @return 执行状态码 int32_t
 */
extern int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut);
/** @} */  // 本地通知

/**
 * @name 数据读写相关
 * @{
 */

/**
 * @brief 单边写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommWriteOnThread(
    ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 归约写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] count 元素个数
 * @param[in] dataType 数据类型
 * @param[in] reduceOp 归约操作类型
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);

/**
 * @brief 带通知的单边写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @param[in] notifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 * @note 当前在A5上主要支持
 * 
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx);

/**
 * @brief 带通知的归约写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] count 元素个数
 * @param[in] dataType 数据类型
 * @param[in] reduceOp 归约操作类型
 * @param[in] remoteNotifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 * 
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void *dst,
    const void *src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx);

/**
 * @brief 单边读操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommReadOnThread(
    ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 归约读操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] count 元素个数
 * @param[in] dataType 数据类型
 * @param[in] reduceOp 归约操作类型
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t count,
    HcommDataType dataType, HcommReduceOp reduceOp);

/**
 * @brief 单边写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 单边写操作
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommWriteNbi(ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 带通知的单边写操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @param[in] notifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 * @note 当前在A5上主要支持
 */
extern int32_t HcommWriteWithNotifyNbiOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx);

/**
 * @brief 带通知的单边写操作
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @param[in] notifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 * @note 当前在A5上主要支持
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommWriteWithNotifyNbi(ChannelHandle channel, void *dst, const void *src,
    uint64_t len, uint32_t remoteNotifyIdx);

/**
 * @brief 单边读操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 单边读操作
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommReadNbi(ChannelHandle channel, void *dst, const void *src, uint64_t len);

/**
 * @brief 批量传输操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[in] transferDescs 批量传输描述符数组
 * @param[in] transferDescNum 批量传输描述符数量
 * @return int32_t 执行结果，0表示成功
 */
extern int32_t HcommBatchTransferOnThread(ThreadHandle thread, ChannelHandle channel,
    const HcommBatchTransferDesc *transferDescs, uint32_t transferDescNum);

/**
 * @brief Send data to the peer's default receive mailbox on a channel.
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[in] src 源内存地址
 * @param[in] len 数据长度（字节）
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API.
 */
extern int32_t HcommSendOnThread(ThreadHandle thread, ChannelHandle channel, const void *src, uint64_t len);

/**
 * @brief Receive data from the local default receive mailbox on a channel.
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[out] dst 目标内存地址
 * @param[in] dstCapacity 目标内存容量（字节）
 * @param[out] receivedLen 实际接收的数据长度（字节）
 * @param[in] timeoutMs 超时时间（毫秒）
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API.
 */
extern int32_t HcommRecvOnThread(ThreadHandle thread, ChannelHandle channel, void *dst, uint64_t dstCapacity,
    uint64_t *receivedLen, uint32_t timeoutMs);


/** @} */  // 数据读写相关

/**
 * @name 通知
 * @{
 */

/**
 * @brief 记录通知事件
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[in] remoteNotifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);

/**
 * @brief 记录通知事件
 * @param[in] channel 通道句柄
 * @param[in] remoteNotifyIdx 远端通知索引
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx);

/**
 * @brief 等待通知事件
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @param[in] localNotifyIdx 本地通知索引
 * @param[in] timeOut 超时时间(毫秒)
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut);

/**
 * @brief 等待通知事件
 * @param[in] channel 通道句柄
 * @param[in] localNotifyIdx 本地通知索引
 * @param[in] timeOut 超时时间(毫秒)
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut);

/** @} */  // 通知

/**
 * @defgroup 批量下发设置接口
 * @{
 */

/**
 * @brief 批量模式执行开始
 * @param batchTag 批量Id
 * @return int32_t 执行结果状态码
 * @note Start和End及中间的批量任务需要在同一个线程上执行
 */
extern int32_t HcommBatchModeStart(const char *batchTag);

/**
 * @brief 批量模式执行结束
 * @param batchTag 批量Id
 * @return int32_t 执行结果状态码
 * @note Start和End及中间的批量任务需要在同一个线程上执行
 */
extern int32_t HcommBatchModeEnd(const char *batchTag);

/** @} */  // 批量下发设置接口

/** @} */  // 数据面编程接口
/** @} */  // 算子编程接口

/**
 * @brief 获取通信域并加锁
 * @param[in] commId 通信域id
 * @return int32_t 执行结果状态码
 * @note 当前仅支持AICPU模式
 */
extern int32_t HcommAcquireComm(const char* commId);

/**
 * @brief 释放通信域
 * @param[in] commId 通信域id
 * @return int32_t 执行结果状态码
 * @note 当前仅支持AICPU模式
 */
extern int32_t HcommReleaseComm(const char* commId);

/**
 * @brief DPU 数据面 Fence 接口
 * @param[in] thread 线程句柄
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommFenceOnThread(ThreadHandle thread);

/**
 * @brief 通信通道级内存屏障操作
 * @param[in] thread 线程句柄
 * @param[in] channel 通道句柄
 * @return int32_t 执行结果状态码
 */
extern int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel);

/**
 * @brief 通信通道级内存屏障操作
 * @param[in] channel 通道句柄
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommChannelFence(ChannelHandle channel);

/** @} */  // 算子编程接口
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif
