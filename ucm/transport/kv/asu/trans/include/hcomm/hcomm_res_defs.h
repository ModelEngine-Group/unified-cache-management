/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_DEFS_H
#define HCOMM_RES_DEFS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static const uint32_t COMM_ADDR_EID_LEN = 16U;
static const uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fU;
static const uint32_t HCOMM_CHANNEL_VERSION_ONE = 1U;
static const uint32_t HCOMM_CHANNEL_VERSION = HCOMM_CHANNEL_VERSION_ONE;

typedef int32_t HcommResult;

/* 网络设备句柄 */
typedef void *EndpointHandle;

/**
 * @brief 内存句柄类型（不透明结构）
 */
typedef void *HcommMemHandle;

typedef void *HcommSocket;

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

/**
 * @brief 通信引擎类型枚举
 */
typedef enum {
    COMM_ENGINE_RESERVED = -1,    ///< 保留的通信引擎
    COMM_ENGINE_CPU = 0,          ///< HOST CPU引擎
    COMM_ENGINE_CPU_TS = 1,       ///< HOST CPU TS引擎
    COMM_ENGINE_AICPU = 2,        ///< AICPU引擎
    COMM_ENGINE_AICPU_TS = 3,     ///< AICPU TS引擎
    COMM_ENGINE_AIV = 4,          ///< AIV引擎
    COMM_ENGINE_CCU = 5,          ///< CCU引擎
} CommEngine;

/**
 * @brief 通信协议类型枚举
 */
typedef enum {
    COMM_PROTOCOL_RESERVED = -1,  ///< 保留协议类型
    COMM_PROTOCOL_HCCS = 0,       ///< HCCS协议
    COMM_PROTOCOL_ROCE = 1,       ///< RDMA over Converged Ethernet
    COMM_PROTOCOL_PCIE = 2,       ///< PCIE协议
    COMM_PROTOCOL_SIO = 3,        ///< SIO协议
    COMM_PROTOCOL_UBC_CTP = 4,    ///< 华为统一总线UBC_CTP
    COMM_PROTOCOL_UBC_TP = 5,     ///< 华为统一总线UBC_TP
    COMM_PROTOCOL_UB_MEM = 6,     ///< UB_MEM
    COMM_PROTOCOL_UBOE = 7,       ///< UBoE
    COMM_PROTOCOL_HCCS_ONLY = 8,  ///< 一卡双DIE使用HCCS
} CommProtocol;

/**
 * @brief 通信设备地址类别
 */
typedef enum {
    COMM_ADDR_TYPE_RESERVED = -1, ///< 保留地址类型
    COMM_ADDR_TYPE_IP_V4 = 0,     ///< IPv4地址类型
    COMM_ADDR_TYPE_IP_V6 = 1,     ///< IPv6地址类型
    COMM_ADDR_TYPE_ID = 2,        ///< ID地址类型
    COMM_ADDR_TYPE_EID = 3,       ///< EID地址类型
} CommAddrType;

/**
 * @brief 通信设备地址描述结构体
 * @note 支持CommAddrType的扩展，地址最大长度36字节
 */
typedef struct {
    CommAddrType type;         ///< 通信地址类别
    union {
        uint8_t raws[36];      ///< 通用数据
        struct in_addr addr;   ///< IPv4地址结构
        struct in6_addr addr6; ///< IPv6地址结构
        uint32_t id;           ///< 标识
        uint8_t eid[COMM_ADDR_EID_LEN];  ///< EID地址类型
    };
} CommAddr;

/**
 * @brief 通信设备Endpoint位置类型枚举
 */
typedef enum {
    ENDPOINT_LOC_TYPE_RESERVED = -1, ///< 保留的Endpoint位置
    ENDPOINT_LOC_TYPE_DEVICE = 0,    ///< Endpoint在Device上
    ENDPOINT_LOC_TYPE_HOST = 1,      ///< Endpoint在Host上
} EndpointLocType;

/**
 * @brief Endpoint位置类型结构体
 * @note 支持EndpointLocType的扩展，最大60字节内容
 */
typedef struct {
    EndpointLocType locType;        ///< Endpoint的位置类别
    union {
        uint8_t raws[60];           ///< 通用数据
        struct {
            uint32_t devPhyId;      ///< 设备物理Id
            uint32_t superDevId;    ///< 超节点deviceId
            uint32_t serverIdx;     ///< Server的索引
            uint32_t superPodIdx;   ///< 超节点位置索引
        } device;                   ///< 当locType为DEVICE时使用
        struct {
            uint32_t id;            ///< 普通Id，当locType为HOST等时可能使用
        } host;
    };
} EndpointLoc;

/**
 * @brief 通信设备端点描述结构体
 */
typedef struct {
    CommProtocol protocol;  ///< 通信协议
    CommAddr commAddr;      ///< 通信地址
    EndpointLoc loc;        ///< Endpoint的位置信息
    union {
        uint8_t raws[52];   ///< 通用数据
    };
} EndpointDesc;

/**
 * @enum CommMemType
 * @brief 内存类型枚举定义
 */
typedef enum {
    COMM_MEM_TYPE_INVALID = -1, ///< 无效的内存类别
    COMM_MEM_TYPE_DEVICE = 0,   ///< 设备侧内存（如NPU等）
    COMM_MEM_TYPE_HOST = 1,     ///< 主机侧内存
} CommMemType;

/**
 * @brief 内存段元数据描述结构体
 */
typedef struct {
    CommMemType type; ///< 内存物理位置类型，参见CommMemType
    void *addr;       ///< 内存地址
    uint64_t size;    ///< 内存区域字节数
} CommMem;

/**
 * @brief 兼容Abi字段结构体
 */
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} CommAbiHeader;

/**
 * @brief 套接字角色
 */
typedef enum {
    HCOMM_SOCKET_ROLE_RESERVED = -1, ///< 保留的套接字角色
    HCOMM_SOCKET_ROLE_CLIENT = 0,    ///< 客户端角色，用于发起连接
    HCOMM_SOCKET_ROLE_SERVER = 1,    ///< 服务器角色，用于监听连接
} HcommSocketRole;

/**
 * @brief 通道描述参数
 * @note 结构体末尾扩展需要自增版本号，并补充兼容处理逻辑。
 */
typedef struct {
    CommAbiHeader header;            ///< ABI头部，包含版本等信息
    EndpointDesc remoteEndpoint;     ///< 远端网络设备端侧描述
    uint32_t notifyNum;              ///< channel上使用的通知消息数量
    bool exchangeAllMems;            ///< true表示无需显式传入memHandles
    HcommMemHandle *memHandles;      ///< 注册到通信域的待交换内存句柄
    uint32_t memHandleNum;           ///< 待交换内存句柄数量
    HcommSocket socket;              ///< 预创建socket句柄
    HcommSocketRole role;            ///< 本端角色(SERVER或CLIENT)
    uint16_t port;                   ///< 监听端口或目标端口
    union {
        uint8_t raws[128];           ///< 通用缓存
        struct {
            uint32_t queueNum;       ///< QP数量
            uint32_t retryCnt;       ///< 最大重传次数
            uint32_t retryInterval;  ///< 重传间隔（ms）
            uint8_t tc;              ///< 流量类别（QoS)
            uint8_t sl;              ///< 服务等级（QoS)
            uint32_t qpThreshold;    ///< 多QP场景下，每个QP最小数据量(B)
        } roceAttr;
        struct {
            uint32_t qos;            ///< HCCS QoS
        } hccsAttr;
        struct {
            uint32_t sqDepth;         ///< UB队列深度，0表示使用默认值
        } ubAttr;
    };
} HcommChannelDesc;

/**
 * @brief 初始化EndpointDesc结构体
 *
 * @param[inout] endpoint 返回的端点描述参数
 * @param[in] num 描述数量
 * @return HcommResult 执行结果状态码
 */
static inline HcommResult EndpointDescInit(EndpointDesc *endpoint, uint32_t num)
{
    const HcommResult hcommEPointer = (HcommResult)2; // 对齐HCCL_E_PTR

    for (uint32_t idx = 0; idx < num; ++idx) {
        if (endpoint == NULL) {
            return hcommEPointer;
        }

        // 用0xFF填充整个结构体 
        (void)memset_s(endpoint, sizeof(EndpointDesc), 0xFF, sizeof(EndpointDesc));

        // 显式设置关键字段为无效值 
        endpoint->protocol = COMM_PROTOCOL_RESERVED;
        endpoint->commAddr.type = COMM_ADDR_TYPE_RESERVED;
        endpoint->loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
        ++endpoint; // 移动到下一个描述符 
    }

    return 0;
}

/**
 * @brief 初始化HcommChannelDesc结构体
 *
 * @param[inout] channelDesc 返回的通道描述参数
 * @param[in] descNum 描述数量
 * @return HcommResult 执行结果状态码
 */
static inline HcommResult HcommChannelDescInit(HcommChannelDesc *channelDesc, uint32_t descNum)
{
    const HcommResult hcommEPointer = (HcommResult)2; // 对齐HCCL_E_PTR
    const HcommResult hcommEInternal = (HcommResult)4; // 对齐HCCL_E_INTERNAL

    for (uint32_t idx = 0; idx < descNum; ++idx) {
        if (channelDesc == NULL) {
            return hcommEPointer;
        }
        
        (void)memset_s(channelDesc, sizeof(HcommChannelDesc), 0xFF, sizeof(HcommChannelDesc));
        channelDesc->header.version = HCOMM_CHANNEL_VERSION;
        channelDesc->header.magicWord = HCOMM_CHANNEL_MAGIC_WORD;
        channelDesc->header.size = sizeof(HcommChannelDesc);
        channelDesc->header.reserved = 0;
        channelDesc->notifyNum = 0;
        channelDesc->exchangeAllMems = false;
        channelDesc->memHandles = NULL;
        channelDesc->memHandleNum = 0;
        channelDesc->socket = NULL;
        channelDesc->role = HCOMM_SOCKET_ROLE_RESERVED;
        channelDesc->port = 0;
        if (EndpointDescInit(&channelDesc->remoteEndpoint, 1) != 0) {
            return hcommEInternal;
        }
        ++channelDesc;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_RES_DEFS_H
