// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_COMMON__H__
#define __NDS_COMMON__H__

#include <stddef.h>
#include <stdint.h>

#define MAX_ADDR_NUM_IPC 4096  // 每个NDS_REQ内batch的上限
#define MAX_ADDR_NUM 128       // 存储包每个IO请求Batch上限
#define MAX_CTL_ADDR_NUM 512   // 控制接口（exist）Batch上限
#define MAX_QP_REQ_ADDR 128    // 每个IO请求Batch大小

#define MAX_QP_NUM 63               // 每个链接最大QP个数
#define RQ_DEPTH 128                // RQ 深度，影响Recv Buffer的大小
#define HEARTBEAT_TIMEOUT 10000000  // 超时时间(us)
#define CQE_BATCH 8                 // 每次从CQ中轮询的完成事件数

#define MAX_QP_WR 8              // 每个QP内队列深度，支持同时处理请求个数上限
#define MAX_SLOTS MAX_QP_WR      // QP发送请求槽位个数
#define SLOT_SIZE (8192 + 4096)  // QP发送请求槽位大小

#define SEND_MR_SIZE SLOT_SIZE *MAX_SLOTS                // 每个QP所分配Send Buf大小
#define RECV_MR_SIZE RQ_DEPTH * sizeof(recv_data_block)  // recv收到的是uint64_t的task_id

#define TCP_TIMEOUT_SEC 5
#define KV_KATO 5000
#define ADMIN_QP 0

#define SLAB_NUM 999
#define CQ_QUEN_HEAD "/IPC_for_task_"
#ifdef __cplusplus
using namespace std;
#endif

typedef struct {
    int max_qp_wr;
    int max_slots;
    int slot_size;
    int max_qp_req_addr;
    int io_qp_num;
    int max_batch_num;
    size_t send_mr_size;
    uint8_t priority;
    uint8_t tos;
    int ip_count;
    uint16_t kato;
    bool timer_switch;
    uint32_t busy_duration;
    int *interval;
    int interval_count;
    uint32_t heartbeat_interval;
    int sceneMode;
} AppConfig;

#pragma pack(push, 1)
typedef struct {
    uint64_t key_low;     // block_hash作为存储的key[7:0]
    uint16_t key_high;    // key[9:8]
    uint16_t reserve[3];  // [15:10]
} block_key;
#pragma pack(pop)

// #pragma pack(push, 1)
// typedef struct {
//     uint32_t offset;        // 存储侧offset
//     block_key block_id;
//     uint64_t hbm_addr;      // HBM地址
//     uint32_t length_mr_key0;        // block 大小，实际只用了前24bit，高8bit为reserve
//     uint32_t mr_keys_dptr_type1;
// } addr_pair;
// #pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    uint32_t offset;      // 存储侧offset
    block_key block_id;   // 块ID，为block_hash作为存储的key
    uint64_t hbm_addr;    // HBM地址
    uint32_t length;      // block 大小
    uint32_t remote_key;  // hbm的mr key, 上层业务不用填
} addr_pair;
#pragma pack(pop)

/* 上层应用携带参数 */
typedef enum __attribute__((aligned(2))) {
    KV_LOAD = 0,  // NPU 读存储kvcache
    KV_WRITE,     // NPU 写存储kvcache
    KV_LOAD_BATCH,
    KV_WRITE_BATCH,
    HEARTBEAT,          // 心跳检测
    INIT_COMM_LINK,     // 初始化通信链路
    DISTROY_COMM_LINK,  // 断开通信链路
    IS_EXIST,           // 查询命中指令
    KV_DELETE,
} OP_TYPE;

#ifdef __cplusplus
enum OP_CODE : uint8_t {
#else
typedef enum OP_CODE {
#endif
    Store = 1,
    Retrieve = 2,
    Batch_Store = 69,
    Batch_Retrieve = 70,
    Delete = 8,
    Exist = 12,
    Heart_Beat = 244,
};
#ifndef __cplusplus
typedef uint8_t OP_CODE;
#endif

typedef int NDS_ERROR_T;
#define NDS_SUCCESS 0            // 操作成功
#define NDS_FAILED 1             // 通用/未知错误
#define NDS_ERR_INVALID_PARAM 2  // 参数无效（如传入空指针）
#define NDS_ERR_BUSY 3           // 设备忙/资源被占用
#define NDS_ERR_TIMEOUT 4        // 操作超时

#define All_CONN_ERR -34
#define NDS_TCP_CONN_ERR -31
#define NDS_GET_CAP_ERR -32
#define NDS_GET_CAP_RES_ERR -33
#define IN_FLIGHT_MESSAGE_LIST_NULL -1
#define IN_FLIGHT_LIST_LEN_MAX -2
#define IN_FLIGHT_LIST_TASK_MAX -3
#define IN_FLIGHT_LIST_REMOVE_ERROR -4
#define NDS_NOTIFY_ERROR -5
#define NDS_COMMON_ERR -6
#define NDS_WRITE_ERR -7
#define TASK_INFO_NULL -10
#define MQ_OPEN_FAILED -11
#define MQ_SEND_FAILED -12
#define MQ_TYPE_ERROR -20
#define ADDR_TYPE_ERROR -21

/* 盘框返回错误码 */
#define KR_SUCCESS 0
#define KR_FAILED 1
#define KR_EXIST 1

#endif
