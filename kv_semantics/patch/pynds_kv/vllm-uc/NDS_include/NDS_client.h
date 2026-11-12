// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef __NDS_CLIETN_H__
#define __NDS_CLIETN_H__

// #define _GNU_SOURCE

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>
// #include <stdatomic.h>
#include <algorithm>
#include <atomic>
#include <pthread.h>
#include <semaphore.h>
#include <setjmp.h>
#include <signal.h>
#include <zlib.h>  // 需要安装 zlib-devel
#include "NDS_message_interface.h"
#include "log.h"
#include "xrt_rdma_neg_msg.h"

static inline void cpu_relax()
{
#if defined(__i386__) || defined(__x86_64__)
    // x86 架构
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    // ARM 架构 (yield 相当于 x86 的 pause)
    __asm__ __volatile__("yield" ::: "memory");
#else
    // 其他架构，使用编译器屏障
    __asm__ __volatile__("" ::: "memory");
#endif
}

// int cid = 0;    // 全局递增
#define MAX_CID_COUNT 50000
#define MAX_DPU_NUM 64
#define QP_NUM_T 9
#define IP_NUM_T 400
#define MAX_LINE_LEN 256
#define DEFAULT_KEEPALIVE_INTERVAL 1
typedef enum {
    KV_CM_OP_CONN_REQ = 0,     // 连接请求
    KV_CM_OP_CONN_RESP = 1,    // 连接响应
    KV_CM_OP_TERM_REQ = 2,     // 断链请求
    KV_CM_OP_TERM_RESP = 3,    // 断链响应
    KV_CM_OP_GET_CAP = 4,      // 获取能力请求
    KV_CM_OP_GET_CAP_RESP = 5  // 获取能力响应
} kv_cm_opcode_t;

typedef enum {
    ROCE_PROTO = 0,
    UB_PROTO = 1,
} kv_protocol_opcode_t;

// 私有kv协议
struct kv_cm_conn_cap {
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t queue_num;
    uint16_t adminq_depth;
    uint16_t ioq_depth;
    uint16_t ioq_key_batch;
    uint16_t total_key_batch;
    uint32_t sr_mdts;
    uint32_t bsr_mdts;
    uint16_t store_key_batch;
    uint16_t retrieve_key_batch;
    uint16_t delete_key_batch;
    uint16_t exist_key_batch;
};

struct kv_cm_conn_cap_client_server {
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t kato;
    uint8_t mtu;
};

struct kv_cm_conn_roce_info {
    union ibv_gid gid;       // client的gid
    uint32_t lid;            // ibv_query_port查询
    uint8_t roce_mtu;        // 参考ibv_mtu：4096 = 5，2048 = 4...
    uint8_t max_read_depth;  // 网卡read接收的并发能力,ibv_query_device查询（max_qp_rd_atom）
    uint8_t rnr_retry;       // 用7，recv not ready无限重试
    uint8_t min_rnr_timer;   // 12, 0.64ms
    uint8_t retry_cnt;       // 用7，重传7次
    uint8_t retry_timeout;   // 17，参考内核CMA_IBOE_PACKET_LIFETIME + 1(1次的网络rtt时间，约0.5S)
    uint8_t sl;              // 流控优先级，推荐用3
    uint8_t traffic_class;   // trust使用dscp场景下使用，推荐使用104=》tos 26
    uint8_t rsv0[2];
    uint16_t
        queue_num;  // qpn的数量，是否要限定最大的queue数量，影响是否要使用0长数组——讨论首版本最大64个qpn
    uint8_t rsv1[2];
    uint32_t qpn[MAX_QP_NUM];
};

typedef struct ip_pair {
    char local_ip[32];
    char server_ip[32];
    int server_port;
    int roce_idx;
    int roce_port;
    int dpu_id;
} gids_ip_pair_t;

typedef struct param {
    gids_ip_pair_t ip[IP_NUM_T];
    int ip_count;
    int start_core;
    LogLevel log_level;
    int qp_slot_num;    // qp队列深度
    int use_qp_num;     // 可用数据传输qp个数
    int max_batch_num;  // 单Req下发最大Batch数, 超过会做拆分
    int core_num;
    uint8_t priority;  // 配置优先级流控制
    uint8_t tos;       // TOS 字段在 IP 数据包中用于指示数据包的优先级和服务类型
    uint16_t kato;
    bool timer_switch;
    uint32_t busy_duration;
    uint32_t heartbeat_interval;
} param_t;

struct rdma_connection {
    // TCP连接
    int clnt_fd;

    // KV能力信息
    struct kv_cm_conn_cap kv_cap_info;

    // RDMA资源
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_cq *send_cq[MAX_QP_NUM];
    struct ibv_cq *recv_cq[MAX_QP_NUM];
    struct ibv_qp *qps[MAX_QP_NUM];
    struct ibv_mr *rw_mr[SLAB_NUM];
    struct ibv_mr *send_mr[MAX_QP_NUM];
    struct ibv_mr *recv_mr[MAX_QP_NUM];
    struct ibv_mr *ret_mr;
    void *rw_buf[SLAB_NUM];
    uint64_t slab_size[SLAB_NUM];
    uint64_t ret_buf_size;
    void *send_buf[MAX_QP_NUM];
    void *recv_buf[MAX_QP_NUM];
    void *ret_buf;

    // 内核模块资源
    void *map_addr[SLAB_NUM];
    void *notify_addr;

    // 心跳和处理线程
    volatile int is_alive;           // 连接状态标志位
    struct timespec last_heartbeat;  // 最后一次心跳时间
    pthread_t recv_parser_thread;    // 接收处理线程
    pthread_t send_recv_parser_thread;
    pthread_t heartbeat_thread_rx;

    // 本地属性
    int gid_index;
    int port_num;
    xrt_rdma_neg_qp_attr_t client_roce_info;

    // 远端属性
    xrt_rdma_neg_qp_attr_t server_roce_info;

    // 辅助字段：用于哈希标识
    int conn_idx;
    int qp_num;
    std::atomic<bool> online;  // 连接在线状态标志
    std::atomic<bool> is_destroying;
    pthread_mutex_t destroy_connection_lock;
    std::atomic<bool> is_reconnecting;  // 是否正在重连
    pthread_mutex_t reconnection_lock;  // 重连锁
    std::atomic<bool> is_starting_timer;
    pthread_mutex_t start_timer_lock;
    int reconnection_count;
    gids_ip_pair_t ip;
    uint64_t io_count;
    uint64_t old_io_count;
    uint64_t connSendCounter;
    uint64_t connRecvCounter;
    uint8_t control_recv_buf[sizeof(xrt_rdma_neg_msg_t)];
    size_t control_recv_len;
    bool is_used;       // 在建链的时候标记为true，更新视图时若对应的dpuip被移除标记为false
    bool is_destroyed;  // s_destroyed初始化为true，建链成功或失败都为false，视图变更销毁成功为true
};

struct kv_cm_hdr {
    uint8_t major_version;
    uint8_t minor_version;
    uint8_t opcode;
    uint8_t protocol;
    uint32_t msg_len;
};

struct kv_cm_get_cap {
    struct kv_cm_hdr hdr;
};

struct kv_cm_get_cap_resp {
    struct kv_cm_hdr hdr;
    uint16_t cmd_depth;
    uint16_t max_queue;
    uint16_t key_depth;
    uint16_t batch_num;
};

#pragma pack(push, 4)
struct kv_cm_conn_req {
    struct kv_cm_hdr hdr;
    uint32_t
        kato;  // 心跳超时时间,单位ms，推荐使用5s
               // (参考nvme，且大于7*0.5,0.5表示传输层没收到ack的超时时间，7代表重传次数)，demo可以先写0(没有心跳)
    uint16_t cmd_depth;  // 不大于cap时获取的
    uint8_t resv[2];
    union {
        struct kv_cm_conn_roce_info roce_info;
    };
} __attribute__((aligned(4)));
#pragma pack(pop)

#pragma pack(push, 4)
struct kv_cm_conn_resp {
    struct kv_cm_hdr hdr;
    uint16_t status;
    uint8_t resv[2];
    struct kv_cm_conn_roce_info roce_info;
} __attribute__((aligned(4)));
#pragma pack(pop)

struct nvme_common_command {
    __u8 opcode;
    __u8 flags;
    __u16 command_id;
    __le32 nsid;
    __le32 cdw2[2];
    __le64 metadata;
    __le64 dptr[2];
    __le32 cdw10[6];
};

struct nvme_completion {
    /*
     * Used by Admin and Fabrics commands to return data:
     */
    union nvme_result {
        __le16 u16;
        __le32 u32;
        __le64 u64;
    } result;
    __le16 sq_head;   /* how much of this queue may be reclaimed */
    __le16 sq_id;     /* submission queue that generated this entry */
    __u16 command_id; /* of the command which completed */
    __le16 status;    /* did the command fail, and if so, why? */
};
extern struct rdma_connection conn[IP_NUM_T];

static inline void ipc_storage_message_transfer(const IPC_data_block *ipcDataBlock,
                                                data_block *dataBlock, uint32_t rkey)
{
    dataBlock->nsid = ipcDataBlock->nsid;
    dataBlock->task_id = ipcDataBlock->task_id;

    dataBlock->batch_num = ipcDataBlock->addr_num;
    dataBlock->dptr_buf = 0;

    dataBlock->dptr_len = ipcDataBlock->addr_num * 32;
    dataBlock->dptr_type = 1;

    for (int i = 0; i < ipcDataBlock->addr_num; i++) {
        dataBlock->addr_pair_list[i].block_id.key_low =
            ipcDataBlock->addr_pair_list[i].block_id.key_low;
        dataBlock->addr_pair_list[i].block_id.key_high =
            ipcDataBlock->addr_pair_list[i].block_id.key_high;
        dataBlock->addr_pair_list[i].hbm_addr = ipcDataBlock->addr_pair_list[i].hbm_addr;
        dataBlock->addr_pair_list[i].offset = ipcDataBlock->addr_pair_list[i].offset;
        dataBlock->addr_pair_list[i].length = ipcDataBlock->addr_pair_list[i].length;
        dataBlock->addr_pair_list[i].remote_key = rkey;
    }
    // cid += 1;
}

static inline void ipc_storage_message_transfer_single(const IPC_data_block *ipcDataBlock,
                                                       data_block_single *dataBlock, uint32_t rkey)
{
    dataBlock->nsid = ipcDataBlock->nsid;
    dataBlock->task_id = ipcDataBlock->task_id;
    uint8_t type = 0x40;
    dataBlock->rsvd1 = 0x40;

    dataBlock->dptr_buf = ipcDataBlock->addr_pair_list[0].hbm_addr;

    dataBlock->dptr_len_key =
        (ipcDataBlock->addr_pair_list[0].length & 0x00FFFFFF) | ((rkey & 0x000000FF) << 24);
    dataBlock->dptr_key_type = ((rkey >> 8) & 0x00FFFFFF) | ((uint32_t)type << 24);
    dataBlock->offset = ipcDataBlock->addr_pair_list[0].offset;
    dataBlock->length = ipcDataBlock->addr_pair_list[0].length & 0x00FFFFFF;  // 取低24位
    dataBlock->block_id.key_low = ipcDataBlock->addr_pair_list[0].block_id.key_low;
    dataBlock->block_id.key_high = ipcDataBlock->addr_pair_list[0].block_id.key_high;
}

static inline void ipc_storage_message_transfer_scatter_indexed(const IPC_data_block *ipcDataBlock,
                                                                data_block *dataBlock,
                                                                uint32_t rkey, const int *indices,
                                                                int count)
{
    dataBlock->nsid = ipcDataBlock->nsid;
    dataBlock->task_id = ipcDataBlock->task_id;
    dataBlock->batch_num = count;  // 这一片（Batch）的大小

    dataBlock->dptr_buf = 0;
    dataBlock->dptr_len = count * 36;
    dataBlock->dptr_type = 1;
    uint8_t type = 0x40;
    dataBlock->rsvd1 = 0x40;

    for (int i = 0; i < count; i++) {
        // 核心修改：从索引数组中获取该 key 在原始 ipcDataBlock 中的位置
        int src_idx = indices[i];

        dataBlock->addr_pair_list[i].offset = ipcDataBlock->addr_pair_list[src_idx].offset;
        dataBlock->addr_pair_list[i].block_id.key_low =
            ipcDataBlock->addr_pair_list[src_idx].block_id.key_low;
        dataBlock->addr_pair_list[i].block_id.key_high =
            ipcDataBlock->addr_pair_list[src_idx].block_id.key_high;
        dataBlock->addr_pair_list[i].hbm_addr = ipcDataBlock->addr_pair_list[src_idx].hbm_addr;
        dataBlock->addr_pair_list[i].length =
            (ipcDataBlock->addr_pair_list[src_idx].length & 0x00FFFFFF) |
            ((rkey & 0x000000FF) << 24);
        // printf("dataBlock->addr_pair_list[i].length: %u\n", dataBlock->addr_pair_list[i].length);
        dataBlock->addr_pair_list[i].remote_key =
            ((rkey >> 8) & 0x00FFFFFF) | ((uint32_t)type << 24);
    }
}

static inline void ipc_storage_message_transfer_ctl_scatter(IPC_data_block *ipcDataBlock,
                                                            ctl_block *ctlBlock, const int *indices,
                                                            int count)
{
    ctlBlock->nsid = ipcDataBlock->nsid;
    ctlBlock->task_id = ipcDataBlock->task_id;
    ctlBlock->batch_num = count;

    ctlBlock->dptr_buf = 0;
    ctlBlock->dptr_len = count * sizeof(block_key);
    ctlBlock->dptr_type = 1;
    ctlBlock->rsvd1 = 0x40;

    ctlBlock->access_control = 0;
    for (int i = 0; i < count; i++) {
        int src_idx = indices[i];
        ctlBlock->key[i].key_low = ipcDataBlock->addr_pair_list[src_idx].block_id.key_low;
        ctlBlock->key[i].key_high = ipcDataBlock->addr_pair_list[src_idx].block_id.key_high;
        // printf("ipcDataBlock->addr_pair_list[src_idx].block_id.key_low: %lu\n",
        // ipcDataBlock->addr_pair_list[src_idx].block_id.key_low);
        // printf("ctlBlock->key[i].key_low: %lu\n", ctlBlock->key[i].key_low);
    }
}

extern void **g_kv_cache;
extern int g_mr_num;
extern AppConfig g_cfg;
extern int exit_flag;
extern uint64_t counter;
extern int acquire_cid();
extern int cid_to_slot[MAX_CID_COUNT];
typedef struct {
    sem_t window_sem;           // 流控信号量
    std::atomic<int> inflight;  // 在途计数（可选，用于调试）
    pthread_mutex_t slot_lock;
    bool *slot_used;
    bool *ack_arrived;
    bool *resp_arrived;
    int *slot_to_cid;  // reverse mapping: slot_id -> cid, -1 if unused
} QPContext;
extern QPContext g_qp_ctx[IP_NUM_T][QP_NUM_T];

typedef enum {
    IO_ERR_INVALID_CMD = 1,
    IO_ERR_INVALID_FIELD = 2,
    IO_ERR_INTERNAL = 6,
    IO_ERR_WRITE_FAULT = 640,
    IO_ERR_UNRECOVERED_READ = 641,
    IO_ERR_KEY_NOT_EXIST = 1793,
    IO_ERR_OUT_OF_SIZE = 1810,
    IO_ERR_IO_TIMEOUT = 1814,
    IO_ERR_KEY_EXISTED = 1827,
    IO_ERR_RESOURCE_BUSY = 1841,
} io_error_code_t;

void die(const char *msg);
struct ibv_device *find_ib_device_by_ip(const char *target_ip);
void wire_gid_to_gid(const char *wgid, union ibv_gid *gid);
void gid_to_wire_gid(const union ibv_gid *gid, char wgid[]);
void init_rdma_connection(struct rdma_connection *conn);
NDS_ERROR_T get_roce_info_base_tcp(struct rdma_connection *conn, int qp_num);
NDS_ERROR_T setup_tcp_connection(struct rdma_connection *conn, const char *local_ip,
                                 const char *server_ip, int port);
NDS_ERROR_T setup_rdma_connection(struct rdma_connection *conn, int gid_index, int port_num);
NDS_ERROR_T transfer_data_send(int qpID, int ipIdx, struct rdma_connection *conn,
                               const void *message, uint16_t batchNum);
NDS_ERROR_T transfer_data_send_end(int qpID, struct rdma_connection *conn, const void *message);

void transfer_data_recv();
bool destroy_connection(struct rdma_connection *conn);
void destroy_connection_locked(struct rdma_connection *conn);
extern NDS_ERROR_T init_rdma_device_and_mr(struct rdma_connection *conn, const char *local_ip,
                                           void **kv_cache, int mr_num);

typedef struct {
    uint64_t task_id;                            // reqid+layerid融合信息，只用于notify
    uint64_t mr_id;                              // 标志hbm kvcache起始地址
    uint16_t addr_num;                           // 包内block_id_list中有效块数 batch_num
    uint32_t nsid;                               // 工具配置项需填入
    addr_pair addr_pair_list[MAX_ADDR_NUM_IPC];  // 地址对数组
} NDS_REQ;

typedef struct kv_mr {
    uint64_t addr;
    uint64_t len;
} kv_cache;

typedef struct {
    std::vector<int> batch_code_set;
    std::vector<int> key_code_set;
} CODE_SET;

NDS_ERROR_T nds_init(void **kv_cache, int mr_num);  // mr_num可能是layer个数
uint64_t nds_kv_exist(NDS_REQ *req);                // 同步接口
NDS_ERROR_T nds_kv_delete(NDS_REQ *req);            // 同步接口
NDS_ERROR_T nds_kv_get(NDS_REQ *req);
NDS_ERROR_T nds_kv_put(NDS_REQ *req);
NDS_ERROR_T nds_kv_get_single(NDS_REQ *req);
NDS_ERROR_T nds_kv_put_single(NDS_REQ *req);
CODE_SET nds_kv_wait(uint64_t task_id, uint32_t total_blocks);
void nds_kv_get_nds_config(const char *path_nds_config);

#define MAX_CONNS_PER_DPU 100
#define VNODE_PER_DPU 50   // 每个物理节点的虚拟节点数, dpu个数
#define VNODE_PER_CONN 50  // 每个物理节点的虚拟节点数
#define RING_SIZE (IP_NUM_T * VNODE_PER_CONN)
// -- 新增多dpu链接相关，hash打散相关 --

// --- 哈希环与任务上下文 ---
typedef struct {
    int dpu_id;
    struct rdma_connection *conns[MAX_CONNS_PER_DPU];  // 该 DPU 下的所有连接
    int conn_count;
    uint64_t next_conn_idx;  // 用于 DPU 内部的 Round-Robin 轮询
} dpu_group_t;

typedef struct {
    uint64_t hash_coord;
    struct rdma_connection *h_conn;
    dpu_group_t *target_dpu;
} hash_node_t;

typedef struct {
    hash_node_t nodes[RING_SIZE];
    int total_nodes;
    pthread_rwlock_t lock;
} hash_ring_t;

typedef struct {
    uint64_t task_id;
    std::atomic<int> expected_packets;  // 该 Task 拆分出的总包数
    std::atomic<int> received_packets;  // 已收到的回包数
} task_ctx_t;

void hash_ring_init(hash_ring_t *ring);
void hash_ring_reset(hash_ring_t *ring);
void hash_ring_add_conn(hash_ring_t *ring, struct rdma_connection *tmp_conn);
void hash_ring_add_dpu(hash_ring_t *ring, dpu_group_t *dpu);
struct rdma_connection *hash_ring_lookup(hash_ring_t *ring, uint64_t h);
extern void cleanup_conn_cids(int conn_idx);

NDS_ERROR_T rebuild_cluster_hash_ring();

typedef enum { SINGLE = 0, BATCH = 1, CLT_EXIST_DELETE = 2, HEARTBEAT = 3 } operation_type;

typedef struct {
    int original_indices[MAX_CTL_ADDR_NUM];  // 该 CID 包包含的 Key 在原始数组中的下标
    int count;                               // 包含的 Key 数量
    int qp_index;                            // qp
    int ip_index;                            // 链接
    struct rdma_connection *tmp_conn;
    operation_type optype;  // 1 batch; 0 single ; 2 clt:exist/delete
    data_block send_msg;
    data_block_single single_send_msg;
    ctl_block ctl_send_msg;
    bool released;  // true = already released to pool, prevent double release
} cid_metadata_t;

extern cid_metadata_t g_cid_metadata[MAX_CID_COUNT];

extern int acquire_cid();
extern void release_cid(uint16_t val);

typedef struct {
    int indices[MAX_ADDR_NUM_IPC];
    int count;
} conn_group_t;

#endif  // END __NDS3_CLIETN_H__
