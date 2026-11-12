// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "NDS_client.h"
#include <atomic>
#include <inttypes.h>
#include <iostream>
#include <mutex>
#include <openssl/md5.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "agent_adapter.h"
#include "cluster.h"
#include "epoll.h"
#include "keepalive.h"
#include "utils.h"

#define XXH_STATIC_LINKING_ONLY /* access advanced declarations */
#define XXH_IMPLEMENTATION      /* access definitions */
#define TASKIDNOTFIND -1
#define TIMEOUTRETRY -2
#define NDSERRORINVALIDOPTYPE -3
#define SENDRETRYFAILED -4
#define IPBECLEANED -5
#define SENDFAILED -6
#define POLLCQERROR -7
#define SDK_VERSION_INFO "SDK version 10.28"
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MIN3(x, y, z) MIN(MIN(x, y), z)

#include "../NDS_include/xxhash.h"
// #include <immintrin.h>
#include <chrono>
#include <iostream>
#include <time.h>

int exit_flag = 0;
struct rdma_connection conn[IP_NUM_T];
struct rdma_connection g_conn[IP_NUM_T];
struct ibv_context *g_context;
volatile int task_done = 0;
static uint16_t cid_pool[MAX_CID_COUNT];
// static int cid_top = -1;
pthread_mutex_t cid_pool_lock = PTHREAD_MUTEX_INITIALIZER;

int cid_to_slot[MAX_CID_COUNT];
unsigned int g_notify_addr[MAX_CID_COUNT];
volatile int send_back_status[MAX_CID_COUNT];
int g_hitmap[MAX_CID_COUNT][512];
QPContext g_qp_ctx[IP_NUM_T][QP_NUM_T];

static int cid_head = 0;
static int cid_tail = 0;
static int cid_count = 0;

const char *global_path_nds_config;
uint32_t g_io_request_timeout = 10000;  // 10 秒超时
uint32_t g_io_max_retry_count = 1;

uint64_t counter = 0;

std::unordered_map<uint64_t, std::vector<uint16_t>> g_task_cid_map;
std::mutex g_map_mutex;
pthread_t epoll_tid;
pthread_t keepalive_tid;

// -- 多DPU连接相关 --
// struct rdma_connection conn[IP_NUM_T];
task_ctx_t *cid_to_task_map[MAX_CID_COUNT];  // cid2 task_id

cid_metadata_t g_cid_metadata[MAX_CID_COUNT];
dpu_group_t g_dpus[MAX_DPU_NUM];
int g_dpu_count = 0;

std::vector<std::string> old_dpu_ip_list;
std::vector<uint32_t> old_nsid_list;
std::unordered_map<std::string, std::unordered_set<uint32_t>> old_dpuip_to_nsid;
std::unordered_map<uint32_t, std::vector<std::string>> old_nsid_to_dpuip;
std::unordered_map<std::string, int32_t> old_dpuip_to_dpuid;

typedef struct {
    int start_thread;
    int step;
} thread_args_t;

AppConfig g_cfg;

void init_runtime_configs(param_t *param)
{
    g_cfg.max_qp_wr = param->qp_slot_num;
    g_cfg.max_slots = g_cfg.max_qp_wr;
    g_cfg.slot_size = 8192 + 1024;
    g_cfg.max_qp_req_addr = 32;

    g_cfg.io_qp_num = param->use_qp_num;
    g_cfg.max_batch_num = param->max_batch_num;
    g_cfg.priority = param->priority;
    g_cfg.tos = param->tos;
    g_cfg.ip_count = param->ip_count;
    g_cfg.kato = param->kato;
    g_cfg.timer_switch = param->timer_switch;
    g_cfg.busy_duration = param->busy_duration;
    g_cfg.heartbeat_interval = param->heartbeat_interval;

    // 根据动态传入的值计算内存大小
    g_cfg.send_mr_size = (size_t)g_cfg.slot_size * g_cfg.max_slots;
}

hash_ring_t g_ring;

void **g_kv_cache;
int g_mr_num;

// --- 哈希环逻辑 ---

static uint64_t vnode_hash(const char *key)
{
    uint64_t hash = XXH3_64bits(key, strlen(key));
    // printf("hash %lu\n",hash);
    return hash;
}

static int compare_nodes(const void *a, const void *b)
{
    uint64_t h1 = ((hash_node_t *)a)->hash_coord;
    uint64_t h2 = ((hash_node_t *)b)->hash_coord;
    return (h1 > h2) ? 1 : ((h1 < h2) ? -1 : 0);
}

void hash_ring_init(hash_ring_t *ring)
{
    ring->total_nodes = 0;
    pthread_rwlock_init(&ring->lock, NULL);
}

void hash_ring_reset(hash_ring_t *ring)
{
    pthread_rwlock_wrlock(&ring->lock);
    ring->total_nodes = 0;
    pthread_rwlock_unlock(&ring->lock);
}

void hash_ring_add_conn(hash_ring_t *ring, struct rdma_connection *tmp_conn)
{
    pthread_rwlock_wrlock(&ring->lock);
    for (int i = 0; i < VNODE_PER_CONN; i++) {
        if (ring->total_nodes >= RING_SIZE) break;
        char name[64];
        snprintf(name, 64, "CONN-%d-V-%d", tmp_conn->conn_idx, i);
        ring->nodes[ring->total_nodes].hash_coord = vnode_hash(name);
        ring->nodes[ring->total_nodes].h_conn = tmp_conn;
        ring->total_nodes++;
    }
    qsort(ring->nodes, ring->total_nodes, sizeof(hash_node_t), compare_nodes);

    pthread_rwlock_unlock(&ring->lock);
}

void hash_ring_add_dpu(hash_ring_t *ring, dpu_group_t *dpu)
{
    pthread_rwlock_wrlock(&ring->lock);
    for (int i = 0; i < VNODE_PER_DPU; i++) {
        if (ring->total_nodes >= RING_SIZE) break;
        char name[64];
        snprintf(name, 64, "DPU-%d-V-%d", dpu->dpu_id, i);
        ring->nodes[ring->total_nodes].hash_coord = vnode_hash(name);
        ring->nodes[ring->total_nodes].target_dpu = dpu;
        ring->total_nodes++;
    }
    qsort(ring->nodes, ring->total_nodes, sizeof(hash_node_t), compare_nodes);
    pthread_rwlock_unlock(&ring->lock);
}

struct rdma_connection *hash_ring_lookup(hash_ring_t *ring, uint64_t h)
{
    pthread_rwlock_rdlock(&ring->lock);
    if (ring->total_nodes == 0) {
        pthread_rwlock_unlock(&ring->lock);
        return NULL;
    }
    int low = 0, high = ring->total_nodes - 1, idx = 0;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (ring->nodes[mid].hash_coord >= h) {
            idx = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    struct rdma_connection *c = ring->nodes[idx].h_conn;
    pthread_rwlock_unlock(&ring->lock);
    if (c && c->online && !c->is_destroying && !c->is_reconnecting) { return c; }
    return NULL;
}

struct rdma_connection *hash_ring_lookup_dpu(hash_ring_t *ring, uint64_t h)
{
    pthread_rwlock_rdlock(&ring->lock);
    if (ring->total_nodes == 0) {
        pthread_rwlock_unlock(&ring->lock);
        return NULL;
    }
    // 1: 二分查找定位DPU
    int low = 0, high = ring->total_nodes - 1, idx = 0;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (ring->nodes[mid].hash_coord >= h) {
            idx = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    dpu_group_t *dpu = ring->nodes[idx].target_dpu;
    pthread_rwlock_unlock(&ring->lock);

    // 2: DPU内部Round_Robin平均找conn
    if (dpu->conn_count == 0) return NULL;

    uint64_t temp_index = __sync_fetch_and_add(&dpu->next_conn_idx, 1);
    int start = temp_index % dpu->conn_count;
    for (int i = 0; i < dpu->conn_count; i++) {
        int c_sub_idx = (start + i) % dpu->conn_count;
        struct rdma_connection *c = dpu->conns[c_sub_idx];
        if (c && c->online && !c->is_destroying && !c->is_reconnecting) { return c; }
    }
    return NULL;
}

void PrintChangesDpuips(const std::vector<std::string> &Dpuips)
{
    for (const auto &ip : Dpuips) { LOG_W("%s ", ip.c_str()); }
}

void PrintChangesNsids(const std::vector<uint32_t> &Nsids)
{
    for (const auto &nsid : Nsids) { LOG_W("%u ", nsid); }
}

NDS_ERROR_T rebuild_cluster_hash_ring()
{
    std::unordered_map<std::string, std::unordered_set<uint32_t>> new_dpuip_to_nsid =
        BuildDpuipNsidMap();
    std::unordered_map<std::string, uint32_t> name_nsid_map = GetNameNsidMap();
    std::unordered_map<uint32_t, std::vector<std::string>> new_nsid_to_dpuip =
        GetNsidDpuipMap(name_nsid_map);
    std::vector<uint32_t> new_nsid_list = GetAllNsidList();
    std::vector<std::string> new_dpu_ip_list = GetAllInDpuipList();
    std::unordered_map<std::string, int32_t> new_dpuip_to_dpuid = GetAllInDpuipDpuidMap();

    // 第一步：找出新增dpu ip，只新建链接，找到is_used=false&&is_destroyed=true的conn对象
    std::vector<std::string> addDpuips = ParseNewAddDpuipList(old_dpu_ip_list, new_dpu_ip_list);
    if (addDpuips.size() != 0) {
        LOG_W("[rebuild_cluster_hash_ring] Added %ld dpuips:", addDpuips.size());
        PrintChangesDpuips(addDpuips);
    }
    SetUpAllAddDpuipConn(addDpuips, new_dpuip_to_dpuid);
    // 第二步：找出减少dpu ip，置is_used=false
    std::vector<std::string> deleteDpuips = ParseDeleteDpuipList(old_dpu_ip_list, new_dpu_ip_list);
    if (deleteDpuips.size() != 0) {
        LOG_W("[rebuild_cluster_hash_ring] Deleted %ld dpuips:", deleteDpuips.size());
        PrintChangesDpuips(deleteDpuips);
    }
    ModifyRemovedDpuipConnFlag(deleteDpuips);
    // 第三步：找出新增ns，将空的NsidConn加到map里，若此刻来io也找不到链接
    std::vector<std::uint32_t> addNsids = ParseNewAddNsids(old_nsid_list, new_nsid_list);
    if (addNsids.size() != 0) {
        LOG_W("[rebuild_cluster_hash_ring] Added %ld nsids:", addNsids.size());
        PrintChangesNsids(addNsids);
    }
    CreateAllAddNsidConn(addNsids);
    // 第四步：找出减少nsid，把对应的nsidConn对象从nsid_conn_map中移除
    std::vector<std::uint32_t> deleteNsids = ParseDeleteNsids(old_nsid_list, new_nsid_list);
    if (deleteNsids.size() != 0) {
        LOG_W("[rebuild_cluster_hash_ring] Deleted %ld nsids:", deleteNsids.size());
        PrintChangesNsids(deleteNsids);
    }
    DestroyRemovedOldNsidConn(deleteNsids);
    // 第五步：遍历所有链接，对is_used=true的链接加到新环里面去
    AddConnToNewRing(new_dpuip_to_nsid, new_dpuip_to_dpuid);
    // 第六步：遍历nsid_conn_map，都切到新环
    NsidConnMapSwitchRing();
    // 第七步：遍历nsid_conn_map，获取老环的写锁，销毁老环
    NsidConnMapReleaseOldRing();
    // 第八步：遍历所有链接，
    // 若is_used=false，online=true，is_destroying=false，is_reconnecting=false，去销毁，销毁成功标记is_destroyed=true
    DestroyRemovedConn();

    old_dpuip_to_nsid = new_dpuip_to_nsid;
    old_nsid_to_dpuip = new_nsid_to_dpuip;
    old_nsid_list = new_nsid_list;
    old_dpu_ip_list = new_dpu_ip_list;
    old_dpuip_to_dpuid = new_dpuip_to_dpuid;
    return NDS_SUCCESS;
}

// -- 多DPU连接相关 --

void init_qp_contexts()
{
    for (int k = 0; k < IP_NUM_T; k++) {
        for (int i = 0; i < QP_NUM_T; i++) {
            sem_init(&g_qp_ctx[k][i].window_sem, 0, g_cfg.max_qp_wr);
            if (pthread_mutex_init(&g_qp_ctx[k][i].slot_lock, NULL) != 0) {
                perror("pthread_mutex_init failed");
            }
            size_t n = g_cfg.max_slots;
            g_qp_ctx[k][i].slot_used = (bool *)calloc(n, sizeof(bool));
            g_qp_ctx[k][i].ack_arrived = (bool *)calloc(n, sizeof(bool));
            g_qp_ctx[k][i].resp_arrived = (bool *)calloc(n, sizeof(bool));
            g_qp_ctx[k][i].slot_to_cid = (int *)calloc(n, sizeof(int));
            for (size_t j = 0; j < n; j++) { g_qp_ctx[k][i].slot_to_cid[j] = -1; }

            // 检查分配是否成功
            if (!g_qp_ctx[k][i].slot_used || !g_qp_ctx[k][i].slot_to_cid) {
                perror("Failed to allocate memory for QP slots");
                exit(0);
            }
        }
    }
}

void die(const char *msg)
{
    perror(msg);
    exit(1);
}

// 初始化：按顺序填满池子
void init_cid_pool()
{
    pthread_mutex_lock(&cid_pool_lock);
    for (int i = 0; i < MAX_CID_COUNT; i++) {
        cid_pool[i] = (uint16_t)i;
        cid_to_slot[i] = -1;
        g_notify_addr[i] = 0;
        g_cid_metadata[i].released = false;  // 初始状态：未释放
    }
    cid_head = 0;
    cid_tail = 0;
    cid_count = MAX_CID_COUNT;
    pthread_mutex_unlock(&cid_pool_lock);
}

// 获取 CID：从头部取 (FIFO)
int acquire_cid()
{
    int res = -1;
    pthread_mutex_lock(&cid_pool_lock);
    if (cid_count > 0) {
        res = cid_pool[cid_head];
        cid_head = (cid_head + 1) % MAX_CID_COUNT;
        cid_count--;
        g_cid_metadata[res].released = false;
        g_cid_metadata[res].count = 0;
        g_cid_metadata[res].ip_index = -1;
        g_cid_metadata[res].qp_index = -1;
        g_cid_metadata[res].tmp_conn = NULL;
        memset(g_cid_metadata[res].original_indices, 0,
               sizeof(g_cid_metadata[res].original_indices));
        memset(&g_cid_metadata[res].send_msg, 0, sizeof(data_block));
        memset(&g_cid_metadata[res].single_send_msg, 0, sizeof(data_block_single));
        memset(&g_cid_metadata[res].ctl_send_msg, 0, sizeof(ctl_block));
        memset(g_hitmap[res], (uint8_t)SENDRETRYFAILED, 512 * sizeof(g_hitmap[res][0]));
        send_back_status[res] = -1;
        g_notify_addr[res] = 0;
    }
    pthread_mutex_unlock(&cid_pool_lock);
    return res;
}

// 归还 CID：放到尾部 (idempotent)
void release_cid(uint16_t val)
{
    pthread_mutex_lock(&cid_pool_lock);
    if (g_cid_metadata[val].released) {
        pthread_mutex_unlock(&cid_pool_lock);
        return;  // already released, skip
    }
    g_cid_metadata[val].released = true;
    g_cid_metadata[val].count = 0;
    g_cid_metadata[val].ip_index = -1;
    g_cid_metadata[val].qp_index = -1;
    g_cid_metadata[val].tmp_conn = NULL;
    memset(g_cid_metadata[val].original_indices, 0, sizeof(g_cid_metadata[val].original_indices));
    memset(&g_cid_metadata[val].send_msg, 0, sizeof(data_block));
    memset(&g_cid_metadata[val].single_send_msg, 0, sizeof(data_block_single));
    memset(&g_cid_metadata[val].ctl_send_msg, 0, sizeof(ctl_block));
    if (cid_count < MAX_CID_COUNT) {
        cid_pool[cid_tail] = val;
        cid_tail = (cid_tail + 1) % MAX_CID_COUNT;  // 环形移动
        cid_count++;
    }
    pthread_mutex_unlock(&cid_pool_lock);
}

static struct ibv_device *find_ib_device_by_ip_once(const char *target_ip)
{
    struct ibv_device **dev_list = NULL;
    struct ibv_device *matched_dev = NULL;
    int num_devices = 0;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) { return NULL; }

    for (int i = 0; i < num_devices; i++) {
        struct ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx) { continue; }

        struct ibv_port_attr port_attr;
        if (ibv_query_port(ctx, 1, &port_attr) != 0) {
            ibv_close_device(ctx);
            continue;
        }

        for (int gid_idx = 0; gid_idx < port_attr.gid_tbl_len; gid_idx++) {
            union ibv_gid gid;
            if (ibv_query_gid(ctx, 1, gid_idx, &gid) != 0) { continue; }

            char gid_str[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &gid, gid_str, sizeof(gid_str));

            if (strstr(gid_str, target_ip) != NULL) {
                matched_dev = dev_list[i];
                ibv_close_device(ctx);
                goto cleanup;
            }
        }
        ibv_close_device(ctx);
    }

cleanup:
    ibv_free_device_list(dev_list);
    return matched_dev;
}

struct ibv_device *find_ib_device_by_ip(const char *target_ip)
{
    struct ibv_device *dev = find_ib_device_by_ip_once(target_ip);
    if (dev) {
        LOG_D("find_ib_device_by_ip succeeded for IP %s", target_ip);
        return dev;
    }
    fprintf(stderr, "No IB/RoCE device found for IP %s\n", target_ip);
    return NULL;
}

// GID转换函数（保持原实现）
void wire_gid_to_gid(const char *wgid, union ibv_gid *gid)
{
    char tmp[9];
    __be32 v32;
    int i;
    uint32_t tmp_gid[4];

    for (tmp[8] = 0, i = 0; i < 4; ++i) {
        memcpy(tmp, wgid + i * 8, 8);
        sscanf(tmp, "%x", &v32);
        tmp_gid[i] = be32toh(v32);
    }
    memcpy(gid, tmp_gid, sizeof(*gid));
}

void gid_to_wire_gid(const union ibv_gid *gid, char wgid[])
{
    uint32_t tmp_gid[4];
    int i;

    memcpy(tmp_gid, gid, sizeof(tmp_gid));
    for (i = 0; i < 4; ++i) sprintf(&wgid[i * 8], "%08x", htobe32(tmp_gid[i]));
}

NDS_ERROR_T setup_tcp_connection(struct rdma_connection *conn, const char *local_ip,
                                 const char *server_ip, int port)
{
    struct sockaddr_in local_addr, serv_addr;
    struct timeval timeout;

    conn->control_recv_len = 0;

    // 创建 Socket，添加 SOCK_CLOEXEC 属性
    conn->clnt_fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (conn->clnt_fd < 0) {
        LOG_E("Failed to create socket: %s", strerror(errno));
        return NDS_TCP_CONN_ERR;
    }

    // 设置REUSEADDR 和 REUSEPORT
    int opt_on = 1;
    setsockopt(conn->clnt_fd, SOL_SOCKET, SO_REUSEADDR, &opt_on, sizeof(opt_on));
    setsockopt(conn->clnt_fd, SOL_SOCKET, SO_REUSEPORT, &opt_on, sizeof(opt_on));

    int snd_buf_size = 1024 * 1024;
    int rcv_buf_size = 1024 * 1024;
    setsockopt(conn->clnt_fd, SOL_SOCKET, SO_SNDBUF, &snd_buf_size, sizeof(snd_buf_size));
    setsockopt(conn->clnt_fd, SOL_SOCKET, SO_RCVBUF, &rcv_buf_size, sizeof(rcv_buf_size));

    // 设置发送和接收超时时间，默认是5s
    timeout.tv_sec = TCP_TIMEOUT_SEC;
    timeout.tv_usec = 0;

    if (setsockopt(conn->clnt_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        LOG_E("setsockopt SO_SNDTIMEO failed: %s", strerror(errno));
        close(conn->clnt_fd);
        conn->clnt_fd = -1;
        return NDS_TCP_CONN_ERR;
    }

    if (setsockopt(conn->clnt_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        LOG_E("setsockopt SO_RCVTIMEO failed: %s", strerror(errno));
        close(conn->clnt_fd);
        conn->clnt_fd = -1;
        return NDS_TCP_CONN_ERR;
    }

    // 绑定本地指定的 IP
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_addr.s_addr = inet_addr(local_ip);
    local_addr.sin_port = htons(0);

    if (bind(conn->clnt_fd, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0) {
        perror("Bind failed");
        close(conn->clnt_fd);
        return 1;
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid address/ Address not supported \n");
        close(conn->clnt_fd);
        return NDS_TCP_CONN_ERR;
    }

    // 5. 发起连接：受 SO_SNDTIMEO 影响，如果网络不通，connect 可能会在超时时间到达后返回错误
    if (connect(conn->clnt_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        LOG_E("Failed to connect to %s:%d: %s", server_ip, port, strerror(errno));
        close(conn->clnt_fd);
        conn->clnt_fd = -1;
        return NDS_TCP_CONN_ERR;
    }

    return NDS_SUCCESS;
}

void print_hex(const char *label, const uint8_t *data, size_t len)
{
    printf("    %s: ", label);
    for (size_t i = 0; i < len; ++i) { printf("%02x ", data[i]); }
}

void print_xrt_rdma_neg_head(const xrt_rdma_neg_head_t *head)
{
    if (!head) return;

    printf("[ Message Header ]\n");
    printf("  CRC:     0x%08X\n", head->crc);
    printf("  Version: %u\n", head->ver);
    printf("  Command: %u (0x%02X)\n", head->cmd, head->cmd);
    printf("  Length:  %u\n", head->len);
}

void print_kv_cm_conn_cap(const struct kv_cm_conn_cap *cap)
{
    if (!cap) return;

    printf("    KV CM Connection Capabilities:\n");
    printf("        Version:            %u.%u\n", cap->major_version, cap->minor_version);
    printf("        Queue Num:          %u\n", cap->queue_num);
    printf("        Admin Queue Depth:  %u\n", cap->adminq_depth);
    printf("        IO Queue Depth:     %u\n", cap->ioq_depth);
    printf("        IOQ Key Batch:      %u\n", cap->ioq_key_batch);
    printf("        Total Key Batch:    %u\n", cap->total_key_batch);
    printf("        SR MDTS:            %u\n", cap->sr_mdts);
    printf("        BSR MDTS:           %u\n", cap->bsr_mdts);
    printf("        Store Key Batch:    %u\n", cap->store_key_batch);
    printf("        Retrieve Key Batch: %u\n", cap->retrieve_key_batch);
    printf("        Delete Key Batch:   %u\n", cap->delete_key_batch);
    printf("        Exist Key Batch:    %u\n", cap->exist_key_batch);
}

void print_auth_body(const xrt_rdma_neg_conn_auth_t *auth)
{
    printf("    Cap: %u\n", auth->cap);
    printf("    Private Len: %u\n", auth->private_len);
    print_kv_cm_conn_cap((struct kv_cm_conn_cap *)(auth->private_data));
}

// 打印 QP 属性 (REQ 和 RSP 通用)
void print_qp_attr(const xrt_rdma_neg_qp_attr_t *qp)
{
    // 打印 GID (16 字节)
    printf("    GID: ");
    for (int i = 0; i < 16; i++) printf("%02x%s", qp->gid.raw[i], (i == 15) ? "" : ":");

    printf("\n    LID: %u, MTU: %u, Total QP Num: %u\n", qp->lid, qp->mtu, qp->total_qp_num);
    printf("    SL: %u, Traffic Class: %u, RNR Timer: %u, RNR Retry: %u\n", qp->sl,
           qp->traffic_class, qp->rnr_timer, qp->rnr_retry_cnt);
    printf("    Timeout: %u, Retry Cnt: %u, RD Atom: %u\n", qp->timeout, qp->retry_cnt,
           qp->qp_rd_atom);
    printf("    Start PSN: %u\n", qp->start_psn);

    printf("    QPN List: ");
    for (int i = 0; i < qp->total_qp_num && i < XRT_MAX_NEG_QP_NUM; ++i) {
        printf("%u ", qp->qpn[i]);
    }
}

void xrt_rdma_neg_msg_dump(const xrt_rdma_neg_msg_t *msg, xrt_rdma_neg_cmd_t type)
{
    printf("\n========== RDMA NEG MSG DUMP ==========\n");
    print_xrt_rdma_neg_head(&msg->head);

    switch (type) {
        case XRT_RDMA_NEG_CONN_AUTH:
            printf("[ Body: CONN_AUTH ]\n");
            print_auth_body(&msg->body.conn_auth);
            break;
        case XRT_RDMA_NEG_CONN_REQ:
            printf("[ Body: CONN_REQ ]\n");
            print_qp_attr(&msg->body.conn_req.qp_attr);
            break;
        case XRT_RDMA_NEG_CONN_RSP:
            printf("[ Body: CONN_RSP ]\n");
            print_qp_attr(&msg->body.conn_rsp.qp_attr);
            break;
        default:
            printf("Unknown Message Type!\n");
            print_hex("Raw Body", msg->body.raw, XRT_MAX_NEG_BODY_SIZE);
            break;
    }
    printf("=======================================\n\n");
}

static uint32_t g_crctable[256] = {  // crc32表长度256
    0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
    0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
    0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
    0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
    0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
    0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
    0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
    0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
    0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
    0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
    0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
    0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
    0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
    0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
    0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
    0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
    0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
    0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
    0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
    0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
    0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
    0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
    0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
    0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
    0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
    0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
    0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
    0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
    0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
    0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
    0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
    0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351};

void xnet_crc32(const void *p, uint32_t len, uint32_t *p_crc)
{
    uint8_t *data = (uint8_t *)p;
    while (len-- > 0) {
        *p_crc = g_crctable[(*p_crc ^ *data++) & 0xFFL] ^ (*p_crc >> 8);  // crc值右移8位
    }
}

uint32_t compute_msg_crc(const xrt_rdma_neg_msg_t *req)
{
    if (req == nullptr) return 0;
    // uint32_t checksum = crc32(0L, Z_NULL, 0);

    // // 跳过首部 4 字节 (sizeof(uint32_t))，计算剩余部分
    // const unsigned char* data_start = (const unsigned char*)(req) + sizeof(uint32_t);
    // size_t data_len = sizeof(xrt_rdma_neg_msg_t) - sizeof(uint32_t);
    // checksum = crc32(checksum, data_start, data_len);

    uint32_t crc = 0;
    xnet_crc32(&req->head.crc + 1, sizeof(xrt_rdma_neg_head_t) - sizeof(req->head.crc), &crc);
    if (req->head.len != 0) { xnet_crc32(&req->body, ntohl(req->head.len), &crc); }

    printf("crc = 0x%x\n", crc);
    return htonl(crc);
}

// ---------------------------------------------------------
// 内部处理函数：处理 qp_attr 字节序
// ---------------------------------------------------------
static void _convert_qp_attr(xrt_rdma_neg_qp_attr_t *qp_attr, int to_network)
{
    if (to_network) {
        qp_attr->lid = htons(qp_attr->lid);
        qp_attr->start_psn = htonl(qp_attr->start_psn);
        for (int i = 0; i < XRT_MAX_NEG_QP_NUM; i++) { qp_attr->qpn[i] = htonl(qp_attr->qpn[i]); }
    } else {
        qp_attr->lid = ntohs(qp_attr->lid);
        qp_attr->start_psn = ntohl(qp_attr->start_psn);
        for (int i = 0; i < XRT_MAX_NEG_QP_NUM; i++) { qp_attr->qpn[i] = ntohl(qp_attr->qpn[i]); }
    }
    // 注意：gid (union) 通常作为字节流传输，不需要转换，uint8_t 字段不需要转换
}

// ---------------------------------------------------------
// 内部处理函数：处理 conn_auth 字节序
// ---------------------------------------------------------
static void _convert_conn_auth(xrt_rdma_neg_conn_auth_t *auth, int to_network)
{
    if (to_network) {
        auth->cap = htonl(auth->cap);
        auth->private_len = htonl(auth->private_len);
    } else {
        auth->cap = ntohl(auth->cap);
        auth->private_len = ntohl(auth->private_len);
    }
    // rsv 和 private_data 是字节数组，不需要转换
}

void xrt_rdma_neg_msg_h_to_n(xrt_rdma_neg_msg_t *msg)
{
    if (!msg) return;

    switch (msg->head.cmd) {
        case XRT_RDMA_NEG_CONN_AUTH: _convert_conn_auth(&msg->body.conn_auth, 1); break;
        case XRT_RDMA_NEG_CONN_REQ: _convert_qp_attr(&msg->body.conn_req.qp_attr, 1); break;
        case XRT_RDMA_NEG_CONN_RSP: _convert_qp_attr(&msg->body.conn_rsp.qp_attr, 1); break;
        default: break;
    }

    msg->head.crc = htonl(msg->head.crc);
    msg->head.len = htonl(msg->head.len);
    // ver, cmd, pad 是 uint8_t，不需要转换
}

void xrt_rdma_neg_msg_n_to_h(xrt_rdma_neg_msg_t *msg)
{
    if (!msg) return;

    // 注意：cmd 是 uint8_t，不需要转换即可直接在 switch 中读取
    msg->head.crc = ntohl(msg->head.crc);
    msg->head.len = ntohl(msg->head.len);

    switch (msg->head.cmd) {
        case XRT_RDMA_NEG_CONN_AUTH: _convert_conn_auth(&msg->body.conn_auth, 0); break;
        case XRT_RDMA_NEG_CONN_REQ: _convert_qp_attr(&msg->body.conn_req.qp_attr, 0); break;
        case XRT_RDMA_NEG_CONN_RSP: _convert_qp_attr(&msg->body.conn_rsp.qp_attr, 0); break;
        default: break;
    }
}

NDS_ERROR_T get_kv_cap_base_tcp(struct rdma_connection *conn)
{
    printf("get_kv_cap_base_tcp entry...\n");
    /* 1. 赋值 */
    memset(&conn->kv_cap_info, 0, sizeof(conn->kv_cap_info));
    conn->kv_cap_info.major_version = 1;
    conn->kv_cap_info.minor_version = 0;

    /* 2. 构造 */
    xrt_rdma_neg_msg_t req;
    memset(&req, 0, sizeof(req));

    req.head.ver = 1;
    req.head.cmd = XRT_RDMA_NEG_CONN_AUTH;
    req.head.len = sizeof(xrt_rdma_neg_conn_auth_t);

    /* 传入 kato 和 mtu */
    kv_cm_conn_cap_client_server kvData;
    memset(&kvData, 0, sizeof(kvData));
    kvData.major_version = conn->kv_cap_info.major_version;
    kvData.minor_version = conn->kv_cap_info.minor_version;
    kvData.kato = g_cfg.kato;

    req.body.conn_auth.cap = 0;
    req.body.conn_auth.private_len = sizeof(struct kv_cm_conn_cap_client_server);
    memcpy(req.body.conn_auth.private_data, &kvData, sizeof(struct kv_cm_conn_cap_client_server));

    xrt_rdma_neg_msg_dump(&req, XRT_RDMA_NEG_CONN_AUTH);

    /* 3. 发送 */
    struct iovec iov_send[1];
    xrt_rdma_neg_msg_h_to_n(&req);
    req.head.crc = compute_msg_crc(&req);

    iov_send[0].iov_base = &req;
    iov_send[0].iov_len = sizeof(xrt_rdma_neg_head_t) + sizeof(xrt_rdma_neg_conn_auth_t);

    struct msghdr msg_send;
    memset(&msg_send, 0, sizeof(msg_send));
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 1;

    ssize_t s_ret;
    if ((s_ret = sendmsg(conn->clnt_fd, &msg_send, MSG_NOSIGNAL)) < 0) {
        LOG_E("get_kv_cap_base_tcp sendmsg error");
        return NDS_GET_CAP_ERR;
    }
    printf("get_kv_cap_base_tcp sendmsg success!\n");

    /* 4. 接收 */
    xrt_rdma_neg_msg_t resp;
    struct iovec iov_recv[1];
    iov_recv[0].iov_base = &resp;
    iov_recv[0].iov_len = sizeof(resp);

    struct msghdr msg_recv;
    memset(&msg_recv, 0, sizeof(msg_recv));
    msg_recv.msg_iov = iov_recv;
    msg_recv.msg_iovlen = 1;

    while (1) {
        s_ret = recvmsg(conn->clnt_fd, &msg_recv, 0);

        if (s_ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { continue; }
            LOG_E("get_kv_cap_base_tcp recvmsg error");
            return NDS_GET_CAP_RES_ERR;
        } else if (s_ret == 0) {
            fprintf(stderr, "get_kv_cap_base_tcp connection closed by peer\n");
            return NDS_GET_CAP_RES_ERR;
        } else {
            break;
        }
    }

    xrt_rdma_neg_msg_n_to_h(&resp);
    memcpy(&(conn->kv_cap_info), resp.body.conn_auth.private_data, sizeof(struct kv_cm_conn_cap));
    xrt_rdma_neg_msg_dump(&resp, XRT_RDMA_NEG_CONN_AUTH);

    printf("get_kv_cap_base_tcp success!\n\n");
    return NDS_SUCCESS;
}

void adjust_max_batch_num(struct rdma_connection *conn)
{
    // 只支持MTU 1K/2K/4K，其他会建链失败
    if (conn->server_roce_info.mtu >= 3 && conn->server_roce_info.mtu <= 5) {
        uint32_t mtu_payload = 1024 * (1 << (conn->server_roce_info.mtu - 3));
        uint16_t mtu_key_batch = (mtu_payload - 64) / 36;
        // 发包最大batch_num = min(kv能力协商的batch_num, 协商后mtu计算的batch_num,
        // sdk配置文件batch_num)
        uint16_t min_batch_num =
            MIN3(mtu_key_batch, conn->kv_cap_info.store_key_batch, g_cfg.max_batch_num);
        g_cfg.max_batch_num = min_batch_num;
        printf("g_cfg.max_batch_num: %d\n", g_cfg.max_batch_num);
    }
}

NDS_ERROR_T get_roce_info_base_tcp(struct rdma_connection *conn, int qp_num)
{
    printf("get_roce_info_base_tcp entry...\n");
    /* 1. 赋值 */
    conn->client_roce_info.total_qp_num = qp_num;

    conn->client_roce_info.sl = g_cfg.priority;
    conn->client_roce_info.traffic_class = g_cfg.tos;
    conn->client_roce_info.rnr_timer = 12;
    conn->client_roce_info.rnr_retry_cnt = 7;
    conn->client_roce_info.timeout = 17;
    conn->client_roce_info.retry_cnt = 7;
    conn->client_roce_info.qp_rd_atom = 16;
    conn->client_roce_info.start_psn = 0;

    for (int i = 0; i < qp_num; i++) conn->client_roce_info.qpn[i] = conn->qps[i]->qp_num;

    /* 2. 构造 */
    xrt_rdma_neg_msg_t req;
    memset(&req, 0, sizeof(req));

    req.head.ver = 1;
    req.head.cmd = XRT_RDMA_NEG_CONN_REQ;
    req.head.len = sizeof(xrt_rdma_neg_conn_req_t);

    memcpy(&req.body.conn_req.qp_attr, &(conn->client_roce_info), sizeof(xrt_rdma_neg_conn_req_t));

    xrt_rdma_neg_msg_dump(&req, XRT_RDMA_NEG_CONN_REQ);

    /* 3. 发送 */
    struct iovec iov_send[1];
    xrt_rdma_neg_msg_h_to_n(&req);
    req.head.crc = compute_msg_crc(&req);

    iov_send[0].iov_base = &req;
    iov_send[0].iov_len = sizeof(xrt_rdma_neg_head_t) + sizeof(xrt_rdma_neg_conn_req_t);

    struct msghdr msg_send;
    memset(&msg_send, 0, sizeof(msg_send));
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 1;

    ssize_t s_ret;
    if ((s_ret = sendmsg(conn->clnt_fd, &msg_send, MSG_NOSIGNAL)) < 0) {
        LOG_E("get_roce_info_base_tcp sendmsg error");
        return NDS_GET_CAP_ERR;
    }
    printf("get_roce_info_base_tcp sendmsg success!\n");

    /* 4. 接收 */
    xrt_rdma_neg_msg_t resp;
    struct iovec iov_recv[1];
    iov_recv[0].iov_base = &resp;
    iov_recv[0].iov_len = sizeof(resp);

    struct msghdr msg_recv;
    memset(&msg_recv, 0, sizeof(msg_recv));
    msg_recv.msg_iov = iov_recv;
    msg_recv.msg_iovlen = 1;

    while (1) {
        s_ret = recvmsg(conn->clnt_fd, &msg_recv, 0);

        if (s_ret < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { continue; }
            LOG_E("get_roce_info_base_tcp recvmsg error");
            return NDS_GET_CAP_RES_ERR;
        } else if (s_ret == 0) {
            fprintf(stderr, "get_roce_info_base_tcp connection closed by peer\n");
            return NDS_GET_CAP_RES_ERR;
        } else {
            break;
        }
    }

    xrt_rdma_neg_msg_n_to_h(&resp);
    memcpy(&(conn->server_roce_info), &(resp.body.conn_rsp.qp_attr),
           sizeof(xrt_rdma_neg_conn_rsp_t));
    xrt_rdma_neg_msg_dump(&resp, XRT_RDMA_NEG_CONN_RSP);

    adjust_max_batch_num(conn);

    printf("get_roce_info_base_tcp success!\n\n");
    return NDS_SUCCESS;
}

NDS_ERROR_T get_done_info_base_tcp(struct rdma_connection *conn)
{
    printf("get_done_info_base_tcp entry...\n");
    /* 1. 构造 */
    xrt_rdma_neg_msg_t req;
    memset(&req, 0, sizeof(req));

    req.head.ver = 1;
    req.head.cmd = XRT_RDMA_NEG_CONN_DONE;
    req.head.len = 0;
    req.head.crc = compute_msg_crc(&req);

    /* 2. 发送 */
    struct iovec iov_send[1];
    iov_send[0].iov_base = &req;
    iov_send[0].iov_len = sizeof(xrt_rdma_neg_head_t);

    struct msghdr msg_send;
    memset(&msg_send, 0, sizeof(msg_send));
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 1;

    ssize_t s_ret;
    if ((s_ret = sendmsg(conn->clnt_fd, &msg_send, MSG_NOSIGNAL)) < 0) {
        LOG_E("get_done_info_base_tcp sendmsg error");
        return NDS_GET_CAP_ERR;
    }
    printf("get_done_info_base_tcp sendmsg success!\n");
    printf("get_done_info_base_tcp success!\n\n");
    return NDS_SUCCESS;

    // /* 3. 接收 */
    // xrt_rdma_neg_msg_t resp;
    // struct iovec iov_recv[1];
    // iov_recv[0].iov_base = &resp;
    // iov_recv[0].iov_len = sizeof(resp);

    // struct msghdr msg_recv;
    // memset(&msg_recv, 0, sizeof(msg_recv));
    // msg_recv.msg_iov = iov_recv;
    // msg_recv.msg_iovlen = 1;

    // while (1) {
    //     s_ret = recvmsg(conn->clnt_fd, &msg_recv, 0);

    //     if (s_ret < 0) {
    //         if (errno == EAGAIN || errno == EWOULDBLOCK) {
    //             continue;
    //         }
    //         die("get_done_info_base_tcp recvmsg error");
    //         return NDS_GET_CAP_RES_ERR;
    //     } else if (s_ret == 0) {
    //         fprintf(stderr, "get_done_info_base_tcp connection closed by peer\n");
    //         return NDS_GET_CAP_RES_ERR;
    //     } else {
    //         break;
    //     }
    // }
}

NDS_ERROR_T init_rdma_device_and_mr(struct rdma_connection *conn, const char *local_ip,
                                    void **kv_cache, int mr_num)
{
    struct ibv_device *devRoce = find_ib_device_by_ip(local_ip);
    if (devRoce) {
        LOG_I("Matched device: %s", devRoce->name);
    } else {
        LOG_E("No device found for IP %s", local_ip);
        return NDS_FAILED;
    }
    if (g_context == NULL)
        conn->context = ibv_open_device(devRoce);
    else
        conn->context = g_context;
    if (!conn->context) {
        LOG_E("Failed to open device");
        return NDS_FAILED;
    }

    conn->pd = ibv_alloc_pd(conn->context);
    if (!conn->pd) {
        LOG_E("Failed to allocate protection domain");
        return NDS_FAILED;
    }
    if (kv_cache == NULL || mr_num <= 0) {
        LOG_E("Failed to register mr with null kv cache");
        return NDS_FAILED;
    }

    struct kv_mr *mrs = (struct kv_mr *)(*kv_cache);
    for (int i = 0; i < mr_num; i++) {
        void *target_addr = (void *)mrs[i].addr;
        size_t target_len = (size_t)mrs[i].len;
        if (target_addr == NULL || target_len == 0) { continue; }
        if ((uintptr_t)target_addr % 4096 != 0) {
            printf("Index %d: Address %p is NOT page aligned!\n", i, target_addr);
        }
        conn->rw_buf[i] = target_addr;
        conn->rw_mr[i] =
            ibv_reg_mr(conn->pd, conn->rw_buf[i], target_len,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!conn->rw_mr[i]) {
            LOG_E("Failed to register mr at index %d", i);
            return NDS_FAILED;
        }
    }
    return NDS_SUCCESS;
}

NDS_ERROR_T setup_rdma_connection(struct rdma_connection *conn, int gid_index, int port_num)
{
    int ret;
    struct ibv_qp_attr qp_attr;
    int conn_idx = conn->conn_idx;
    bool is_reconnect = conn->is_reconnecting;

    if (is_reconnect) {
        for (int i = 0; i < QP_NUM_T; i++) {
            QPContext *ctx = &g_qp_ctx[conn_idx][i];
            if (!ctx->slot_used) continue;

            pthread_mutex_lock(&ctx->slot_lock);
            memset(ctx->slot_used, 0, g_cfg.max_slots * sizeof(bool));
            memset(ctx->ack_arrived, 0, g_cfg.max_slots * sizeof(bool));
            memset(ctx->resp_arrived, 0, g_cfg.max_slots * sizeof(bool));
            for (int s = 0; s < g_cfg.max_slots; s++) { ctx->slot_to_cid[s] = -1; }
            pthread_mutex_unlock(&ctx->slot_lock);

            ctx->inflight.store(0, std::memory_order_relaxed);

            int sem_val;
            sem_getvalue(&ctx->window_sem, &sem_val);
            int to_post = g_cfg.max_slots - sem_val;

            if (to_post > 0) {
                for (int k = 0; k < to_post; k++) { sem_post(&ctx->window_sem); }
            } else if (to_post < 0) {
                int to_wait = -to_post;
                for (int k = 0; k < to_wait; k++) { sem_wait(&ctx->window_sem); }
            }
        }
    }

    conn->qp_num = 0;

    if (get_kv_cap_base_tcp(conn) != NDS_SUCCESS) {
        LOG_E("get_kv_cap_base_tcp failed");
        return NDS_FAILED;
    }

    if (g_cfg.io_qp_num > (conn->kv_cap_info.queue_num - (conn->kv_cap_info.adminq_depth != 0))) {
        LOG_E("use_qp_num: %d > queue_num: %d", g_cfg.io_qp_num,
              conn->kv_cap_info.queue_num - (conn->kv_cap_info.adminq_depth != 0));
        return NDS_FAILED;
    }

    conn->gid_index = gid_index;
    conn->port_num = port_num;

    int startQp = 0;
    if (conn->kv_cap_info.adminq_depth != 0) {
        startQp = 1;
        conn->send_cq[0] =
            ibv_create_cq(conn->context, conn->kv_cap_info.adminq_depth, NULL, NULL, 0);
        if (!conn->send_cq[0]) {
            LOG_E("Failed to create admin tx completion queue");
            goto err_cleanup;
        }
        conn->recv_cq[0] =
            ibv_create_cq(conn->context, conn->kv_cap_info.adminq_depth, NULL, NULL, 0);
        if (!conn->recv_cq[0]) {
            LOG_E("Failed to create admin rx completion queue");
            ibv_destroy_cq(conn->send_cq[0]);
            conn->send_cq[0] = NULL;
            goto err_cleanup;
        }
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = conn->send_cq[0];
        qp_init_attr.recv_cq = conn->recv_cq[0];
        qp_init_attr.cap.max_send_wr = conn->kv_cap_info.adminq_depth;
        qp_init_attr.cap.max_recv_wr = conn->kv_cap_info.adminq_depth;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.qp_type = IBV_QPT_RC;

        conn->qps[0] = ibv_create_qp(conn->pd, &qp_init_attr);
        if (!conn->qps[0]) {
            LOG_E("Failed to create admin queue pair");
            ibv_destroy_cq(conn->recv_cq[0]);
            conn->recv_cq[0] = NULL;
            ibv_destroy_cq(conn->send_cq[0]);
            conn->send_cq[0] = NULL;
            goto err_cleanup;
        }
        conn->qp_num = 1;
    }

    for (int i = startQp; i < startQp + g_cfg.io_qp_num; i++) {
        conn->send_cq[i] = ibv_create_cq(conn->context, conn->kv_cap_info.ioq_depth, NULL, NULL, 0);
        if (!conn->send_cq[i]) {
            LOG_E("Failed to create tx completion queue for QP %d", i);
            goto err_cleanup;
        }
        conn->recv_cq[i] = ibv_create_cq(conn->context, conn->kv_cap_info.ioq_depth, NULL, NULL, 0);
        if (!conn->recv_cq[i]) {
            LOG_E("Failed to create rx completion queue for QP %d", i);
            ibv_destroy_cq(conn->send_cq[i]);
            conn->send_cq[i] = NULL;
            goto err_cleanup;
        }

        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = conn->send_cq[i];
        qp_init_attr.recv_cq = conn->recv_cq[i];
        qp_init_attr.cap.max_send_wr = conn->kv_cap_info.ioq_depth;
        qp_init_attr.cap.max_recv_wr = conn->kv_cap_info.ioq_depth;
        qp_init_attr.cap.max_send_sge = 4;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.qp_type = IBV_QPT_RC;

        conn->qps[i] = ibv_create_qp(conn->pd, &qp_init_attr);
        if (!conn->qps[i]) {
            LOG_E("Failed to create io queue pair %d", i);
            ibv_destroy_cq(conn->recv_cq[i]);
            conn->recv_cq[i] = NULL;
            ibv_destroy_cq(conn->send_cq[i]);
            conn->send_cq[i] = NULL;
            goto err_cleanup;
        }
        conn->qp_num = i + 1;
    }

    {
        struct ibv_port_attr port_attr;
        if (ibv_query_port(conn->context, conn->port_num, &port_attr)) {
            LOG_E("Failed to query port attributes");
            goto err_cleanup;
        }
        conn->client_roce_info.lid = port_attr.lid;
        conn->client_roce_info.mtu = (uint8_t)port_attr.active_mtu;
    }

    {
        int gid_retry_count = 0;
        const int GID_MAX_RETRIES = 50;
        const int GID_RETRY_DELAY_US = 100000;
        bool gid_ok = false;
        while (gid_retry_count < GID_MAX_RETRIES) {
            if (ibv_query_gid(conn->context, conn->port_num, conn->gid_index,
                              &(conn->client_roce_info.gid)) != 0) {
                LOG_E("ibv_query_gid failed (attempt %d/%d), retrying in %d ms",
                      gid_retry_count + 1, GID_MAX_RETRIES, GID_RETRY_DELAY_US / 1000);
                usleep(GID_RETRY_DELAY_US);
                gid_retry_count++;
                continue;
            }
            bool all_zero = true;
            for (int b = 0; b < 16; b++) {
                if (conn->client_roce_info.gid.raw[b] != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero) {
                gid_ok = true;
                break;
            }
            LOG_E("ibv_query_gid returned all-zero GID (attempt %d/%d), retrying in %d ms",
                  gid_retry_count + 1, GID_MAX_RETRIES, GID_RETRY_DELAY_US / 1000);
            usleep(GID_RETRY_DELAY_US);
            gid_retry_count++;
        }
        if (!gid_ok) {
            LOG_E("can't read valid sgid of index %d after %d retries", gid_index, GID_MAX_RETRIES);
            goto err_cleanup;
        }
        if (gid_retry_count > 0) {
            LOG_I("ibv_query_gid succeeded after %d retries", gid_retry_count);
        }
    }

    conn->ret_buf_size = 512;
    conn->ret_buf = malloc(conn->ret_buf_size);
    if (!conn->ret_buf) {
        LOG_E("Memory allocation ret_buf failed");
        goto err_cleanup;
    }
    memset(conn->ret_buf, 0, conn->ret_buf_size);

    conn->ret_mr =
        ibv_reg_mr(conn->pd, conn->ret_buf, conn->ret_buf_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!conn->ret_mr) {
        LOG_E("Failed to register ret_mr");
        free(conn->ret_buf);
        conn->ret_buf = NULL;
        goto err_cleanup;
    }

    for (int i = 0; i < conn->qp_num; i++) {
        conn->send_buf[i] = malloc(g_cfg.send_mr_size);
        if (!conn->send_buf[i]) {
            LOG_E("Failed to allocate send_buf for QP %d", i);
            goto err_cleanup_bufs;
        }

        conn->send_mr[i] =
            ibv_reg_mr(conn->pd, conn->send_buf[i], g_cfg.send_mr_size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!conn->send_mr[i]) {
            LOG_E("Failed to register send memory region for QP %d", i);
            free(conn->send_buf[i]);
            conn->send_buf[i] = NULL;
            goto err_cleanup_bufs;
        }

        conn->recv_buf[i] = malloc(RECV_MR_SIZE);
        if (!conn->recv_buf[i]) {
            LOG_E("Failed to allocate recv_buf for QP %d", i);
            goto err_cleanup_bufs;
        }

        conn->recv_mr[i] =
            ibv_reg_mr(conn->pd, conn->recv_buf[i], RECV_MR_SIZE,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!conn->recv_mr[i]) {
            LOG_E("Failed to register recv memory region for QP %d", i);
            free(conn->recv_buf[i]);
            conn->recv_buf[i] = NULL;
            goto err_cleanup_bufs;
        }
    }

    for (int i = 0; i < conn->qp_num; i++) {
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.pkey_index = 0;
        qp_attr.port_num = conn->port_num;
        qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

        ret = ibv_modify_qp(conn->qps[i], &qp_attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (ret) {
            LOG_E("Failed to modify QP %d to INIT state, ret=%d", i, ret);
            goto err_cleanup_bufs;
        }
    }

    if (get_roce_info_base_tcp(conn, conn->qp_num) != NDS_SUCCESS) {
        LOG_E("get_roce_info_base_tcp failed");
        goto err_cleanup_bufs;
    }

    for (int i = 0; i < conn->qp_num; i++) {
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = (enum ibv_mtu)(conn->server_roce_info.mtu);
        qp_attr.dest_qp_num = conn->server_roce_info.qpn[i];
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.grh.sgid_index = conn->gid_index;
        qp_attr.ah_attr.port_num = conn->port_num;
        qp_attr.ah_attr.grh.dgid = conn->server_roce_info.gid;
        qp_attr.ah_attr.dlid = conn->server_roce_info.lid;

        qp_attr.rq_psn = conn->server_roce_info.start_psn;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.max_dest_rd_atomic = conn->server_roce_info.qp_rd_atom;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.min_rnr_timer = conn->server_roce_info.rnr_timer;
        qp_attr.ah_attr.sl = conn->server_roce_info.sl;
        qp_attr.ah_attr.grh.traffic_class = conn->server_roce_info.traffic_class;

        ret = ibv_modify_qp(conn->qps[i], &qp_attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret) {
            LOG_E("Failed to modify QP %d to RTR state, ret=%d", i, ret);
            goto err_cleanup_bufs;
        }

        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.sq_psn = 0;
        qp_attr.max_rd_atomic = conn->server_roce_info.qp_rd_atom;
        qp_attr.rnr_retry = conn->client_roce_info.rnr_retry_cnt;
        qp_attr.retry_cnt = conn->client_roce_info.retry_cnt;
        qp_attr.timeout = conn->client_roce_info.timeout;

        ret = ibv_modify_qp(conn->qps[i], &qp_attr,
                            IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret) {
            LOG_E("Failed to modify QP %d to RTS state, ret=%d", i, ret);
            goto err_cleanup_bufs;
        }
    }

    for (int i = 0; i < conn->kv_cap_info.adminq_depth; i++) {
        struct ibv_sge recv_sge = {
            .addr = (uint64_t)((uint8_t *)conn->recv_buf[ADMIN_QP] + i * sizeof(recv_data_block)),
            .length = sizeof(recv_data_block),
            .lkey = conn->recv_mr[ADMIN_QP]->lkey,
        };

        struct ibv_recv_wr recv_wr = {};
        recv_wr.wr_id = (uint64_t)(recv_sge.addr);
        recv_wr.sg_list = &recv_sge;
        recv_wr.num_sge = 1;

        recv_data_block *data =
            (recv_data_block *)((uint8_t *)conn->recv_buf[ADMIN_QP] + i * sizeof(recv_data_block));
        for (int k = 0; k < 16; k++) { data->res_buf[k] = 0; }

        struct ibv_recv_wr *bad_recv_wr;
        ret = ibv_post_recv(conn->qps[ADMIN_QP], &recv_wr, &bad_recv_wr);
        __sync_fetch_and_add(&conn->connRecvCounter, 1);
        if (ret) {
            LOG_E("ibv_post_recv failed for admin QP, prepost %d, ret=%d", i, ret);
            goto err_cleanup_bufs;
        }
    }

    for (int i = startQp; i < conn->qp_num; i++) {
        for (int j = 0; j < conn->kv_cap_info.ioq_depth; j++) {
            struct ibv_sge recv_sge = {
                .addr = (uint64_t)((uint8_t *)conn->recv_buf[i] + j * sizeof(recv_data_block)),
                .length = sizeof(recv_data_block),
                .lkey = conn->recv_mr[i]->lkey,
            };

            struct ibv_recv_wr recv_wr = {};
            recv_wr.wr_id = (uint64_t)(recv_sge.addr);
            recv_wr.sg_list = &recv_sge;
            recv_wr.num_sge = 1;

            recv_data_block *data =
                (recv_data_block *)((uint8_t *)conn->recv_buf[i] + j * sizeof(recv_data_block));
            for (int k = 0; k < 16; k++) { data->res_buf[k] = 0; }

            struct ibv_recv_wr *bad_recv_wr;
            ret = ibv_post_recv(conn->qps[i], &recv_wr, &bad_recv_wr);
            __sync_fetch_and_add(&conn->connRecvCounter, 1);
            if (ret) {
                LOG_E("ibv_post_recv failed for QP %d, prepost %d, ret=%d", i, j, ret);
                goto err_cleanup_bufs;
            }
        }
    }

    if (get_done_info_base_tcp(conn) != NDS_SUCCESS) {
        LOG_E("get_done_info_base_tcp failed");
        goto err_cleanup_bufs;
    }

    return NDS_SUCCESS;

err_cleanup_bufs:
    for (int i = 0; i < conn->qp_num; i++) {
        if (conn->recv_mr[i]) {
            ibv_dereg_mr(conn->recv_mr[i]);
            conn->recv_mr[i] = NULL;
        }
        if (conn->recv_buf[i]) {
            free(conn->recv_buf[i]);
            conn->recv_buf[i] = NULL;
        }
        if (conn->send_mr[i]) {
            ibv_dereg_mr(conn->send_mr[i]);
            conn->send_mr[i] = NULL;
        }
        if (conn->send_buf[i]) {
            free(conn->send_buf[i]);
            conn->send_buf[i] = NULL;
        }
    }
    if (conn->ret_mr) {
        ibv_dereg_mr(conn->ret_mr);
        conn->ret_mr = NULL;
    }
    if (conn->ret_buf) {
        free(conn->ret_buf);
        conn->ret_buf = NULL;
    }

err_cleanup:
    for (int i = 0; i < conn->qp_num; i++) {
        if (conn->qps[i]) {
            ibv_destroy_qp(conn->qps[i]);
            conn->qps[i] = NULL;
        }
        if (conn->recv_cq[i]) {
            ibv_destroy_cq(conn->recv_cq[i]);
            conn->recv_cq[i] = NULL;
        }
        if (conn->send_cq[i]) {
            ibv_destroy_cq(conn->send_cq[i]);
            conn->send_cq[i] = NULL;
        }
    }
    conn->qp_num = 0;
    return NDS_FAILED;
}

void *unified_poll_thread(void *arg)
{
    thread_args_t *para = (thread_args_t *)arg;
    int start_idx = para->start_thread;
    int step = para->step;
    free(para);
    struct ibv_wc wcs[CQE_BATCH];
    struct ibv_wc wc;

    while (!exit_flag) {
        bool busy = false;
        for (int idx = start_idx; idx < IP_NUM_T; idx += step) {
            if (!conn[idx].online || conn[idx].is_destroying || conn[idx].is_reconnecting) continue;
            for (int j = 0; j < conn[idx].qp_num; j++) {
                struct ibv_cq *recv_cq = conn[idx].recv_cq[j];
                QPContext &ctx = g_qp_ctx[idx][j];
                int slot_id;
                if (recv_cq == NULL) continue;
                int ne_recv = ibv_poll_cq(recv_cq, CQE_BATCH, wcs);
                if (ne_recv > 0) {
                    busy = true;
                    for (int i = 0; i < ne_recv; i++) {
                        recv_data_block *data;
                        __sync_fetch_and_sub(&conn[idx].connRecvCounter, 1);
                        LOG_D("conn[%d].connRecvCounter: %ld\n", idx, conn[idx].connRecvCounter);
                        wc = wcs[i];
                        if (wc.status != IBV_WC_SUCCESS) {
                            LOG_D(
                                "Work completion error: status(%u), description(%s) conn(%d) "
                                "qp(%d)",
                                wc.status, ibv_wc_status_str(wc.status), idx, j);
                            goto replenish_cqe;
                        }
                        task_done = 1;
                        data = (recv_data_block *)wc.wr_id;
                        LOG_D("get recv package cid(%u), status(%u)", data->cid, data->status);

                        g_notify_addr[data->cid] = data->result;

                        if ((data->status != 0x732) && (data->status != 0)) {
                            for (int i = 0; i < 16; i++) { data->res_buf[i] = 0xffffffff; }
                        }

                        for (int group = 0; group < g_cid_metadata[data->cid].count; group++) {
                            int word_idx = group / 8;  // 哪个 uint32_t（每个 word 有 8 个 4bit 组）
                            int nibble_idx = group % 8;  // 第几个 4bit 组（0~7）

                            uint32_t word = data->res_buf[word_idx];
                            int code = (word >> (nibble_idx * 4)) & 0xF;  // 掩码 0xF 取出 4bit
                            g_hitmap[data->cid][group] = code;
                        }

                        for (int i = 0; i < 16; i++) { data->res_buf[i] = 0; }

                        if (g_cid_metadata[data->cid].optype != HEARTBEAT) {
                            __sync_fetch_and_add(&conn[idx].io_count, 1);
                        }

                        slot_id = cid_to_slot[data->cid];
                        send_back_status[data->cid] = data->status;

                        if (g_cid_metadata[data->cid].optype == HEARTBEAT) {
                            if (slot_id >= 0 && slot_id < g_cfg.max_slots) {
                                pthread_mutex_lock(&ctx.slot_lock);
                                ctx.slot_used[slot_id] = false;
                                ctx.ack_arrived[slot_id] = false;
                                ctx.resp_arrived[slot_id] = false;
                                ctx.slot_to_cid[slot_id] = -1;
                                pthread_mutex_unlock(&ctx.slot_lock);
                                sem_post(&ctx.window_sem);
                                cid_to_slot[data->cid] = -1;
                            }
                            g_cid_metadata[data->cid].ip_index = -1;
                            send_back_status[data->cid] = -1;
                            g_notify_addr[data->cid] = 0;
                            release_cid(data->cid);
                            goto replenish_cqe;
                        }

                        if (slot_id < 0 || slot_id >= g_cfg.max_slots) goto replenish_cqe;

                        pthread_mutex_lock(&ctx.slot_lock);

                        if (!ctx.slot_used[slot_id]) {
                            cid_to_slot[data->cid] = -1;
                            pthread_mutex_unlock(&ctx.slot_lock);
                            goto replenish_cqe;
                        }

                        ctx.resp_arrived[slot_id] = true;
                        if (ctx.ack_arrived[slot_id]) {
                            ctx.slot_used[slot_id] = false;
                            ctx.ack_arrived[slot_id] = false;
                            ctx.resp_arrived[slot_id] = false;
                            ctx.slot_to_cid[slot_id] = -1;
                            cid_to_slot[data->cid] = -1;
                            pthread_mutex_unlock(&ctx.slot_lock);
                            sem_post(&ctx.window_sem);
                        } else {
                            pthread_mutex_unlock(&ctx.slot_lock);
                        }
                    replenish_cqe:
                        // 补充CQE
                        struct ibv_qp *qp = conn[idx].qps[j];
                        struct ibv_mr *recv_mr = conn[idx].recv_mr[j];
                        if (qp == NULL || recv_mr == NULL) continue;
                        struct ibv_sge recv_sge = {
                            .addr = (uint64_t)(wc.wr_id),
                            .length = sizeof(recv_data_block),
                            .lkey = recv_mr->lkey,
                        };
                        struct ibv_recv_wr recv_wr = {};
                        recv_wr.wr_id = (uint64_t)(recv_sge.addr);
                        recv_wr.sg_list = &recv_sge;
                        recv_wr.num_sge = 1;

                        struct ibv_recv_wr *bad_recv_wr;
                        int ret = ibv_post_recv(qp, &recv_wr, &bad_recv_wr);
                        if (ret) {
                            LOG_E("ibv_post_recv failed: ret=%d conn %d qp %d", ret, idx, j);
                        } else {
                            __sync_fetch_and_add(&conn[idx].connRecvCounter, 1);
                            LOG_D(" add conn[%d].connRecvCounter: %ld\n", idx,
                                  conn[idx].connRecvCounter);
                        }
                    }
                }

                struct ibv_cq *send_cq = conn[idx].send_cq[j];
                if (send_cq == NULL) continue;
                int ne_send = ibv_poll_cq(send_cq, CQE_BATCH, wcs);
                if (ne_send > 0) {
                    busy = true;
                    // 收到 ack, 此链接的connSendCounter原子减 1
                    __sync_fetch_and_sub(&conn[idx].connSendCounter, ne_send);
                    LOG_D("conn[%d].connSendCounter: %ld\n", idx, conn[idx].connSendCounter);
                    for (int x = 0; x < ne_send; x++) {
                        if (wcs[x].status != IBV_WC_SUCCESS) {
                            LOG_D("WC failed: %s (conn idx %d id: %lu)",
                                  ibv_wc_status_str(wcs[x].status), idx, wcs[x].wr_id);
                            // poll 到网卡返的错误 ack，不去做断链，也不用唤醒，断链时会置错误码
                            // 因为 req 超过10秒，会触发断链，同一在wait函数中处理
                            int failed_slot = (int)wcs[x].wr_id;
                            if (failed_slot >= 0 && failed_slot < g_cfg.max_slots) {
                                pthread_mutex_lock(&ctx.slot_lock);
                                if (ctx.slot_used[failed_slot]) {
                                    ctx.slot_used[failed_slot] = false;
                                    int cid = ctx.slot_to_cid[failed_slot];
                                    ctx.slot_to_cid[failed_slot] = -1;
                                    if (cid >= 0) cid_to_slot[cid] = -1;
                                    pthread_mutex_unlock(&ctx.slot_lock);
                                    sem_post(&ctx.window_sem);
                                } else {
                                    pthread_mutex_unlock(&ctx.slot_lock);
                                }
                            }
                            continue;  // 处理剩余 CQE
                        }

                        int completed_slot_id = (int)wcs[x].wr_id;
                        LOG_D("poll ack : conn %d, slot %d", idx, completed_slot_id);

                        QPContext &ctx = g_qp_ctx[idx][j];
                        pthread_mutex_lock(&ctx.slot_lock);

                        if (!ctx.slot_used[completed_slot_id]) {
                            pthread_mutex_unlock(&ctx.slot_lock);
                            continue;
                        }

                        ctx.ack_arrived[completed_slot_id] = true;

                        if (ctx.resp_arrived[completed_slot_id]) {
                            int completed_cid = ctx.slot_to_cid[completed_slot_id];
                            ctx.slot_used[completed_slot_id] = false;
                            ctx.ack_arrived[completed_slot_id] = false;
                            ctx.resp_arrived[completed_slot_id] = false;
                            ctx.slot_to_cid[completed_slot_id] = -1;
                            pthread_mutex_unlock(&ctx.slot_lock);
                            if (completed_cid >= 0) { cid_to_slot[completed_cid] = -1; }
                            sem_post(&ctx.window_sem);
                        } else {
                            pthread_mutex_unlock(&ctx.slot_lock);
                        }
                    }
                }
            }
            if (!busy) { sched_yield(); }
        }
    }
    return NULL;
}

NDS_ERROR_T transfer_data_send(int qpID, int ipIdx, struct rdma_connection *conn,
                               const void *message, uint16_t batchNum)
{
    task_done = 0;
    QPContext &ctx = g_qp_ctx[ipIdx][qpID];
    data_block *msg = (data_block *)message;
    sem_wait(&ctx.window_sem);

    if (!conn->online || conn->is_destroying || conn->is_reconnecting || !conn->is_used) {
        LOG_D("Conn %d QP %d not usable (online=%d destroying=%d reconnecting=%d), skip send",
              ipIdx, qpID, conn->online.load(), conn->is_destroying.load(),
              conn->is_reconnecting.load());
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    if (conn->send_mr[qpID] == NULL || conn->qps[qpID] == NULL) {
        LOG_D("Conn %d QP %d resources freed after sem_wait, skip", ipIdx, qpID);
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    // 锁定并寻找空闲 Slot 索引 (0-31)
    int slot_id = -1;
    pthread_mutex_lock(&ctx.slot_lock);
    for (int i = 0; i < g_cfg.max_slots; i++) {
        if (!ctx.slot_used[i]) {
            ctx.slot_used[i] = true;
            ctx.ack_arrived[i] = false;
            ctx.resp_arrived[i] = false;
            slot_id = i;
            break;
        }
    }
    LOG_D("ipIdx %d cid %d qp id %d, get slot id %d, slot size (%d), total send_buf size (%zu)",
          qpID, msg->cid, qpID, slot_id, g_cfg.slot_size, g_cfg.send_mr_size);
    pthread_mutex_unlock(&ctx.slot_lock);

    if (slot_id == -1) {
        sem_post(&ctx.window_sem);  // 回滚
        return NDS_TCP_CONN_ERR;    // error
    }

    cid_to_slot[msg->cid] = slot_id;
    ctx.slot_to_cid[slot_id] = msg->cid;

    size_t paylen = sizeof(data_block);
    paylen = 64 + sizeof(addr_pair_storge) * batchNum;
    if (paylen > conn->send_mr[qpID]->length) paylen = conn->send_mr[qpID]->length;

    // // 计算当前线程该使用的内存偏移
    uint8_t *dest_ptr = (uint8_t *)conn->send_buf[qpID] + (slot_id * g_cfg.slot_size);
    // // 拷贝 8KB 数据到对应的 Slot
    memcpy(dest_ptr, message, paylen);

    // struct ibv_sge sge = {
    //     .addr = (uint64_t)conn->send_mr[qpID]->addr, .length = paylen, .lkey =
    //     conn->send_mr[qpID]->lkey};
    struct ibv_sge sge = {
        .addr = (uint64_t)((uint8_t *)conn->send_mr[qpID]->addr + (slot_id * g_cfg.slot_size)),
        .length = (uint32_t)paylen,
        .lkey = conn->send_mr[qpID]->lkey};

    struct ibv_send_wr send_wr = {};
    send_wr.wr_id = (uint64_t)slot_id;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr = NULL;
    if (ibv_post_send(conn->qps[qpID], &send_wr, &bad_wr)) {
        pthread_mutex_lock(&ctx.slot_lock);
        ctx.slot_used[slot_id] = false;
        pthread_mutex_unlock(&ctx.slot_lock);
        sem_post(&g_qp_ctx[ipIdx][qpID].window_sem);
        LOG_E("Failed to post send on conn %d qp %d", ipIdx, qpID);
        return NDS_TCP_CONN_ERR;
    }
    // printf("send succeed \n");
    return NDS_SUCCESS;
}

NDS_ERROR_T transfer_data_send_single(int qpID, int ipIdx, struct rdma_connection *conn,
                                      const void *message)
{
    task_done = 0;
    QPContext &ctx = g_qp_ctx[ipIdx][qpID];
    data_block_single *msg = (data_block_single *)message;
    sem_wait(&ctx.window_sem);

    if (!conn->online || conn->is_destroying || conn->is_reconnecting || !conn->is_used) {
        LOG_D(
            "Conn %d QP %d not usable (online=%d destroying=%d reconnecting=%d), skip send "
            "(single)",
            ipIdx, qpID, conn->online.load(), conn->is_destroying.load(),
            conn->is_reconnecting.load());
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    if (conn->send_mr[qpID] == NULL || conn->qps[qpID] == NULL) {
        LOG_D("Conn %d QP %d resources freed after sem_wait (single), skip", ipIdx, qpID);
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    int slot_id = -1;
    pthread_mutex_lock(&ctx.slot_lock);
    for (int i = 0; i < g_cfg.max_slots; i++) {
        if (!ctx.slot_used[i]) {
            ctx.slot_used[i] = true;
            ctx.ack_arrived[i] = false;
            ctx.resp_arrived[i] = false;
            slot_id = i;
            break;
        }
    }
    LOG_D("ipIdx %d cid %d qp id %d, get slot id %d, slot size (%d), total send_buf size (%zu)",
          qpID, msg->cid, qpID, slot_id, g_cfg.slot_size, g_cfg.send_mr_size);
    pthread_mutex_unlock(&ctx.slot_lock);

    if (slot_id == -1) {
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    cid_to_slot[msg->cid] = slot_id;
    ctx.slot_to_cid[slot_id] = msg->cid;

    size_t paylen = sizeof(data_block_single);
    if (paylen > conn->send_mr[qpID]->length) paylen = conn->send_mr[qpID]->length;

    // // 计算当前线程该使用的内存偏移
    uint8_t *dest_ptr = (uint8_t *)conn->send_buf[qpID] + (slot_id * g_cfg.slot_size);
    // // 拷贝 8KB 数据到对应的 Slot
    memcpy(dest_ptr, message, paylen);

    // struct ibv_sge sge = {
    //     .addr = (uint64_t)conn->send_mr[qpID]->addr, .length = paylen, .lkey =
    //     conn->send_mr[qpID]->lkey};
    struct ibv_sge sge = {
        .addr = (uint64_t)((uint8_t *)conn->send_mr[qpID]->addr + (slot_id * g_cfg.slot_size)),
        .length = (uint32_t)paylen,
        .lkey = conn->send_mr[qpID]->lkey};

    struct ibv_send_wr send_wr = {};
    send_wr.wr_id = (uint64_t)slot_id;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr = NULL;
    if (ibv_post_send(conn->qps[qpID], &send_wr, &bad_wr)) {
        pthread_mutex_lock(&ctx.slot_lock);
        ctx.slot_used[slot_id] = false;
        pthread_mutex_unlock(&ctx.slot_lock);
        sem_post(&g_qp_ctx[ipIdx][qpID].window_sem);
        LOG_E("Failed to post send (transfer_data_send) conn %d qp %d", ipIdx, qpID);
        return NDS_TCP_CONN_ERR;
    }
    return NDS_SUCCESS;
}

NDS_ERROR_T transfer_ctl_send(int qpID, int ipIdx, struct rdma_connection *conn,
                              const void *message, uint16_t batchNum)
{
    task_done = 0;
    QPContext &ctx = g_qp_ctx[ipIdx][qpID];
    ctl_block *msg = (ctl_block *)message;
    sem_wait(&ctx.window_sem);

    if (!conn->online || conn->is_destroying || conn->is_reconnecting || !conn->is_used) {
        LOG_D("Conn %d QP %d not usable (online=%d destroying=%d reconnecting=%d), skip send (ctl)",
              ipIdx, qpID, conn->online.load(), conn->is_destroying.load(),
              conn->is_reconnecting.load());
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    if (conn->send_mr[qpID] == NULL || conn->qps[qpID] == NULL) {
        LOG_E("Conn %d QP %d resources freed after sem_wait (ctl), skip", ipIdx, qpID);
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    int slot_id = -1;
    pthread_mutex_lock(&ctx.slot_lock);
    for (int i = 0; i < g_cfg.max_slots; i++) {
        if (!ctx.slot_used[i]) {
            ctx.slot_used[i] = true;
            ctx.ack_arrived[i] = false;
            ctx.resp_arrived[i] = false;
            slot_id = i;
            break;
        }
    }
    LOG_D("cid %d qp id %d, get slot id %d, slot size (%d), total send_buf size (%zu)", msg->cid,
          qpID, slot_id, g_cfg.slot_size, g_cfg.send_mr_size);
    pthread_mutex_unlock(&ctx.slot_lock);
    if (slot_id == -1) {
        sem_post(&ctx.window_sem);
        return 1;
    }

    cid_to_slot[msg->cid] = slot_id;
    ctx.slot_to_cid[slot_id] = msg->cid;

    size_t paylen = sizeof(ctl_block);
    paylen = 64 + sizeof(block_key) * batchNum;
    if (paylen > conn->send_mr[qpID]->length) paylen = conn->send_mr[qpID]->length;
    // // 计算当前线程该使用的内存偏移
    uint8_t *dest_ptr = (uint8_t *)conn->send_buf[qpID] + (slot_id * g_cfg.slot_size);
    // // 拷贝 数据到对应的 Slot
    memcpy(dest_ptr, message, paylen);

    // struct ibv_sge sge = {
    //     .addr = (uint64_t)conn->send_mr[qpID]->addr, .length = paylen, .lkey =
    //     conn->send_mr[qpID]->lkey};
    struct ibv_sge sge = {
        .addr = (uint64_t)((uint8_t *)conn->send_mr[qpID]->addr + (slot_id * g_cfg.slot_size)),
        .length = (uint32_t)paylen,
        .lkey = conn->send_mr[qpID]->lkey};

    struct ibv_send_wr send_wr = {};
    send_wr.wr_id = (uint64_t)slot_id;
    send_wr.sg_list = &sge;
    send_wr.num_sge = 1;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr = NULL;
    if (ibv_post_send(conn->qps[qpID], &send_wr, &bad_wr)) {
        pthread_mutex_lock(&ctx.slot_lock);
        ctx.slot_used[slot_id] = false;
        pthread_mutex_unlock(&ctx.slot_lock);
        sem_post(&g_qp_ctx[ipIdx][qpID].window_sem);
        LOG_E("Failed to post send (transfer_ctl_send) conn %d qp %d", ipIdx, qpID);
        return NDS_TCP_CONN_ERR;
    }
    return NDS_SUCCESS;
}

// 1. 扫描所有 CID，标记属于该连接的 CID 及其占用的 slot
static void move_qps_to_err(struct rdma_connection *conn)
{
    for (int i = 0; i < conn->qp_num; i++) {
        int ipIdx = conn->conn_idx;
        QPContext &ctx = g_qp_ctx[ipIdx][i];
        if (conn->qps[i]) {
            struct ibv_qp_attr qp_attr;
            memset(&qp_attr, 0, sizeof(qp_attr));
            qp_attr.qp_state = IBV_QPS_ERR;
            int ret = ibv_modify_qp(conn->qps[i], &qp_attr, IBV_QP_STATE);
            if (ret) {
                LOG_W("[DESTROY_DEBUG] conn %d qp %d: failed to move to ERR state (ret=%d)", ipIdx,
                      i, ret);
                int sem_val;
                sem_getvalue(&ctx.window_sem, &sem_val);
                int to_post = g_cfg.max_slots - sem_val;
                if (to_post > 0) {
                    for (int k = 0; k < to_post; k++) { sem_post(&ctx.window_sem); }
                }
            }
        }
    }
}

// 第二步：轮询send_cq和recv_cq直到排空，释放对应的slot和窗口信号量
static void drain_cqs(struct rdma_connection *conn)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    while (conn->connSendCounter != 0 || conn->connRecvCounter != 0) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        if (elapsed.count() > g_io_request_timeout) {
            LOG_W("Drain_cqs timeout");
            break;
        }
        for (int i = 0; i < conn->qp_num; i++) {
            int ipIdx = conn->conn_idx;
            QPContext &ctx = g_qp_ctx[ipIdx][i];
            int sem_val_before;
            sem_getvalue(&ctx.window_sem, &sem_val_before);
            if (conn->qps[i]) {
                LOG_D("[DESTROY_DEBUG] conn %d qp %d: moved to ERR state for flush", ipIdx, i);
                usleep(1000);
                if (conn->send_cq[i]) {
                    struct ibv_wc wc[16];
                    int flushed = 0;
                    int slots_released = 0;
                    int ncqe = ibv_poll_cq(conn->send_cq[i], 16, wc);
                    if (ncqe <= 0) goto poll_recvcq;
                    flushed += ncqe;
                    __sync_fetch_and_sub(&conn->connSendCounter, ncqe);
                    LOG_D("destroy conn->connSendCounter: %ld\\n", conn->connSendCounter);
                    for (int k = 0; k < ncqe; k++) {
                        int slot_id = (int)wc[k].wr_id;
                        if (slot_id >= 0 && slot_id < g_cfg.max_slots) {
                            pthread_mutex_lock(&ctx.slot_lock);
                            bool was_used = ctx.slot_used[slot_id];
                            ctx.slot_used[slot_id] = false;
                            int cid = ctx.slot_to_cid[slot_id];
                            ctx.slot_to_cid[slot_id] = -1;
                            if (cid >= 0 && cid < MAX_CID_COUNT) { cid_to_slot[cid] = -1; }
                            pthread_mutex_unlock(&ctx.slot_lock);

                            if (was_used) {
                                sem_post(&ctx.window_sem);
                                slots_released++;
                            }

                            if (cid >= 0 && cid < MAX_CID_COUNT && !g_cid_metadata[cid].released) {
                                send_back_status[cid] = IPBECLEANED;
                                g_notify_addr[cid] = 0;
                                LOG_D(
                                    "[DESTROY_FLUSH] conn %d qp %d slot %d cid %d: set TIMEOUT "
                                    "status",
                                    ipIdx, i, slot_id, cid);
                            }
                        }
                    }
                    if (flushed > 0) {
                        LOG_D(
                            "[DESTROY_DEBUG] conn %d qp %d: flushed %d WQEs, released %d valid "
                            "slots (sem was %d)",
                            ipIdx, i, flushed, slots_released, sem_val_before);
                    }
                    int sem_val_after;
                    sem_getvalue(&ctx.window_sem, &sem_val_after);
                    int additional_post = g_cfg.max_slots - sem_val_after;
                    if (additional_post > 0) {
                        for (int k = 0; k < additional_post; k++) { sem_post(&ctx.window_sem); }
                        LOG_D(
                            "[DESTROY_DEBUG] conn %d qp %d: posted %d additional to reach target "
                            "%d (now %d)",
                            ipIdx, i, additional_post, g_cfg.max_slots, sem_val_after);
                    }
                }
            poll_recvcq:
                if (conn->recv_cq[i]) {
                    struct ibv_wc wc[16];
                    int flushed = 0;
                    int ncqe = ibv_poll_cq(conn->recv_cq[i], 16, wc);
                    if (ncqe <= 0) continue;
                    flushed += ncqe;
                    __sync_fetch_and_sub(&conn->connRecvCounter, ncqe);
                    LOG_D("destroy conn->connRecvCounter: %ld\\n", conn->connRecvCounter);
                    if (flushed > 0) {
                        LOG_D("[DESTROY_DEBUG] conn %d qp %d: flushed %d WQEs from recv_cq", ipIdx,
                              i, flushed);
                    }
                }
            } else {
                int sem_val;
                sem_getvalue(&ctx.window_sem, &sem_val);
                int to_post = g_cfg.max_slots - sem_val;
                if (to_post > 0) {
                    for (int k = 0; k < to_post; k++) { sem_post(&ctx.window_sem); }
                }
            }
        }
    }
}

// 第三步：销毁send_mr/recv_mr、send_buf/recv_buf、QP、CQ
static void destroy_qp_resources(struct rdma_connection *conn)
{
    for (int i = 0; i < conn->qp_num; i++) {
        if (conn->send_mr[i]) {
            ibv_dereg_mr(conn->send_mr[i]);
            conn->send_mr[i] = NULL;
        }
        if (conn->recv_mr[i]) {
            ibv_dereg_mr(conn->recv_mr[i]);
            conn->recv_mr[i] = NULL;
        }
        if (conn->send_buf[i]) {
            free(conn->send_buf[i]);
            conn->send_buf[i] = NULL;
        }
        if (conn->recv_buf[i]) {
            free(conn->recv_buf[i]);
            conn->recv_buf[i] = NULL;
        }
        if (conn->qps[i]) {
            ibv_destroy_qp(conn->qps[i]);
            conn->qps[i] = NULL;
        }
        if (conn->send_cq[i]) {
            ibv_destroy_cq(conn->send_cq[i]);
            conn->send_cq[i] = NULL;
        }
        if (conn->recv_cq[i]) {
            ibv_destroy_cq(conn->recv_cq[i]);
            conn->recv_cq[i] = NULL;
        }
    }
}

// 第四步：注销rw_mr、销毁PD、关设备、关socket
static void dereg_mr_pd_device(struct rdma_connection *conn)
{
    for (int i = 0; i < SLAB_NUM; i++) {
        if (i != SLAB_NUM - 1) {
            if (conn->rw_mr[i]) {
                ibv_dereg_mr(conn->rw_mr[i]);
                conn->rw_mr[i] = NULL;
            }
        }
        conn->rw_buf[i] = NULL;
    }

    if (conn->pd) {
        ibv_dealloc_pd(conn->pd);
        conn->pd = NULL;
    }
    if (conn->context) {
        if (conn->context != g_context) { ibv_close_device(conn->context); }
        conn->context = NULL;
    }
    if (conn->clnt_fd > 0) {
        close(conn->clnt_fd);
        conn->clnt_fd = -1;
    }
}

// 销毁连接资源（不含锁操作，调用方需持有 destroy_connection_lock 或确保无竞争）
void destroy_connection_locked(struct rdma_connection *conn)
{
    __sync_synchronize();
    usleep(1000);
    epoll_remove_fd(g_epoll_args.epfd, conn->clnt_fd);
    move_qps_to_err(conn);
    drain_cqs(conn);
    destroy_qp_resources(conn);
    dereg_mr_pd_device(conn);
    cleanup_conn_cids(conn->conn_idx);
    LOG_I("Destroy connection %d success: local=%s -> dpu=%s", conn->conn_idx, conn->ip.local_ip,
          conn->ip.server_ip);
}

bool destroy_connection(struct rdma_connection *conn)
{
    pthread_mutex_lock(&conn->destroy_connection_lock);
    if (conn->is_destroying || !conn->online || !conn->is_used) {
        pthread_mutex_unlock(&conn->destroy_connection_lock);
        return false;
    }
    // 标识正在被销毁以及不在线，抑制其他io发送
    conn->is_destroying = true;
    conn->online = false;
    pthread_mutex_unlock(&conn->destroy_connection_lock);

    destroy_connection_locked(conn);

    pthread_mutex_lock(&conn->destroy_connection_lock);
    conn->is_destroying = false;
    pthread_mutex_unlock(&conn->destroy_connection_lock);
    return true;
}

// 销毁所有被使用到的链接
void destroy_all_connection()
{
    exit_flag = 1;
    pthread_join(keepalive_tid, NULL);
    for (int i = 0; i < IP_NUM_T; i++) {
        if (conn[i].is_used) { try_destroy_connection(&conn[i]); }
    }
    nds_epoll_close();
    pthread_join(epoll_tid, NULL);
    AgentDestroy();
}

// 清理指定连接的所有 in-flight CID（包括心跳 CID，不在 task map 中）
void cleanup_conn_cids(int conn_idx)
{
    std::lock_guard<std::mutex> lock(g_map_mutex);
    // 1. 扫描所有 CID，标记属于该连接的 CID 及其占用的 slot
    for (int cid = 0; cid < MAX_CID_COUNT; cid++) {
        if (g_cid_metadata[cid].ip_index == conn_idx) {
            int slot_id = cid_to_slot[cid];
            if (slot_id >= 0) {
                int qp_idx = g_cid_metadata[cid].qp_index;
                if (qp_idx >= 0 && qp_idx < QP_NUM_T) {
                    QPContext &ctx = g_qp_ctx[conn_idx][qp_idx];
                    pthread_mutex_lock(&ctx.slot_lock);
                    bool was_in_use = ctx.slot_used[slot_id];
                    ctx.slot_used[slot_id] = false;
                    ctx.slot_to_cid[slot_id] = -1;
                    cid_to_slot[cid] = -1;
                    pthread_mutex_unlock(&ctx.slot_lock);
                    if (was_in_use) { sem_post(&ctx.window_sem); }
                }
            }
            // send_back_status[cid] == -1 表示 IO 没有poll到，没有收到回包
            // 此时将batch code置为 IPBECLEANED，key code 置为SENDRETRYFAILED
            if (send_back_status[cid] == -1) {
                send_back_status[cid] = IPBECLEANED;
                for (int k = 0; k < g_cid_metadata[cid].count && k < 512; k++) {
                    g_hitmap[cid][k] = (uint8_t)SENDRETRYFAILED;
                }
            }
            // 类型为HEARTBEAT的请求在这里释放cid，及meta数据
            // 已经调用nds_kv_get成功的cid在nds_kv_wait中释放cid及mate数据
            // 没调用nds_kv_wait的cid需要在send函数中就地释放
            if (g_cid_metadata[cid].optype == HEARTBEAT) { release_cid(cid); }
        }
    }
}

/**
 * @brief 根据一致性哈希将请求的索引分配到对应的连接组中
 */
static void nds_kv_group_by_connection(hash_ring_t *ring, IPC_data_block *ipcDataBlock,
                                       conn_group_t *groups, bool lookup_conn = false)
{
    // 初始化计数
    for (int i = 0; i < IP_NUM_T; i++) { groups[i].count = 0; }
    for (int i = 0; i < ipcDataBlock->addr_num; i++) {
        uint64_t h = ipcDataBlock->addr_pair_list[i].block_id.key_low;
        struct rdma_connection *target_conn =
            lookup_conn ? hash_ring_lookup(ring, h) : hash_ring_lookup_dpu(ring, h);

        if (target_conn && target_conn->online && !target_conn->is_destroying &&
            !target_conn->is_reconnecting) {
            int c_idx = target_conn->conn_idx;
            if (c_idx >= 0 && c_idx < IP_NUM_T && groups[c_idx].count < MAX_ADDR_NUM_IPC) {
                groups[c_idx].indices[groups[c_idx].count++] = i;
            }
            LOG_D("key_low: %lu, key_high: %u , localip: %s, dpuip: %s", h,
                  ipcDataBlock->addr_pair_list[i].block_id.key_high, target_conn->ip.local_ip,
                  target_conn->ip.server_ip);
        }
    }
}

static int nds_kv_group_by_connection_ns(IPC_data_block *ipcDataBlock, conn_group_t *groups)
{
    if (g_cfg.sceneMode == 1) {
        std::shared_ptr<NsidConn> nsidConn = FindNsidConnInMap(ipcDataBlock->nsid);
        if (!nsidConn) { return NDS_FAILED; }
        uint8_t ring_idx = UINT8_MAX;
        hash_ring_t *ring = nsidConn->current_ring(&ring_idx);
        nds_kv_group_by_connection(ring, ipcDataBlock, groups, false);
        nsidConn->release_ring_lock(ring_idx);
    } else {
        hash_ring_t *ring = &g_ring;
        nds_kv_group_by_connection(ring, ipcDataBlock, groups, false);
    }
    return NDS_SUCCESS;
}

// 从task_id对应的vec中删除所有等于 my_cid 的元素。由于每个 CID
// 对应任务是唯一的，所以只会删除一个元素。
static void remove_cid_from_task(uint64_t task_id, uint16_t cid)
{
    std::lock_guard<std::mutex> lock(g_map_mutex);
    auto it = g_task_cid_map.find(task_id);
    if (it != g_task_cid_map.end()) {
        auto &vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), cid), vec.end());
    }
}
// 把上层应用下发的req拆分为若干小req;
// 原则：
//     1、所有链接都失败返回错误码All_CONN_ERR
//     2、中间有小req发送失败就地释放此小req的cid，继续发送下一个
NDS_ERROR_T nds_kv_io_send_scatter_distributed(IPC_data_block *ipcDataBlock)
{
    uint64_t t_id = ipcDataBlock->task_id;
    OP_CODE type = (ipcDataBlock->op_type == KV_LOAD_BATCH) ? Batch_Retrieve : Batch_Store;
    // 若所有的链接都已经断开，返回
    // All_CONN_ERR，上层应用不调用wait函数，直接可以下发下一IO，hbm地址可以复用
    int trans_res = All_CONN_ERR;
    int ioSendSmallReqCounter = 0;

    // 1. 第一级：一致性哈希归类，把key打散至不同链接上，
    // 并用group[i].indices记录conn[i]的key在原始batch中的索引下标
    conn_group_t groups[IP_NUM_T];
    if (nds_kv_group_by_connection_ns(ipcDataBlock, groups) != NDS_SUCCESS) { return All_CONN_ERR; }

    // 2. 第二级：Conn 内部多 QP 分流
    for (int c = 0; c < IP_NUM_T; c++) {
        if (groups[c].count == 0) continue;
        struct rdma_connection *tmp_conn = &conn[c];
        // 此链接已断开不进行发送
        if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
            !tmp_conn->is_used) {
            continue;
        }
        // 把key的数量均分到每个链接的qp上
        int total_keys_in_conn = groups[c].count;
        int available_qps = g_cfg.io_qp_num;
        int use_qp_num = (total_keys_in_conn < available_qps) ? total_keys_in_conn : available_qps;
        int keys_per_qp = total_keys_in_conn / use_qp_num;
        int remaining = total_keys_in_conn % use_qp_num;
        int group_offset = 0;  // 当前连接内已处理的 key 偏移量
        for (int q = 0; q < use_qp_num; q++) {
            int qp_slice_size = keys_per_qp + (q < remaining ? 1 : 0);
            if (qp_slice_size <= 0) continue;
            int qpn = q + 1 - (tmp_conn->kv_cap_info.adminq_depth == 0);
            int slice_sent = 0;
            // 3. 第三级：Batch 限制处理 (切片发送)
            // 若均分qp后，key的数量大于g_cfg.max_batch_num，再按g_cfg.max_batch_num拆分发送
            while (slice_sent < qp_slice_size) {
                int current_batch = ((qp_slice_size - slice_sent) > g_cfg.max_batch_num)
                                        ? g_cfg.max_batch_num
                                        : (qp_slice_size - slice_sent);

                // --- CID 管理与任务映射 ---
                int my_cid = -1;
                int _retry = 0;
                NDS_ERROR_T acquireState = NDS_SUCCESS;
                while ((my_cid = acquire_cid()) == -1) {
                    if (++_retry > 100000) {
                        LOG_E("acquire_cid timeout for task %lu conn %d", t_id, c);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
                        !tmp_conn->is_used) {
                        LOG_E("Conn %d not usable during acquire_cid, abort task %lu", c, t_id);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    cpu_relax();
                }
                if (acquireState == NDS_FAILED) {
                    // 未成功获取cid，break，去尝试发下一个
                    break;
                }
                {
                    // 将获取的cid push到g_task_cid_map字典中，tid为key，cid为元数据
                    std::lock_guard<std::mutex> lock(g_map_mutex);
                    g_task_cid_map[t_id].push_back((uint16_t)my_cid);
                }

                g_cid_metadata[my_cid].count = current_batch;
                // 并用original_indices记录conn[i]的key在原始batch中的索引下标,
                // wait函数收集错误码时会用到original_indices
                for (int k = 0; k < current_batch; k++) {
                    g_cid_metadata[my_cid].original_indices[k] =
                        groups[c].indices[group_offset + k];
                }
                // 填充发送的消息字段，按协议严格填充
                data_block send_msg;
                memset(&send_msg, 0, sizeof(data_block));
                uint32_t current_rkey = 0;
                if (!tmp_conn->online || tmp_conn->rw_mr[ipcDataBlock->mr_id] == NULL) {
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                } else {
                    current_rkey = tmp_conn->rw_mr[ipcDataBlock->mr_id]->rkey;
                }
                ipc_storage_message_transfer_scatter_indexed(ipcDataBlock, &send_msg, current_rkey,
                                                             &groups[c].indices[group_offset],
                                                             current_batch);

                send_msg.cid = (uint16_t)my_cid;
                send_msg.opcode = type;
                LOG_D(
                    "[IO_SEND] Task:%lu | Conn:%d | QP:%d | CID:%d | BatchSize:%d | "
                    "SliceProgress:%d/%d",
                    t_id, c, qpn, my_cid, current_batch, slice_sent + current_batch, qp_slice_size);
                // 将发送的元数据保存在g_cid_metadata[my_cid]中，重传时使用同样的信息发送
                g_cid_metadata[my_cid].qp_index = qpn;
                g_cid_metadata[my_cid].ip_index = c;
                g_cid_metadata[my_cid].tmp_conn = tmp_conn;
                g_cid_metadata[my_cid].send_msg = send_msg;
                g_cid_metadata[my_cid].optype = BATCH;
                // --- 执行发送 ---
                trans_res = transfer_data_send(qpn, c, tmp_conn, &send_msg, current_batch);
                if (trans_res != NDS_SUCCESS) {
                    LOG_D("transfer_data_send kv io failed, conn %d qp %d task_id=%lu cid=%d", c,
                          qpn, t_id, my_cid);
                    // 发送失败, 先从task_id中移除cid，然后就地释放 cid元数据
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                }
                slice_sent += current_batch;
                group_offset += current_batch;
                // 发送计数connSendCounter原子加1
                uint64_t connSendCounter = __sync_fetch_and_add(&tmp_conn->connSendCounter, 1);
                LOG_D("tmp_conn->connSendCounter: %ld\n", connSendCounter);
                ioSendSmallReqCounter++;
            }
        }
    }
    return (ioSendSmallReqCounter == 0) ? All_CONN_ERR : trans_res;
}

// 单key调用接口，ipcDataBlock只有一个key，按key选到链接后，轮训选一个qp发送
NDS_ERROR_T nds_kv_io_send_scatter_distributed_single(IPC_data_block *ipcDataBlock)
{
    uint64_t t_id = ipcDataBlock->task_id;
    OP_CODE type = (ipcDataBlock->op_type == KV_LOAD) ? Retrieve : Store;
    int trans_res = All_CONN_ERR;
    int ioSendSmallReqCounter = 0;

    // 1. 第一级：一致性哈希归类 (Key -> Conn)
    conn_group_t groups[IP_NUM_T];
    if (nds_kv_group_by_connection_ns(ipcDataBlock, groups) != NDS_SUCCESS) { return All_CONN_ERR; }

    // 2. 第二级：Conn 内部多 QP 分流
    for (int c = 0; c < IP_NUM_T; c++) {
        if (groups[c].count == 0) continue;
        // 一个请求场景只有一个req，内部不分qp，默认选择一个
        struct rdma_connection *tmp_conn = &conn[c];
        if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
            !tmp_conn->is_used) {
            continue;
        }
        // 复用了batch的逻辑，实际就一个key不用分，一个qp也就发一个key
        int total_keys_in_conn = groups[c].count;
        int available_qps = g_cfg.io_qp_num;
        int use_qp_num = (total_keys_in_conn < available_qps) ? total_keys_in_conn : available_qps;
        int keys_per_qp = total_keys_in_conn / use_qp_num;
        int remaining = total_keys_in_conn % use_qp_num;
        // for循环实际就用到 q = 0, 其余qp_slice_size都小于0
        for (int q = 0; q < use_qp_num; q++) {
            int qp_slice_size = keys_per_qp + (q < remaining ? 1 : 0);
            if (qp_slice_size <= 0) continue;
            // 轮训选择一个qp
            uint64_t old = __sync_fetch_and_add(&counter, 1);
            int qpn = (tmp_conn->kv_cap_info.adminq_depth != 0) + old % g_cfg.io_qp_num;
            int slice_sent = 0;
            // 3. 第三级：Batch 限制处理 (切片发送)
            while (slice_sent < qp_slice_size) {
                int current_batch = ((qp_slice_size - slice_sent) > g_cfg.max_batch_num)
                                        ? g_cfg.max_batch_num
                                        : (qp_slice_size - slice_sent);

                // --- CID 管理与任务映射 ---
                int my_cid = -1;
                int _retry = 0;
                NDS_ERROR_T acquireState = NDS_SUCCESS;
                while ((my_cid = acquire_cid()) == -1) {
                    if (++_retry > 100000) {
                        LOG_E("acquire_cid timeout for task %lu conn %d (single)", t_id, c);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
                        !tmp_conn->is_used) {
                        LOG_E("Conn %d not usable during acquire_cid (single), abort task %lu", c,
                              t_id);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    cpu_relax();
                }
                if (acquireState == NDS_FAILED) {
                    // 未成功获取cid，break，去尝试发下一个
                    break;
                }
                {
                    std::lock_guard<std::mutex> lock(g_map_mutex);
                    g_task_cid_map[t_id].push_back((uint16_t)my_cid);
                }
                g_cid_metadata[my_cid].count = current_batch;
                data_block_single send_msg;
                memset(&send_msg, 0, sizeof(data_block_single));
                uint32_t current_rkey = 0;
                if (!tmp_conn->online || tmp_conn->rw_mr[ipcDataBlock->mr_id] == NULL) {
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                } else {
                    current_rkey = tmp_conn->rw_mr[ipcDataBlock->mr_id]->rkey;
                }
                ipc_storage_message_transfer_single(ipcDataBlock, &send_msg, current_rkey);
                send_msg.cid = (uint16_t)my_cid;
                send_msg.opcode = type;
                LOG_D("[IO_SEND] Task:%lu | Conn:%d | QP:%d | CID:%d | IO Count: %d", t_id, c, qpn,
                      my_cid, current_batch);
                g_cid_metadata[my_cid].qp_index = qpn;
                g_cid_metadata[my_cid].ip_index = c;
                g_cid_metadata[my_cid].tmp_conn = tmp_conn;
                g_cid_metadata[my_cid].single_send_msg = send_msg;
                g_cid_metadata[my_cid].optype = SINGLE;  // 标识 single
                // --- 执行发送 ---
                trans_res = transfer_data_send_single(qpn, c, tmp_conn, &send_msg);
                if (trans_res != NDS_SUCCESS) {
                    LOG_E(
                        "transfer_data_send_single kv io failed, conn %d qp %d task_id=%lu cid=%d",
                        c, qpn, t_id, my_cid);
                    // 发送失败, 先从task_id中移除cid，然后就地释放 cid元数据
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                }
                LOG_D("send succeed");
                slice_sent += current_batch;
                __sync_fetch_and_add(&tmp_conn->connSendCounter, 1);
                ioSendSmallReqCounter++;
            }
        }
    }
    return (ioSendSmallReqCounter == 0) ? All_CONN_ERR : trans_res;
}

NDS_ERROR_T nds_kv_ctl_send_scatter_distributed(IPC_data_block *ipcDataBlock)
{
    uint64_t t_id = ipcDataBlock->task_id;
    OP_CODE type = (ipcDataBlock->op_type == IS_EXIST) ? Exist : Delete;
    int trans_res = All_CONN_ERR;
    int ioSendSmallReqCounter = 0;

    // 1. 第一级：一致性哈希归类 (Key -> Conn)
    conn_group_t groups[IP_NUM_T];
    if (nds_kv_group_by_connection_ns(ipcDataBlock, groups) != NDS_SUCCESS) { return All_CONN_ERR; }

    // 2. 第二级：Conn 内部多 QP 分流
    for (int c = 0; c < IP_NUM_T; c++) {
        if (groups[c].count == 0) continue;

        struct rdma_connection *tmp_conn = &conn[c];
        if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
            !tmp_conn->is_used) {
            continue;
        }

        int total_keys_in_conn = groups[c].count;

        int available_qps = 1;  // 查询命中只用一个qp就好，不用打散：QP_NUM_T - 1;
        int use_qp_num = (total_keys_in_conn < available_qps) ? total_keys_in_conn : available_qps;
        int keys_per_qp = total_keys_in_conn / use_qp_num;
        int remaining = total_keys_in_conn % use_qp_num;
        int group_offset = 0;  // 当前连接内已处理的 key 偏移量

        for (int q = 0; q < use_qp_num; q++) {
            int qp_slice_size = keys_per_qp + (q < remaining ? 1 : 0);
            if (qp_slice_size <= 0) continue;

            int qpn = q + 1 - (tmp_conn->kv_cap_info.adminq_depth == 0);
            int slice_sent = 0;

            // 3. 第三级：Batch 限制处理 (切片发送)
            while (slice_sent < qp_slice_size) {
                int current_batch = ((qp_slice_size - slice_sent) > MAX_CTL_ADDR_NUM)
                                        ? MAX_CTL_ADDR_NUM
                                        : (qp_slice_size - slice_sent);

                // --- CID 管理与任务映射 ---
                int my_cid = -1;
                int _retry = 0;
                NDS_ERROR_T acquireState = NDS_SUCCESS;
                while ((my_cid = acquire_cid()) == -1) {
                    if (++_retry > 100000) {
                        LOG_E("acquire_cid timeout for task %lu conn %d (ctl)", t_id, c);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    if (!tmp_conn->online || tmp_conn->is_destroying || tmp_conn->is_reconnecting ||
                        !tmp_conn->is_used) {
                        LOG_D("Conn %d not usable during acquire_cid (ctl), abort task %lu", c,
                              t_id);
                        acquireState = NDS_FAILED;
                        break;
                    }
                    cpu_relax();
                }
                if (acquireState == NDS_FAILED) {
                    // 未成功获取cid，break，去尝试发下一个
                    break;
                }
                {
                    std::lock_guard<std::mutex> lock(g_map_mutex);
                    g_task_cid_map[t_id].push_back((uint16_t)my_cid);
                }
                g_cid_metadata[my_cid].count = current_batch;
                for (int k = 0; k < current_batch; k++) {
                    g_cid_metadata[my_cid].original_indices[k] =
                        groups[c].indices[group_offset + k];
                }

                // --- 构造 RDMA 消息 ---
                ctl_block send_msg;
                memset(&send_msg, 0, sizeof(ctl_block));
                if (!tmp_conn->online || tmp_conn->rw_mr[ipcDataBlock->mr_id] == NULL) {
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                }
                ipc_storage_message_transfer_ctl_scatter(
                    ipcDataBlock, &send_msg, &groups[c].indices[group_offset], current_batch);
                send_msg.cid = (uint16_t)my_cid;
                send_msg.opcode = type;
                LOG_D(
                    "[IO_SEND] Task:%lu | Conn:%d | QP:%d | CID:%d | BatchSize:%d | "
                    "SliceProgress:%d/%d",
                    t_id, c, qpn, my_cid, current_batch, slice_sent + current_batch, qp_slice_size);

                g_cid_metadata[my_cid].qp_index = qpn;
                g_cid_metadata[my_cid].ip_index = c;
                g_cid_metadata[my_cid].tmp_conn = tmp_conn;
                g_cid_metadata[my_cid].ctl_send_msg = send_msg;
                g_cid_metadata[my_cid].optype = CLT_EXIST_DELETE;  // ctl
                // --- 执行发送 ---
                trans_res = transfer_ctl_send(qpn, c, tmp_conn, &send_msg, current_batch);
                if (trans_res != NDS_SUCCESS) {
                    LOG_D("transfer_ctl_send kv io failed, conn %d qp %d task_id=%lu cid=%d", c,
                          qpn, t_id, my_cid);
                    remove_cid_from_task(t_id, my_cid);
                    release_cid(my_cid);
                    break;
                }
                LOG_D("send succeed");
                slice_sent += current_batch;
                group_offset += current_batch;
                __sync_fetch_and_add(&tmp_conn->connSendCounter, 1);
                ioSendSmallReqCounter++;
            }
        }
    }
    return (ioSendSmallReqCounter == 0) ? All_CONN_ERR
                                        : trans_res;  // 上层应用没收到All_CONN_ERR都应该调用wait
}

/**
 * 查找并提取 task_id 所有 CID
 */

static std::vector<uint16_t> fetch_cids(uint64_t task_id)
{
    std::vector<uint16_t> target_cids;
    std::lock_guard<std::mutex> lock(g_map_mutex);
    auto it = g_task_cid_map.find(task_id);
    if (it == g_task_cid_map.end()) {
        LOG_E("task id %" PRIu64 " not found", task_id);
        return target_cids;
    }
    target_cids = it->second;
    g_task_cid_map.erase(it);
    return target_cids;
}

/**
 * 按照原始顺序回传结果
 */

static void collect_error_codes_original_order(uint16_t cid, std::vector<uint8_t> &hit_map,
                                               uint32_t total_keys)
{
    cid_metadata_t *meta = &g_cid_metadata[cid];
    for (int i = 0; i < meta->count; ++i) {
        uint32_t original_index = meta->original_indices[i];
        if (original_index < total_keys) { hit_map[original_index] = g_hitmap[cid][i]; }
    }
    g_notify_addr[cid] = 0;
    send_back_status[cid] = -1;
    release_cid(cid);
}

/**
 * 收集错误码，若查到更新hit_map为1
 */
static void collect_hit_maps(uint16_t cid, std::vector<uint8_t> &hit_map, uint32_t total_blocks)
{
    cid_metadata_t *meta = &g_cid_metadata[cid];
    uint32_t hit_count_in_cid = g_notify_addr[cid];

    for (uint32_t i = 0; i < hit_count_in_cid; ++i) {
        uint32_t original_index = meta->original_indices[i];
        if (original_index < total_blocks) { hit_map[original_index] = 1; }
    }
    g_notify_addr[cid] = 0;
    send_back_status[cid] = -1;
    release_cid(cid);
}

/**
 * 收集 key code
 */

static void build_key_error_set(const std::vector<uint8_t> &hit_map, uint32_t total_keys,
                                CODE_SET &code)
{
    for (uint32_t i = 0; i < total_keys; ++i) { code.key_code_set.push_back(hit_map[i]); }
}

/**
 * 根据操作类型选择合适的数据传输方式
 */
static NDS_ERROR_T transfer_data(uint16_t cid)
{
    if (g_cid_metadata[cid].optype == SINGLE) {
        LOG_W(
            "[IO_RESEND] There are failed I/Os, triggering a retransmission. Conn:%d | QP:%d | "
            "CID:%d",
            g_cid_metadata[cid].ip_index, g_cid_metadata[cid].qp_index, cid);
        return transfer_data_send_single(g_cid_metadata[cid].qp_index, g_cid_metadata[cid].ip_index,
                                         g_cid_metadata[cid].tmp_conn,
                                         &g_cid_metadata[cid].single_send_msg);
    } else if (g_cid_metadata[cid].optype == BATCH) {
        LOG_W(
            "[IO_RESEND] There are failed I/Os, triggering a retransmission. Conn:%d | QP:%d | "
            "CID:%d | IO Count: %d",
            g_cid_metadata[cid].ip_index, g_cid_metadata[cid].qp_index, cid,
            g_cid_metadata[cid].count);
        return transfer_data_send(g_cid_metadata[cid].qp_index, g_cid_metadata[cid].ip_index,
                                  g_cid_metadata[cid].tmp_conn, &g_cid_metadata[cid].send_msg,
                                  g_cid_metadata[cid].count);
    } else if (g_cid_metadata[cid].optype == CLT_EXIST_DELETE) {
        LOG_W(
            "[IO_RESEND] There are failed I/Os, triggering a retransmission. Conn:%d | QP:%d | "
            "CID:%d | IO Count: %d",
            g_cid_metadata[cid].ip_index, g_cid_metadata[cid].qp_index, cid,
            g_cid_metadata[cid].count);
        return transfer_ctl_send(g_cid_metadata[cid].qp_index, g_cid_metadata[cid].ip_index,
                                 g_cid_metadata[cid].tmp_conn, &g_cid_metadata[cid].ctl_send_msg,
                                 g_cid_metadata[cid].count);
    } else {
        LOG_E("Error: Invalid optype %u for cid %u\n", g_cid_metadata[cid].optype, cid);
        return NDSERRORINVALIDOPTYPE;
    }
}

/**
 * exist/delete 遍历得到前缀匹配；put/get 按原batch中key的顺序收集key code
 */
static void process_cid_metadata(uint16_t cid, std::vector<uint8_t> &hit_map, uint32_t total_keys,
                                 bool force_error = false)
{
    if (cid >= MAX_CID_COUNT) {
        LOG_E("Invalid cid %u in process_cid_metadata", (unsigned)cid);
        return;
    }
    if (hit_map.size() < total_keys) { LOG_E("hit_map size is insufficient: %zu", hit_map.size()); }
    if (force_error) {
        cid_metadata_t *meta = &g_cid_metadata[cid];
        if (meta->optype == CLT_EXIST_DELETE) {
            // exist/delete error: hit_map stays 0 (not found), no read from g_hitmap
        } else {
            // get/put error: directly write SENDRETRYFAILED, no read from g_hitmap
            for (int i = 0; i < meta->count; ++i) {
                uint32_t original_index = meta->original_indices[i];
                if (original_index < total_keys) {
                    hit_map[original_index] = (uint8_t)SENDRETRYFAILED;
                }
            }
        }
        g_notify_addr[cid] = 0;
        send_back_status[cid] = -1;
        release_cid(cid);
    } else if (g_cid_metadata[cid].optype == CLT_EXIST_DELETE) {
        collect_hit_maps(cid, hit_map, total_keys);
    } else {
        collect_error_codes_original_order(cid, hit_map, total_keys);
    }
}

/**
 * 等待接口：配合全局 CID 通知，put/get接口，返回整个batch的错误码合集及每个子io的错误码,
 * 超时及错误io重传
 */
static NDS_ERROR_T wait_for_io_completion(uint16_t cid, uint32_t total_keys,
                                          std::vector<uint8_t> &hit_map)
{
    NDS_ERROR_T ret = TIMEOUTRETRY;
    bool success = false;
    NDS_ERROR_T trans_res;
    bool retried = false;
    while (!success) {
        struct rdma_connection *conn = g_cid_metadata[cid].tmp_conn;
        int ip_idx = g_cid_metadata[cid].ip_index;

        if (send_back_status[cid] != -1 && ip_idx < 0) {
            ret = send_back_status[cid];
            return IPBECLEANED;
        }

        if (ip_idx < 0) { return IPBECLEANED; }
        auto start_time = std::chrono::high_resolution_clock::now();
        while (send_back_status[cid] == -1) {
            ip_idx = g_cid_metadata[cid].ip_index;
            if (ip_idx < 0 && send_back_status[cid] == -1) { return IPBECLEANED; }
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            if (elapsed.count() > g_io_request_timeout) {
                LOG_W("There is an I/O timeout, triggering a disconnection and reconnection.");
                if (ip_idx >= 0 && ip_idx < IP_NUM_T) {
                    if (try_destroy_connection(&::conn[ip_idx])) {
                        start_reconnect_timer(&::conn[ip_idx]);
                    }
                }
                return TIMEOUTRETRY;
            }
            cpu_relax();
        }
        if (send_back_status[cid] != -1) {
            ret = send_back_status[cid];
            if (ret == IPBECLEANED || ret == POLLCQERROR) {
                return ret;  // 断链错误码：不处理metadata，交由nds_kv_wait
            }
            bool condition = is_io_retry_error(ret) && g_cid_metadata[cid].optype == SINGLE;
            bool flag = condition && !retried;
            if (flag) {
                if (!conn || ip_idx < 0 || !::conn[ip_idx].online || ::conn[ip_idx].is_destroying ||
                    ::conn[ip_idx].is_reconnecting) {
                    g_notify_addr[cid] = 0;
                    send_back_status[cid] = -1;
                    return SENDRETRYFAILED;
                }
                g_notify_addr[cid] = 0;
                send_back_status[cid] = -1;
                if (ret == IO_ERR_RESOURCE_BUSY) {
                    LOG_W("IO_ERR_RESOURCE_BUSY, sleep: %dms", g_cfg.busy_duration);
                    usleep(g_cfg.busy_duration * 1000);
                }
                trans_res = transfer_data(cid);
                if (trans_res != NDS_SUCCESS) {
                    if (ip_idx >= 0 && ip_idx < IP_NUM_T) {
                        if (try_destroy_connection(&::conn[ip_idx])) {
                            start_reconnect_timer(&::conn[ip_idx]);
                        }
                    }
                    return SENDRETRYFAILED;
                }
                start_time = std::chrono::high_resolution_clock::now();
                retried = true;
                continue;
            }
            process_cid_metadata(cid, hit_map, total_keys);
            success = true;
        }
    }
    return ret;
}

/**
 * 等待接口：配合全局 CID 通知，put/get wait接口
 */
CODE_SET nds_kv_wait(uint64_t task_id, uint32_t total_keys)
{
    CODE_SET code;
    std::vector<uint16_t> target_cids = fetch_cids(task_id);

    if (target_cids.empty()) {
        code.batch_code_set.push_back(TASKIDNOTFIND);
        code.key_code_set.push_back(TASKIDNOTFIND);
        return code;
    }

    std::vector<uint8_t> hit_map(total_keys, (uint8_t)SENDRETRYFAILED);
    LOG_D("[DEBUG] TASK: %" PRIu64 ", Start Wait", task_id);

    for (uint16_t cid : target_cids) {
        NDS_ERROR_T ret = wait_for_io_completion(cid, total_keys, hit_map);
        code.batch_code_set.push_back(ret);
        if (ret == TIMEOUTRETRY || ret == SENDRETRYFAILED || ret == IPBECLEANED ||
            ret == POLLCQERROR) {
            process_cid_metadata(cid, hit_map, total_keys, true);
        }
    }

    build_key_error_set(hit_map, total_keys, code);

    LOG_D("[DEBUG] TASK: %" PRIu64 " wait end", task_id);
    return code;
}

void nds_kv_distroy_send(struct rdma_connection *conn, IPC_data_block *ipcDataBlock)
{
    (void)conn;
    (void)ipcDataBlock;
    // TODO
    return;
}

param_t *process_config(const char *path)
{
    param_t *p = (param_t *)malloc(sizeof(param_t));
    if (p == NULL) return NULL;

    memset(p, 0, sizeof(param_t));
    p->ip_count = 0;
    p->timer_switch = false;  // 默认timer_switch为关

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        LOG_W("Unable to open profile %s, using default", path);
        return p;
    }

    char line[MAX_LINE_LEN];
    char local[32], server[32];
    uint64_t server_port, roce_idx, roce_port, dpu_idx, start_core, log_level, slot_num, io_qp_num,
        max_batch_num, core_num;
    uint8_t priority, tos;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        if (strncmp(line, "ip_config", strlen("ip_config")) == 0 && g_cfg.sceneMode == 0) {
            if (p->ip_count < IP_NUM_T) {
                int ret = sscanf(line,
                                 "ip_config = %31[^,], %31[^,], %" SCNu64 ", %" SCNu64 ", %" SCNu64
                                 ", %" SCNu64 "",
                                 local, server, &server_port, &roce_idx, &roce_port, &dpu_idx);

                if (ret == 6) {
                    strcpy(p->ip[p->ip_count].local_ip, local);
                    strcpy(p->ip[p->ip_count].server_ip, server);
                    p->ip[p->ip_count].server_port = server_port;
                    // p->ip[p->ip_count].start_core = start_core;

                    if (roce_idx < 256)
                        p->ip[p->ip_count].roce_idx = roce_idx;
                    else
                        LOG_W("Warning: GID index %" PRIu64
                              " is out of uint8_t range [0 - 255], skip this configuration line",
                              roce_idx);

                    p->ip[p->ip_count].roce_port = roce_port;
                    p->ip[p->ip_count].dpu_id = dpu_idx;
                    p->ip_count++;
                } else {
                    LOG_W(
                        "Warning: ip_config line malformed or missing field (only %d field(s) "
                        "read): %s",
                        ret, line);
                }
            } else {
                LOG_W("Warning: The number of ip_configs has reached the maximum (%d). Skipping",
                      IP_NUM_T);
            }
        }
        if (strstr(line, "start_core")) {
            if (sscanf(line, "start_core = %" SCNu64 "", &start_core) == 1) {
                p->start_core = start_core;
            }
        }
        if (strstr(line, "log_level")) {
            if (sscanf(line, "log_level = %" SCNu64 "", &log_level) == 1) {
                p->log_level = (LogLevel)log_level;
            }
        }
        if (strstr(line, "qp_slot_num")) {
            if (sscanf(line, "qp_slot_num = %" SCNu64 "", &slot_num) == 1) {
                p->qp_slot_num = slot_num;
            }
        }
        if (strstr(line, "use_qp_num")) {
            if (sscanf(line, "use_qp_num = %" SCNu64 "", &io_qp_num) == 1) {
                p->use_qp_num = io_qp_num;
            }
        }
        if (strstr(line, "max_batch_num")) {
            if (sscanf(line, "max_batch_num = %" SCNu64 "", &max_batch_num) == 1) {
                p->max_batch_num = max_batch_num;
            }
        }
        if (strstr(line, "core_num")) {
            if (sscanf(line, "core_num = %" SCNu64 "", &core_num) == 1) { p->core_num = core_num; }
        }
        if (strstr(line, "priority")) {
            int ret = sscanf(line, "priority = %hhu", &priority);
            if (ret != 1) {
                LOG_E("Error: Failed to parse priority value\n");
                p->priority = 3;  // 解析失败默认使用队列3
            } else {
                p->priority = priority;
                if (p->priority > 7 || p->priority < 1) {
                    LOG_E("Error: Invalid priority value %u (must be 1-7)\n", p->priority);
                    p->priority = 3;  // 解析失败默认使用队列3
                }
            }
        }
        if (strstr(line, "tos")) {
            int ret = sscanf(line, "tos = %hhu", &tos);
            if (ret != 1) {
                LOG_E("Error: Failed to parse tos value\n");
                p->tos = 104;  // 解析失败默认使用 104
            } else {
                p->tos = tos;
                if (p->tos < 32) {
                    LOG_E("Error: Invalid tos value %u (must be 32-255)\n", p->tos);
                    p->tos = 104;  // 解析失败默认使用 104
                }
            }
        }
        if (strstr(line, "timer_switch")) {
            char timer_switch[8] = {0};
            int ret = sscanf(line, "timer_switch = %7s", timer_switch);
            if (ret == 1) { p->timer_switch = (strcmp(timer_switch, "on") == 0); }
        }
        if (strstr(line, "busy_duration")) {
            uint32_t busy_duration;
            int ret = sscanf(line, "busy_duration = %u", &busy_duration);
            if (ret != 1) {
                LOG_E("Error: Failed to parse priority value\n");
                p->busy_duration = 1000;  // 解析失败默认1000ms
            } else {
                p->busy_duration = busy_duration;
            }
        }
        if (strstr(line, "reconnection_interval")) {
            g_cfg.interval = NULL;
            g_cfg.interval_count = parse_interval_list(line, &g_cfg.interval);
            if (g_cfg.interval_count <= 0) {
                LOG_E("Error: Failed to parse reconnection_interval");
            }
        }
        if (strstr(line, "heartbeat_interval")) {
            uint32_t heartbeat_interval;
            int ret = sscanf(line, "heartbeat_interval = %u", &heartbeat_interval);
            if (ret != 1) {
                LOG_E("Error: Failed to parse heartbeat_interval value\n");
                p->heartbeat_interval = 1;  // 解析失败默认1000ms
            } else {
                p->heartbeat_interval = heartbeat_interval;
            }
        }
        if (strstr(line, "kato")) {
            unsigned int kato_val;
            int ret = sscanf(line, "kato = %u", &kato_val);
            if (ret != 1 || kato_val > 65535) {
                LOG_E("Error: Failed to parse kato value\n");
                p->kato = 10;
            } else {
                p->kato = (uint16_t)kato_val;
            }
        }
    }

    if (g_cfg.interval_count <= 0) {
        LOG_E("Error: reconnection_interval is missing or invalid in config %s", path);
        free(p);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return p;
}

void nds_kv_get_nds_config(const char *path_nds_config)
{
    global_path_nds_config = path_nds_config;
}

static NDS_ERROR_T nds_init_bind_thread(int base_core)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(base_core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity");
        return NDS_FAILED;
    }
    return NDS_SUCCESS;
}

static void nds_init_reset_conns(void)
{
    for (int i = 0; i < IP_NUM_T; i++) {
        conn[i].conn_idx = -1;
        conn[i].online = false;
        conn[i].is_reconnecting = false;
        conn[i].is_destroying = false;
        conn[i].reconnection_count = 0;
        conn[i].is_starting_timer = false;
        conn[i].io_count = 0;
        conn[i].old_io_count = 0;
        conn[i].connSendCounter = 0;
        conn[i].connRecvCounter = 0;
        conn[i].is_used = false;
        conn[i].is_destroyed = true;
        pthread_mutex_init(&conn[i].destroy_connection_lock, NULL);
        pthread_mutex_init(&conn[i].reconnection_lock, NULL);
        pthread_mutex_init(&conn[i].start_timer_lock, NULL);
    }
}

static NDS_ERROR_T nds_init_epoll_create(void)
{
    // 创建 epoll 实例
    g_epoll_args.epfd = epoll_create(1);
    if (g_epoll_args.epfd < 0) {
        LOG_E("epoll_create1 failed: %s", strerror(errno));
        return NDS_FAILED;
    }
    g_epoll_args.running = true;
    return NDS_SUCCESS;
}

static void nds_init_register_dpu(int i, int d_id)
{
    dpu_group_t *target_dpu = NULL;
    // 查找是否已记录该 DPU
    for (int j = 0; j < g_dpu_count; j++) {
        if (g_dpus[j].dpu_id == d_id) {
            target_dpu = &g_dpus[j];
            break;
        }
    }
    // 如果是新 DPU，初始化并加入哈希环
    if (target_dpu == NULL) {
        target_dpu = &g_dpus[g_dpu_count++];
        target_dpu->dpu_id = d_id;
        target_dpu->conn_count = 0;
        target_dpu->next_conn_idx = 0;

        // 以 DPU 为单位添加虚拟节点到环上
        hash_ring_add_dpu(&g_ring, target_dpu);
    }
    // 将当前连接存入该 DPU 的列表
    target_dpu->conns[target_dpu->conn_count++] = &conn[i];
}

static NDS_ERROR_T nds_init_setup_connection(int i, param_t *param, void **kv_cache, int mr_num)
{
    LOG_I("%s, %s, %d, %d, %d, %d", param->ip[i].local_ip, param->ip[i].server_ip,
          param->ip[i].server_port, param->ip[i].roce_idx, param->ip[i].roce_port,
          param->ip[i].dpu_id);
    conn[i].conn_idx = i;

    save_ip_information(i, param->ip[i]);  // 保存建链信息，重新建链时使用

    if (init_rdma_device_and_mr(&conn[i], param->ip[i].local_ip, kv_cache, mr_num) != NDS_SUCCESS) {
        die("init_rdma_device_and_mr failed");
    }

    if (setup_tcp_connection(&conn[i], param->ip[i].local_ip, param->ip[i].server_ip,
                             param->ip[i].server_port) != NDS_SUCCESS) {
        die("TCP connection setup failed");
    }
    LOG_I("setup_tcp_connection [%d] success", i);

    if (setup_rdma_connection(&conn[i], param->ip[i].roce_idx, param->ip[i].roce_port) !=
        NDS_SUCCESS) {
        die("RDMA connection setup failed");
    }
    LOG_I("setup_rdma_connection [%d] success\n", i);

    conn[i].online = true;
    // 将 TCP fd 添加到 epoll
    if (nds_epoll_add_conn(&conn[i]) != NDS_SUCCESS) {
        LOG_E("Failed to add conn %d to epoll", i);
        return NDS_FAILED;
    }

    nds_init_register_dpu(i, param->ip[i].dpu_id);
    return NDS_SUCCESS;
}

static NDS_ERROR_T nds_init_start_poll_threads(param_t *param)
{
    int base_core = param->start_core;
    // 集群和非集群都按start和step均分到param->core_num个线程上
    int num_connections = param->ip_count;
    cpu_set_t cpuset;
    if (param->core_num <= 0) {
        LOG_E("core_num must be positive, got %d", param->core_num);
        return NDS_FAILED;
    }
    if (param->core_num > num_connections) {
        LOG_W("core_num(%d) > connection_count(%d), adjusting to %d", param->core_num,
              num_connections, num_connections);
        param->core_num = num_connections;
    }

    for (int i = 0; i < param->core_num; i++) {
        thread_args_t *args = (thread_args_t *)malloc(sizeof(thread_args_t));
        if (args == NULL) { die("Failed to allocate thread args"); }
        args->start_thread = i;
        args->step = param->core_num;
        LOG_D("args->start_thread: %d, args->step: %d", args->start_thread, args->step);

        if (pthread_create(&conn[i].recv_parser_thread, NULL, unified_poll_thread, args)) {
            die("Failed to create recv_parser thread");
        }
        CPU_ZERO(&cpuset);
        CPU_SET(base_core + i + 1, &cpuset);
        pthread_setaffinity_np(conn[i].recv_parser_thread, sizeof(cpu_set_t), &cpuset);

        LOG_I("Thread %d started: handling start %d, step %d on core %d", i, args->start_thread,
              args->step, base_core + i + 1);
    }
    return NDS_SUCCESS;
}

static void nds_init_start_aux_threads(void)
{
    // 启动 epoll 监听线程
    if (pthread_create(&epoll_tid, NULL, epoll_listener_thread, NULL)) {
        die("Failed to create epoll listener thread");
    }
    LOG_I("Epoll listener thread started");

    // 启动keepalive线程
    if (pthread_create(&keepalive_tid, NULL, keepalive_thread, NULL)) {
        die("Failed to create keepalive thread");
    }
    LOG_I("KeepAlive thread started");
}

static void nds_init_completion_state(void)
{
    for (int i = 0; i < MAX_CID_COUNT; ++i) {
        g_notify_addr[i] = 0;  // 强制触发写操作，分配物理页
        send_back_status[i] = -1;
    }
    init_cid_pool();
    init_qp_contexts();
}

NDS_ERROR_T nds_init(void **kv_cache, int mr_num)
{
    // 完成建链、mr注册
    printf(SDK_VERSION_INFO "\n");
    signal(SIGPIPE, SIG_IGN);
    g_kv_cache = kv_cache;
    g_mr_num = mr_num;

    g_cfg.sceneMode = SceneMode(global_path_nds_config);
    if (g_cfg.sceneMode == 1) {
        if (AgentInit("conf/agent.conf")) {
            LOG_E("ClusterAgentInit field.");
            return NDS_FAILED;
        }
        g_local_ip_param = ParseLocalipList(global_path_nds_config);
    }

    param_t *param = process_config(global_path_nds_config);
    if (param == NULL) {
        LOG_E("process_config failed, nds_init aborted");
        return NDS_ERR_INVALID_PARAM;
    }
    if (nds_init_bind_thread(param->start_core) != NDS_SUCCESS) { return NDS_FAILED; }
    init_runtime_configs(param);

    nds_init_reset_conns();

    hash_ring_init(&g_ring);
    set_log_level(param->log_level);

    nds_init_completion_state();

    if (param->ip_count > IP_NUM_T) {
        perror("ip_count > IP_NUM_T");
        return NDS_FAILED;
    }

    if (nds_init_epoll_create() != NDS_SUCCESS) { return NDS_FAILED; }
    int conn_count = 0;
    if (g_cfg.sceneMode == 1) {
        // 在构建哈希环前获取一次视图版本号
        // 从而在构建期间如果视图发生变化，在注册update函数后能够感知到这次视图变化
        constexpr int AGENT_VIEW_MAX_RETRIES = 30;
        constexpr useconds_t AGENT_VIEW_RETRY_DELAY_US = 1000000;
        for (int retry = 0; retry < AGENT_VIEW_MAX_RETRIES; ++retry) {
            old_dpu_ip_list = GetAllInDpuipList();
            if (!old_dpu_ip_list.empty()) { break; }
            LOG_W("Agent returned an empty DPU view (attempt %d/%d)", retry + 1,
                  AGENT_VIEW_MAX_RETRIES);
            usleep(AGENT_VIEW_RETRY_DELAY_US);
        }
        if (old_dpu_ip_list.empty()) {
            LOG_E("Agent returned no available DPUs after %d seconds", AGENT_VIEW_MAX_RETRIES);
            return NDS_ERR_TIMEOUT;
        }
        uint64_t view_version = GetViewVersion();
        LOG_I("view_version: %ld", view_version);
        old_nsid_list = GetAllNsidList();
        old_dpuip_to_nsid = BuildDpuipNsidMap();
        old_nsid_to_dpuip = GetNsidDpuipMap(GetNameNsidMap());
        old_dpuip_to_dpuid = GetAllInDpuipDpuidMap();
        if (SetupClusterConnection(g_local_ip_param, old_dpu_ip_list, old_dpuip_to_nsid,
                                   old_dpuip_to_dpuid, conn_count)) {
            LOG_E("nds_init: All connection failed to be established.");
            return NDS_FAILED;
        }
        param->ip_count = conn_count;
        g_cfg.ip_count = conn_count;
        // 在集群视图初始化构建完成后再设置update回调函数
        AgentSetUpdateCb(rebuild_cluster_hash_ring, view_version);
    } else {
        for (int i = 0; i < param->ip_count; i++) {
            conn[i].is_used = true;
            if (nds_init_setup_connection(i, param, kv_cache, mr_num) != NDS_SUCCESS) {
                return NDS_FAILED;
            }
        }
    }

    if (nds_init_start_poll_threads(param) != NDS_SUCCESS) { return NDS_FAILED; }

    nds_init_start_aux_threads();

    free(param);
    return NDS_SUCCESS;
}

/**
 * exist/delete 接口计算前缀命中长度
 */
int calculate_prefix_hit_len(const std::vector<uint8_t> &hit_map, uint32_t total_blocks)
{
    int prefix_hit_len = 0;
    for (uint32_t i = 0; i < total_blocks; ++i) {
        if (hit_map[i] == 1) {
            prefix_hit_len++;
        } else {
            break;
        }
    }
    return prefix_hit_len;
}

/**
 * exist/delete 等待接口
 */
NDS_ERROR_T nds_kv_ctl_wait(uint64_t task_id, uint32_t total_blocks)
{
    std::vector<uint16_t> target_cids = fetch_cids(task_id);
    if (target_cids.empty()) { return TASKIDNOTFIND; }

    std::vector<uint8_t> hit_map(total_blocks, 0);
    LOG_D("[DEBUG] TASK: %" PRIu64 ", Start Wait", task_id);

    // 2. 同步轮询每个 CID 的状态，因为 CID 是全局分配的且写向同一个 base 地址，所以直接轮询即可
    for (uint16_t cid : target_cids) {
        NDS_ERROR_T ret = wait_for_io_completion(cid, total_blocks, hit_map);
        if (ret == TIMEOUTRETRY || ret == SENDRETRYFAILED || ret == IPBECLEANED ||
            ret == POLLCQERROR) {
            process_cid_metadata(cid, hit_map, total_blocks, true);
        }
    }
    int prefix_hit_len = calculate_prefix_hit_len(hit_map, total_blocks);
    return prefix_hit_len;
}

uint64_t nds_kv_exist(NDS_REQ *req)
{
    if (!req) { return -1; }
    IPC_data_block ipc_block;
    uint32_t ret = 0;
    memset(&ipc_block, 0, sizeof(IPC_data_block));

    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.op_type = IS_EXIST;
    ipc_block.nsid = req->nsid;
    // ipc_block.access_control = req->access_control;
    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
    }

    ret = nds_kv_ctl_send_scatter_distributed(&ipc_block);
    if (ret != 0) {
        return -1;  // 请求发送失败
    }
    return nds_kv_ctl_wait(req->task_id, req->addr_num);
}

NDS_ERROR_T nds_kv_delete(NDS_REQ *req)
{
    if (!req) { return NDS_FAILED; }
    IPC_data_block ipc_block;
    uint32_t ret = 0;
    memset(&ipc_block, 0, sizeof(IPC_data_block));

    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.op_type = KV_DELETE;
    ipc_block.nsid = req->nsid;
    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
    }

    ret = nds_kv_ctl_send_scatter_distributed(&ipc_block);
    if (ret != 0) {
        return -1;  // 请求发送失败
    }
    return nds_kv_ctl_wait(req->task_id, req->addr_num);
}

/* TODO: reqid+layerid--> task_id 通信模块不感知模型和推理框架 */
NDS_ERROR_T nds_kv_get(NDS_REQ *req)
{
    if (!req) { return NDS_FAILED; }
    IPC_data_block ipc_block;
    memset(&ipc_block, 0, sizeof(IPC_data_block));
    ipc_block.op_type = KV_LOAD;
    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.mr_id = req->mr_id;
    ipc_block.nsid = req->nsid;
    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
        ipc_block.addr_pair_list[i].offset = req->addr_pair_list[i].offset;
        ipc_block.addr_pair_list[i].length = req->addr_pair_list[i].length;
        ipc_block.addr_pair_list[i].hbm_addr = req->addr_pair_list[i].hbm_addr;
    }
    if (req->addr_num == 1) {
        ipc_block.op_type = KV_LOAD;
        return nds_kv_io_send_scatter_distributed_single(&ipc_block);
    } else {
        ipc_block.op_type = KV_LOAD_BATCH;
        return nds_kv_io_send_scatter_distributed(&ipc_block);
    }
}

NDS_ERROR_T nds_kv_put(NDS_REQ *req)
{
    if (!req) { return NDS_FAILED; }
    IPC_data_block ipc_block;
    memset(&ipc_block, 0, sizeof(IPC_data_block));
    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.mr_id = req->mr_id;
    ipc_block.nsid = req->nsid;

    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
        ipc_block.addr_pair_list[i].offset = req->addr_pair_list[i].offset;
        ipc_block.addr_pair_list[i].length = req->addr_pair_list[i].length;
        ipc_block.addr_pair_list[i].hbm_addr = req->addr_pair_list[i].hbm_addr;
    }
    if (req->addr_num == 1) {
        ipc_block.op_type = KV_WRITE;
        return nds_kv_io_send_scatter_distributed_single(&ipc_block);
    } else {
        ipc_block.op_type = KV_WRITE_BATCH;
        return nds_kv_io_send_scatter_distributed(&ipc_block);
    }
}

NDS_ERROR_T nds_kv_get_single(NDS_REQ *req)
{
    if (!req) { return NDS_FAILED; }
    if (req->addr_num != 1) { return NDS_FAILED; }
    IPC_data_block ipc_block;
    memset(&ipc_block, 0, sizeof(IPC_data_block));
    ipc_block.op_type = KV_LOAD;
    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.mr_id = req->mr_id;
    ipc_block.nsid = req->nsid;
    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
        ipc_block.addr_pair_list[i].offset = req->addr_pair_list[i].offset;
        ipc_block.addr_pair_list[i].length = req->addr_pair_list[i].length;
        ipc_block.addr_pair_list[i].hbm_addr = req->addr_pair_list[i].hbm_addr;
    }
    return nds_kv_io_send_scatter_distributed_single(&ipc_block);
}

NDS_ERROR_T nds_kv_put_single(NDS_REQ *req)
{
    if (!req) { return NDS_FAILED; }
    if (req->addr_num != 1) { return NDS_FAILED; }
    IPC_data_block ipc_block;
    memset(&ipc_block, 0, sizeof(IPC_data_block));
    ipc_block.op_type = KV_WRITE;
    ipc_block.task_id = req->task_id;
    ipc_block.addr_num = req->addr_num;
    ipc_block.mr_id = req->mr_id;
    ipc_block.nsid = req->nsid;
    for (int i = 0; i < req->addr_num; i++) {
        ipc_block.addr_pair_list[i].block_id.key_low = req->addr_pair_list[i].block_id.key_low;
        ipc_block.addr_pair_list[i].block_id.key_high = req->addr_pair_list[i].block_id.key_high;
        ipc_block.addr_pair_list[i].offset = req->addr_pair_list[i].offset;
        ipc_block.addr_pair_list[i].length = req->addr_pair_list[i].length;
        ipc_block.addr_pair_list[i].hbm_addr = req->addr_pair_list[i].hbm_addr;
    }
    return nds_kv_io_send_scatter_distributed_single(&ipc_block);
}
