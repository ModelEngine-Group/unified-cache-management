// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/keepalive.h"
#include <sys/time.h>
#include "../NDS_include/epoll.h"

#define HB_SEM_TIMEOUT_SEC 15

/**
 * @brief KeepAlive心跳保活线程
 *
 * 功能：定期向所有连接发送心跳包，检测连接是否存活
 *
 * @param arg 线程参数（未使用）
 */
void *keepalive_thread(void *arg)
{
    (void)arg;
    uint32_t keepalive_interval = g_cfg.heartbeat_interval;

    while (!exit_flag) {
        // 记录没条链接上一秒的iocount
        for (int ip_index = 0; ip_index < IP_NUM_T; ip_index++) {
            conn[ip_index].old_io_count = conn[ip_index].io_count;
        }
        sleep(keepalive_interval);
        for (int ip_index = 0; ip_index < IP_NUM_T; ip_index++) {
            if (!conn[ip_index].is_used) continue;
            if (conn[ip_index].conn_idx < 0) continue;
            if (conn[ip_index].old_io_count != conn[ip_index].io_count) continue;
            pthread_mutex_lock(&conn[ip_index].destroy_connection_lock);
            bool is_online = conn[ip_index].online && !conn[ip_index].is_destroying;
            pthread_mutex_unlock(&conn[ip_index].destroy_connection_lock);

            if (!is_online) {
                LOG_D("start_reconnect_timer ip_index: %d, (online=%d, is_destroying=%d)", ip_index,
                      conn[ip_index].online.load(), conn[ip_index].is_destroying.load());
                start_reconnect_timer(&conn[ip_index]);
                continue;
            }
            // 若is_used=true, 1秒内无io, 且在线，才发心跳包
            NDS_ERROR_T trans_res = send_heart_beat(&conn[ip_index]);
            if (trans_res == NDS_TCP_CONN_ERR) {
                LOG_W("[Heart_Beat] send_heart_beat error.");
                bool destroyed = try_destroy_connection(&conn[ip_index]);
                if (destroyed) { start_reconnect_timer(&conn[ip_index]); }
            }
        }
    }
    return NULL;
}

NDS_ERROR_T send_heart_beat(struct rdma_connection *conn)
{
    if (!conn->online || conn->is_destroying || conn->is_reconnecting) {
        LOG_I("[Heart_Beat] Skip: online=%d is_destroying=%d is_reconnecting=%d",
              conn->online.load(), conn->is_destroying.load(), conn->is_reconnecting.load());
        return NDS_FAILED;
    }

    data_heart_beat keepalive_msg;
    memset(&keepalive_msg, 0, sizeof(data_heart_beat));
    int my_cid = -1;
    LOG_D("acquire_cid");
    int retry = 0;
    while ((my_cid = acquire_cid()) == -1) {
        if (++retry > 100000) {
            LOG_E("[Heart_Beat] acquire_cid failed after %d retries, conn %d", retry,
                  conn->conn_idx);
            return NDS_FAILED;
        }
        if (!conn->online || conn->is_destroying) {
            LOG_D("[Heart_Beat] acquire_cid abort: online=%d is_destroying=%d", conn->online.load(),
                  conn->is_destroying.load());
            return NDS_FAILED;
        }
        cpu_relax();
    }
    LOG_D("[Heart_Beat] acquired cid=%d", my_cid);

    if (!conn->online || conn->is_destroying || conn->is_reconnecting) {
        LOG_D(
            "[Heart_Beat] CID acquired but conn state changed: online=%d is_destroying=%d "
            "is_reconnecting=%d",
            conn->online.load(), conn->is_destroying.load(), conn->is_reconnecting.load());
        release_cid(my_cid);
        return NDS_FAILED;
    }

    keepalive_msg.opcode = Heart_Beat;
    keepalive_msg.cid = my_cid;
    g_cid_metadata[my_cid].ip_index = conn->conn_idx;
    g_cid_metadata[my_cid].qp_index = (conn->kv_cap_info.adminq_depth != 0);
    g_cid_metadata[my_cid].optype = HEARTBEAT;
    g_cid_metadata[my_cid].tmp_conn = conn;
    uint64_t old = __sync_fetch_and_add(&counter, 1);
    int qpID = (conn->kv_cap_info.adminq_depth != 0) + old % g_cfg.io_qp_num;
    g_cid_metadata[my_cid].qp_index = qpID;
    LOG_D("[Heart_Beat_SEND] Conn:%d | QP:%d | CID:%d", conn->conn_idx, qpID, my_cid);

    NDS_ERROR_T trans_res = transfer_data_heartbeat(conn, &keepalive_msg, qpID);
    if (trans_res != NDS_SUCCESS) {
        LOG_W("[Heart_Beat] transfer failed: %d", trans_res);
        g_cid_metadata[my_cid].ip_index = -1;
        release_cid(my_cid);
        return trans_res;
    }
    LOG_D("KeepAlive sent [Heart_Beat] for conn %d success", conn->conn_idx);
    __sync_fetch_and_add(&conn->connSendCounter, 1);
    return NDS_SUCCESS;
}

NDS_ERROR_T transfer_data_heartbeat(struct rdma_connection *conn, data_heart_beat *message,
                                    int qpID)
{
    int ipIdx = conn->conn_idx;
    QPContext &ctx = g_qp_ctx[ipIdx][qpID];
    data_heart_beat *msg = (data_heart_beat *)message;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += HB_SEM_TIMEOUT_SEC;
    if (sem_timedwait(&ctx.window_sem, &ts) != 0) {
        LOG_W("[HB_XFER] sem_wait timeout conn %d qp %d", ipIdx, qpID);
        return NDS_TCP_CONN_ERR;
    }

    if (conn->send_mr[qpID] == NULL || conn->qps[qpID] == NULL) {
        LOG_W("Conn %d QP %d resources freed after sem_wait (heartbeat), skip", ipIdx, qpID);
        sem_post(&ctx.window_sem);
        return NDS_TCP_CONN_ERR;
    }

    if (!conn->online || conn->is_destroying || conn->is_reconnecting) {
        LOG_W(
            "[HB_XFER] conn state changed after sem_wait: online=%d is_destroying=%d "
            "is_reconnecting=%d",
            conn->online.load(), conn->is_destroying.load(), conn->is_reconnecting.load());
        sem_post(&ctx.window_sem);
        return NDS_FAILED;
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
    pthread_mutex_unlock(&ctx.slot_lock);
    LOG_D("ipIdx %d cid %d qp id %d, get slot id %d, slot size (%d), total send_buf size (%zu)",
          ipIdx, msg->cid, qpID, slot_id, g_cfg.slot_size, g_cfg.send_mr_size);

    if (slot_id == -1) {
        sem_post(&ctx.window_sem);
        return NDS_FAILED;
    }

    cid_to_slot[msg->cid] = slot_id;
    ctx.slot_to_cid[slot_id] = msg->cid;
    size_t paylen = sizeof(data_heart_beat);
    if (paylen > conn->send_mr[qpID]->length) paylen = conn->send_mr[qpID]->length;

    uint8_t *dest_ptr = (uint8_t *)conn->send_buf[qpID] + (slot_id * g_cfg.slot_size);
    memcpy(dest_ptr, message, paylen);

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
        LOG_W("[Heart_Beat] ibv_post_send failed.");
        return NDS_TCP_CONN_ERR;
    }

    return NDS_SUCCESS;
}
