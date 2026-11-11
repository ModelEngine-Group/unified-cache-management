// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/epoll.h"
#define TIME_OUT 1000
#define MAX_EVENTS 512

epoll_thread_args_t g_epoll_args;
conn_event_handler_t g_event_handler;

// 注册连接的文件描述符到 epoll
int nds_epoll_add_conn(struct rdma_connection *conn)
{
    struct epoll_event ev = {};
    ev.events = EPOLLIN | EPOLLET;
    ev.data.ptr = conn;

    if (epoll_ctl(g_epoll_args.epfd, EPOLL_CTL_ADD, conn->clnt_fd, &ev) < 0) {
        LOG_E("epoll_ctl add conn %d failed: %s", conn->conn_idx, strerror(errno));
        return NDS_FAILED;
    }
    return NDS_SUCCESS;
}

int parse_message_header(char *buf, uint8_t *cmd)
{
    *cmd = buf[5];
    return NDS_SUCCESS;
}

int handle_disconnect_request(struct rdma_connection *conn, char *buf)
{
    uint32_t local_qpn = *(uint32_t *)(buf + 16);
    uint32_t remote_qpn = *(uint32_t *)(buf + 20);
    LOG_I("Disconnect request: local_qpn=%u, remote_qpn=%u", local_qpn, remote_qpn);
    if (try_destroy_connection(conn)) { start_reconnect_timer(conn); }
    return NDS_SUCCESS;
}

int handle_socket_event(struct rdma_connection *conn, uint32_t events)
{
    char buf[1024];
    if (events & EPOLLIN) {
        int n = recv(conn->clnt_fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (n < 0) LOG_E("recv error: %s", strerror(errno));
            return NDS_FAILED;
        }
        uint8_t cmd;
        parse_message_header(buf, &cmd);
        LOG_E("Received cmd=%u from conn %u", cmd, conn->conn_idx);
        if (cmd == 4) {
            handle_disconnect_request(conn, buf);
            return NDS_SUCCESS;
        }
    }
    return NDS_SUCCESS;
}

void *epoll_listener_thread(void *arg)
{
    (void)arg;
    const int max_events = MAX_EVENTS;
    struct epoll_event events[max_events];
    while (g_epoll_args.running) {
        int nfds = epoll_wait(g_epoll_args.epfd, events, max_events, TIME_OUT);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            LOG_E("epoll_wait error: %s", strerror(errno));
            break;
        }
        for (int i = 0; i < nfds; i++) {
            struct rdma_connection *conn = (struct rdma_connection *)events[i].data.ptr;
            handle_socket_event(conn, events[i].events);
        }
    }
    return NULL;
}

// 关闭 epoll
void nds_epoll_close()
{
    g_epoll_args.running = false;
    if (g_epoll_args.epfd >= 0) {
        close(g_epoll_args.epfd);
        g_epoll_args.epfd = -1;
    }
}

NDS_ERROR_T epoll_remove_fd(int epfd, int fd)
{
    if (fd > 0) {
        if (epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL) < 0) {
            if (errno == EBADF || errno == ENOENT) {
                LOG_W("fd %d already invalid or not in epoll", fd);
                return NDS_SUCCESS;
            }
            LOG_E("epoll_ctl DEL fd %d failed: %s", fd, strerror(errno));
            return NDS_FAILED;
        }
    }
    return NDS_SUCCESS;
}

bool reconnect(struct rdma_connection *conn)
{
    pthread_mutex_lock(&conn->reconnection_lock);
    if (conn->is_reconnecting) {
        pthread_mutex_unlock(&conn->reconnection_lock);
        LOG_W("Reconnection already in progress for conn %u", conn->conn_idx);
        return false;
    }
    conn->is_reconnecting = true;
    pthread_mutex_unlock(&conn->reconnection_lock);

    int conn_idx = conn->conn_idx;
    // 断链前已经把此链接的cid都清理了，重链前不用清理了
    LOG_D("[RECONN_DEBUG] conn %d: start reconnection", conn_idx);

    bool success = true;
    if (init_rdma_device_and_mr(conn, conn->ip.local_ip, g_kv_cache, g_mr_num) != NDS_SUCCESS) {
        success = false;
    } else if (setup_tcp_connection(conn, conn->ip.local_ip, conn->ip.server_ip,
                                    conn->ip.server_port) != NDS_SUCCESS) {
        destroy_connection_locked(conn);
        success = false;
    } else if (setup_rdma_connection(conn, conn->ip.roce_idx, conn->ip.roce_port) != NDS_SUCCESS) {
        destroy_connection_locked(conn);
        success = false;
    }
    pthread_mutex_lock(&conn->reconnection_lock);
    conn->is_reconnecting = false;
    pthread_mutex_unlock(&conn->reconnection_lock);
    if (conn->reconnection_count < g_cfg.interval_count - 1) { conn->reconnection_count++; }
    if (!success) { return false; }
    conn->online = true;
    LOG_I("[RECONNECT] reconnect conn %d success: local=%s -> dpu=%s", conn->conn_idx,
          conn->ip.local_ip, conn->ip.server_ip);
    return true;
}

void *reconnect_timer_thread(void *arg)
{
    reconnect_args_t *args = (reconnect_args_t *)arg;
    struct rdma_connection *conn = args->conn;
    bool timer_switch = args->timer_switch;
    free(args);
    if (timer_switch) {
        LOG_W("Reconnecting conn %u th %d sleep %d", conn->conn_idx, conn->reconnection_count,
              g_cfg.interval[conn->reconnection_count]);
        sleep(g_cfg.interval[conn->reconnection_count]);
    }
    try_reconnect(conn);
    LOG_D("Reconnecting conn %u", conn->conn_idx);
    pthread_mutex_lock(&conn->start_timer_lock);
    conn->is_starting_timer = false;
    pthread_mutex_unlock(&conn->start_timer_lock);
    return NULL;
}

void start_reconnect_timer(struct rdma_connection *conn)
{
    pthread_mutex_lock(&conn->start_timer_lock);
    if (conn->is_starting_timer) {
        pthread_mutex_unlock(&conn->start_timer_lock);
        LOG_D("Conn %u is_starting_timer, skip start timer", conn->conn_idx);
        return;
    }
    conn->is_starting_timer = true;
    pthread_mutex_unlock(&conn->start_timer_lock);
    reconnect_args_t *args = (reconnect_args_t *)malloc(sizeof(reconnect_args_t));
    args->conn = conn;
    args->timer_switch = g_cfg.timer_switch;
    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&tid, &attr, reconnect_timer_thread, args);
    pthread_attr_destroy(&attr);
}

bool try_destroy_connection(struct rdma_connection *conn) { return destroy_connection(conn); }

bool try_reconnect(struct rdma_connection *conn)
{
    pthread_mutex_lock(&conn->destroy_connection_lock);
    if (conn->is_destroying || conn->online || !conn->is_used) {
        LOG_W("Conn %u is destroying=%d or online=%d, skip reconnect", conn->conn_idx,
              conn->is_destroying.load(), conn->online.load());
        pthread_mutex_unlock(&conn->destroy_connection_lock);
        return false;
    }
    pthread_mutex_unlock(&conn->destroy_connection_lock);

    if (!reconnect(conn)) { return false; }
    nds_epoll_add_conn(conn);
    return true;
}
