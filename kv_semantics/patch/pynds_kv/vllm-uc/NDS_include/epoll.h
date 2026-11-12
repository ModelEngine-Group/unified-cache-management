// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef NDS_INCLUDE_EPOLL_H_
#define NDS_INCLUDE_EPOLL_H_

#include <sys/epoll.h>
#include "NDS_client.h"

// epoll 事件回调函数声明
typedef void (*conn_event_handler_t)(int fd, uint32_t events, void *arg);

// epoll 监听线程参数
typedef struct {
    int epfd;
    volatile bool running;
} epoll_thread_args_t;

typedef struct {
    struct rdma_connection *conn;
    bool timer_switch;
} reconnect_args_t;

extern epoll_thread_args_t g_epoll_args;
extern conn_event_handler_t g_event_handler;

extern NDS_ERROR_T nds_epoll_add_conn(struct rdma_connection *conn);

NDS_ERROR_T handle_socket_event(struct rdma_connection *conn, uint32_t events);

extern void *epoll_listener_thread(void *arg);

extern void nds_epoll_close();

extern NDS_ERROR_T epoll_remove_fd(int epfd, int fd);

extern void *reconnect_timer_thread(void *arg);

extern void start_reconnect_timer(struct rdma_connection *conn);

bool reconnect(struct rdma_connection *conn);

extern bool try_destroy_connection(struct rdma_connection *conn);

extern bool try_reconnect(struct rdma_connection *conn);

#endif  // NDS_INCLUDE_EPOLL_H_
