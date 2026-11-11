// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include "../NDS_include/utils.h"

/**
 * @brief 解析形如 "reconnection_interval = 1, 2, 3, 4" 的配置行
 *
 * @param input 包含等号的配置行字符串
 * @param out_intervals 输出参数，返回解析出的整数数组指针
 * @return 解析出的整数个数，失败返回 -1
 */
int parse_interval_list(const char *input, int **out_intervals)
{
    if (!input || !out_intervals) return -1;
    const char *p = strchr(input, '=');
    if (!p) return -1;
    p++;
    while (*p == ' ') p++;

    char buf[256];
    strncpy(buf, p, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *comment = strchr(buf, '#');
    if (comment) *comment = '\0';
    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == ' ' || buf[len - 1] == '\t' || buf[len - 1] == '\n' ||
                       buf[len - 1] == '\r')) {
        buf[--len] = '\0';
    }

    int count = 0;
    const char *tmp = buf;
    while (*tmp) {
        if (*tmp == ',') count++;
        tmp++;
    }
    count++;
    if (count <= 0) return -1;

    int *intervals = (int *)malloc(count * sizeof(int));
    if (!intervals) return -1;

    int idx = 0;
    char *token = strtok(buf, ", ");
    while (token && idx < count) {
        intervals[idx++] = atoi(token);
        token = strtok(NULL, ", ");
    }

    if (idx == 0) {
        free(intervals);
        return -1;
    }

    *out_intervals = intervals;
    return idx;
}

/**
 * @brief 保存IP连接信息到连接结构体
 *
 * @param ip_index 连接索引
 * @param ip IP配置信息
 */
void save_ip_information(int ip_index, gids_ip_pair_t ip)
{
    snprintf(conn[ip_index].ip.local_ip, sizeof(conn[ip_index].ip.local_ip), "%s", ip.local_ip);
    snprintf(conn[ip_index].ip.server_ip, sizeof(conn[ip_index].ip.server_ip), "%s", ip.server_ip);
    conn[ip_index].ip.server_port = ip.server_port;
    conn[ip_index].ip.roce_idx = ip.roce_idx;
    conn[ip_index].ip.roce_port = ip.roce_port;
    conn[ip_index].ip.dpu_id = ip.dpu_id;
}

void save_cluster_ip_information(struct rdma_connection *conn, local_ip_pair_t localip,
                                 const std::string &dpu_ip)
{
    snprintf(conn->ip.local_ip, sizeof(conn->ip.local_ip), "%s", localip.local_ip);
    snprintf(conn->ip.server_ip, sizeof(conn->ip.server_ip), "%s", dpu_ip.c_str());
    conn->ip.server_port = localip.server_port;
    conn->ip.roce_idx = localip.roce_idx;
    conn->ip.roce_port = localip.roce_port;
}

/**
 * @brief 判断IO错误码是否为可重试错误
 *
 * @param code IO错误码
 * @return true 可重试，false 不可重试
 */
bool is_io_retry_error(int code)
{
    return (code == IO_ERR_INTERNAL || code == IO_ERR_WRITE_FAULT ||
            code == IO_ERR_UNRECOVERED_READ || code == IO_ERR_IO_TIMEOUT ||
            code == IO_ERR_RESOURCE_BUSY);
}

// 返回可用的链接下标，没找到返回-1
int FindAvailableConn()
{
    int conn_idx = -1;
    for (int i = 0; i < IP_NUM_T; i++) {
        if (!conn[i].is_used && conn[i].is_destroyed) {
            conn_idx = i;
            break;
        }
    }
    return conn_idx;
}