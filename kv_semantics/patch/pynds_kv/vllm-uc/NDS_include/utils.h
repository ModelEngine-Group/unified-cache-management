// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#ifndef NDS_INCLUDE_UTILS_H_
#define NDS_INCLUDE_UTILS_H_

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "NDS_client.h"
#include "cluster.h"

extern int parse_interval_list(const char *input, int **out_intervals);
extern void save_ip_information(int ip_index, gids_ip_pair_t ip);
extern void save_cluster_ip_information(struct rdma_connection *conn, local_ip_pair_t localip,
                                        const std::string &dpu_ip);
extern bool is_io_retry_error(int code);
extern int FindAvailableConn();

#endif  // NDS_INCLUDE_UTILS_H_
