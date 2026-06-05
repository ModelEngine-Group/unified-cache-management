/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CANN_HIXL_SRC_HIXL_OPS_HIXL_KERNEL_HIXL_SEND_H_
#define CANN_HIXL_SRC_HIXL_OPS_HIXL_KERNEL_HIXL_SEND_H_

#include <cstdint>
#include "hcomm/hcomm_res_defs.h"

struct HixlSendParam {
  ThreadHandle thread;
  ChannelHandle channel;
  const void *local_src;
  uint64_t len;
};

struct HixlRecvParam {
  ThreadHandle thread;
  ChannelHandle channel;
  void *local_dst;
  uint64_t dst_capacity;
  uint64_t *received_len;
  uint32_t timeout_ms;
};

#ifdef __cplusplus
extern "C" {
#endif

uint32_t HixlSend(HixlSendParam *param);

uint32_t HixlRecv(HixlRecvParam *param);

#ifdef __cplusplus
}
#endif

#endif  // CANN_HIXL_SRC_HIXL_OPS_HIXL_KERNEL_HIXL_SEND_H_
