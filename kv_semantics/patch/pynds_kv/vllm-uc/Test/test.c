// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "NDS_include/NDS_client.h"

#define TEST_LAYER_NUM 1
#define N 1
#define ONE_LAYER_CACHE_SIZE (64 * 1024 * 1024)  // 64MB
#define BLOCK_SIZE_TEST 128 * 1024

volatile int keep_running = 1;

uint32_t checksum_sum(uint8_t *data, size_t length)
{
    uint32_t checksum = 0;
    for (size_t i = 0; i < length; i++) { checksum += data[i]; }
    return checksum;
}

void fill_test_data(uint64_t addr, size_t length)
{
    uint8_t *ptr = (uint8_t *)addr;
    memset(ptr, 'A', length);
}

int main()
{
    NDS_ERROR_T ret;
    uint32_t check_val;

    // 1. 准备 kv_mr 数组
    struct kv_mr *my_mrs = (struct kv_mr *)malloc(sizeof(struct kv_mr) * TEST_LAYER_NUM);
    if (!my_mrs) return -1;

    for (int i = 0; i < TEST_LAYER_NUM; i++) {
        void *ptr = NULL;
        if (posix_memalign(&ptr, 4096, ONE_LAYER_CACHE_SIZE) != 0) { return -1; }
        my_mrs[i].addr = (uint64_t)ptr;
        my_mrs[i].len = ONE_LAYER_CACHE_SIZE;

        printf("[Test] Layer %d allocated at 0x%lx\n", i, my_mrs[i].addr);
    }

    // 2. 初始化 SDK
    printf("[Test] Calling nds_init...\n");
    ret = nds_init((void **)&my_mrs, TEST_LAYER_NUM);
    if (ret != NDS_SUCCESS) {
        printf("SDK Init Failed\n");
        return -1;
    }
    printf("SDK Init success\n");

    while (keep_running) {
        printf("正在维持连接，处理数据中...\n");
        usleep(3000000);
    }

    uint16_t target_layer = 0;
    uint32_t final_check;
    NDS_REQ req;
    memset(&req, 0, sizeof(NDS_REQ));
    req.task_id = 0x123456;
    req.mr_id = target_layer;
    req.addr_num = N;
    for (int i = 0; i < N; i++) {
        req.addr_pair_list[i].block_id.key_low = 100 + i;
        req.addr_pair_list[i].block_id.key_high = 0;
        req.addr_pair_list[i].length = BLOCK_SIZE_TEST;
        req.addr_pair_list[i].offset = i * BLOCK_SIZE_TEST;
        req.addr_pair_list[i].hbm_addr =
            (uint64_t)my_mrs[target_layer].addr + (uint64_t)i * BLOCK_SIZE_TEST;
        printf("[Test] io(%d) hbm addr(%p)\n", i, (uint64_t *)req.addr_pair_list[i].hbm_addr);
    }
    // STEP 1: 在本地内存填入数据并 Put 到远端
    fill_test_data(my_mrs[target_layer].addr, BLOCK_SIZE_TEST * N);
    check_val = checksum_sum((uint8_t *)my_mrs[target_layer].addr, BLOCK_SIZE_TEST * N);
    printf("[Test] Original Checksum: %u\n", check_val);
    printf("[Test] Executing nds_kv_put (Synchronous)...\n");
    ret = nds_kv_put(&req);
    if (ret != NDS_SUCCESS) {
        printf("Put failed!\n");
        goto cleanup;
    }
    printf("[Test] Put Success.\n");
    // STEP 2: 修改本地内存（清零），尝试从远端 Get 回来
    memset((void *)my_mrs[target_layer].addr, 0, BLOCK_SIZE_TEST * N);
    ret = nds_kv_get(&req);
    if (ret != NDS_SUCCESS) {
        printf("Get failed!\n");
        goto cleanup;
    }
    // STEP 3: 校验数据一致性
    final_check = checksum_sum((uint8_t *)my_mrs[target_layer].addr, BLOCK_SIZE_TEST * N);
    printf("[Test] Final Checksum: %u\n", final_check);
    if (final_check == check_val) {
        printf("[Test] SUCCESS: Data integrity verified!\n");
    } else {
        printf("[Test] FAILED: Data corruption detected!\n");
    }

cleanup:
    for (int i = 0; i < TEST_LAYER_NUM; i++) { free((void *)my_mrs[i].addr); }
    free(my_mrs);
    return 0;
}
