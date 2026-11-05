// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>

#define API_EXPORT __attribute__((visibility("default")))

#ifdef __cplusplus
#include <vector>

// 80-bit block_key（SDK block_key：key_low[7:0] + key_high[9:8]，共 10 字节）：
// 所有读写查接口的 key 以 (keyLow, keyHigh) 形式传递，keyHigh 为高 16 位。
// wrapper 默认对整个 80-bit key 做 xxhash 打散（conf key_scatter=0 可关闭），
// 使盘框分盘/分链均匀；读写查经同一变换，语义一致。
namespace NDS {
API_EXPORT int32_t init(void *memAddr, uint64_t length, const char *path_nds_config = nullptr);

API_EXPORT int32_t initMulti(std::vector<void *> memAddrs, std::vector<uint64_t> lengths,
                             const char *path_nds_config = nullptr);

API_EXPORT int32_t isExists(std::vector<uint64_t> blockIdLows, std::vector<uint16_t> blockIdHighs,
                            uint32_t nsid = 0);

API_EXPORT int32_t get(uint64_t blockIdLow, uint16_t blockIdHigh, uint8_t *blockAddr, size_t offset,
                       size_t length, uint32_t nsid);

API_EXPORT int32_t batchGet(std::vector<uint64_t> blockIdLows, std::vector<uint16_t> blockIdHighs,
                            std::vector<uint8_t *> blockAddrs, std::vector<size_t> offsets,
                            std::vector<size_t> lengths, std::vector<uint32_t> nsids);

API_EXPORT int32_t put(uint64_t blockIdLow, uint16_t blockIdHigh, uint8_t *blockAddr, size_t offset,
                       size_t length, uint32_t nsid);

API_EXPORT int32_t batchPut(std::vector<uint64_t> blockIdLows, std::vector<uint16_t> blockIdHighs,
                            std::vector<uint8_t *> blockAddrs, std::vector<size_t> offsets,
                            std::vector<size_t> lengths, std::vector<uint32_t> nsids);
}  // namespace NDS
#endif

extern "C" {
API_EXPORT int32_t c_init(void *memAddr, uint64_t length, const char *path_nds_config);

API_EXPORT int32_t c_init_multi(const void *const *memAddrs, const uint64_t *lengths, size_t count,
                                const char *path_nds_config);

API_EXPORT int32_t c_isExists(const uint64_t *blockIds, size_t count, uint32_t nsid);

API_EXPORT int32_t c_get(uint64_t blockId, uint8_t *blockAddr, size_t offset, size_t length,
                         uint32_t nsid);

API_EXPORT int32_t c_batchGet(const uint64_t *blockIds, uint8_t **blockAddrs, const size_t *offsets,
                              const size_t *lengths, const uint32_t *nsids, size_t count);

API_EXPORT int32_t c_put(uint64_t blockId, uint8_t *blockAddr, size_t offset, size_t length,
                         uint32_t nsid);

API_EXPORT int32_t c_batchPut(const uint64_t *blockIds, uint8_t **blockAddrs, const size_t *offsets,
                              const size_t *lengths, const uint32_t *nsids, size_t count);
}
