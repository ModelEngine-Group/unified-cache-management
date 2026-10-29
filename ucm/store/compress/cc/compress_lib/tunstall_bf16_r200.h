#ifndef TUNSTALL_BF16_R200_H
#define TUNSTALL_BF16_R200_H

#include <cstddef>
#include <cstdint>
#include "tunstall.h"

enum class R200PayloadMode : uint8_t {
    TUNSTALL,
    FP8_FALLBACK,
    INVALID,
};

// p_src 里有 n_bf16 个 BF16
// 把它们压缩到 p_dst 里，占 n_bf16 字节
// 压缩率固定为 2x
extern int TunstallCompressBF16(uint8_t* p_dst,  // 需要 n_bf16*2 字节, 但压缩后起始只有 n_bf16 字节
                                const uint16_t* p_src,  //      n_bf16*2 字节
                                size_t n_bf16);

// p_src 里是压缩流，占 n_bf16 字节
// 把它们解压到 p_dst ，解压出来 n_bf16 个 BF16
extern int TunstallDecompressBF16(uint16_t* p_dst,       //      n_bf16*2 字节
                                  const uint8_t* p_src,  //      n_bf16 字节
                                  size_t n_bf16);

// In-place BF16 decompression.
// Before: p_data[0:n_bf16] contains compressed bytes and p_data[0:n_bf16*2] is writable.
// After : p_data[0:n_bf16*2] contains n_bf16 BF16 values.
extern int TunstallDecompressBF16Inplace(
    uint8_t* p_data,  // 可访问范围是 n_bf16*2 字节，解压前的数据占据 p_data[0:n_bf16]
                      // 字节，解压后占据 p_data[0:n_bf16*2] 字节
    size_t n_bf16);

// Infer the decoder path from an R200 payload. payload_bytes equals n_bf16 because
// every R200 payload occupies one byte per BF16 value.
extern R200PayloadMode TunstallGetBF16R200Mode(const uint8_t* p_src, size_t payload_bytes) noexcept;

#endif  // TUNSTALL_BF16_R200_H
