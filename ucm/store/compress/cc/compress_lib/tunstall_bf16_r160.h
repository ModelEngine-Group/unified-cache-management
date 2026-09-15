#ifndef TUNSTALL_BF16_R160_H
#define TUNSTALL_BF16_R160_H

#include <cstddef>
#include <cstdint>
#include "tunstall.h"

enum R160DecodeError {
    R_ERR_R160_STREAM_SIZE = 100,
    R_ERR_R160_E8_TAG = 101,
    R_ERR_R160_E8_METADATA = 102,
    R_ERR_R160_E8_EXPANSION = 103,
};

enum class R160PayloadMode : uint8_t {
    HIGH_PRECISION = 0,
    QUANTIZED = 1,
    INVALID = 0xff,
};

// Fixed-budget BF16 codec:
//   * output size is supplied by the caller and must be between 1/2 and 5/8 of
//     the BF16 input bytes; the pipeline uses the nominal 5/8 size rounded down
//     to a 4 KiB boundary;
//   * high-precision mode preserves sign, full E8 and M6:M2 for the whole shard;
//     spare payload space first stores an M1 prefix, then an M0 prefix after all M1;
//   * if the full-E8 stream cannot meet the fixed budget, the whole shard falls back to
//     E5M2-like quantization; remaining bytes preserve M4 first and then an M3 prefix;
//   * mode selection is once per shard, never per 32-value packing block.
// High-precision payload:
//   SM4 | M32 | dynamic Tunstall E8 | optional low bits | padding.
// Within each 32-value tile, SM4, M32 and E8 use the R200 lane pairing:
// [0,8], [1,9], ... and [16,24], ... . One M32 byte holds the same lane's
// M3:M2 bits for the four 8-value output groups. In the optional region, the
// prefix that preserves both M1 and M0 is stored as lane-major M10 tiles; the
// remaining M1-only groups use one bit-plane byte per eight values.
// Quantized payload:
//   E5M2-like bytes | optional M4 prefix | optional M3 prefix.
// The E8 byte at offset 3*N/4 is TS_MODE_DYNAMIC in high-precision mode. The quantized
// path clears the low bit at the same offset. More than 64 distinct E8 values or an E8
// stream that exceeds the budget causes whole-shard quantized fallback. Optional
// M1/M0 group counts are inferred from the E8 payload length and fixed total size.
//
// *p_dst_len is both the requested payload size and destination capacity on input,
// and remains the fixed payload size on success.
int TunstallCompressBF16R160(uint8_t* p_dst, size_t* p_dst_len, const uint16_t* p_src,
                             size_t n_bf16);

// Out-of-place decompression for codec-level benchmarks. p_src and p_dst must not overlap.
int TunstallDecompressBF16R160(const uint8_t* p_src, size_t src_len, uint16_t* p_dst,
                               size_t n_bf16);

// In-place decompression. p_data is uint16_t-aligned, initially contains src_len bytes of
// R160 payload, and has capacity for n_bf16 BF16 values.
int TunstallDecompressBF16R160Inplace(uint8_t* p_data, size_t n_bf16, size_t src_len);

// Infer the shard-level mode from the fixed discriminator in a headerless payload.
R160PayloadMode TunstallGetBF16R160Mode(const uint8_t* p_src, size_t src_len, size_t n_bf16);

#endif  // TUNSTALL_BF16_R160_H
