#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "tunstall_bf16.h"    // TunstallCompressDynamic

#define  ENABLE_NEON_DECOMPPRESSION   1
#define  ENABLE_EXTRA_DECOMPRESSION   0
#define  DECOMPRESSION_VERSION        2
#define  MIN_BF16_COUNT               16384


#if ENABLE_NEON_DECOMPPRESSION && (defined(__ARM_NEON) || defined(__aarch64__))
#define NEON_DECOMPRESSION
#include <arm_neon.h>
#endif


#define RET_ERROR_IF(err_code, condition) { int c = (condition); int e = (err_code); if (c) { return e; } }

#define BF16_EXP_MIN (128 - (1 << 4))
#define BF16_EXP_MAX (128 + (1 << 4) - 1)



static uint8_t bf16_to_exp (uint16_t x) {
    uint16_t e = (x >> 7) & 0xFF;
    if (e < BF16_EXP_MIN) e = BF16_EXP_MIN;
    if (e > BF16_EXP_MAX) e = BF16_EXP_MAX;
    return (uint8_t)(e - BF16_EXP_MIN);
}



static uint8_t bf16_sign_mant8 (uint16_t x) {
    return (uint8_t)(((x >> 8) & 0x80) | (x & 0x7F));
}



static uint8_t bf16_pack_extra8 (const uint16_t *p_src) {
    uint8_t extra8 = 0;
    for (size_t lane = 0; lane < 8; lane++) {
        extra8 |= (uint8_t)(((p_src[lane] >> 3) & 0x01) << (7 - lane));
    }
    return extra8;
}



static void compress_bf16_to_fp8 (
    uint8_t        *p_dst,
    const uint16_t *p_src,
    size_t          n_bf16
) {
    for (size_t i = 0; i < n_bf16; i++) {
        uint16_t x = p_src[i];
        uint8_t exp5 = bf16_to_exp(x);
        uint8_t sign = (uint8_t)((x >> 15) & 0x01);
        uint8_t mant2 = (uint8_t)((x >> 5) & 0x03);
        p_dst[i] = (uint8_t)((exp5 << 3) | (sign << 2) | mant2);
    }
}



static void decompress_fp8_to_bf16 (
    uint16_t      *p_dst,
    const uint8_t *p_src,
    size_t         n_bf16
) {
    for (size_t i = 0; i < n_bf16; i++) {
        uint16_t fp8 = p_src[i];
        uint16_t exp8  = (uint16_t)((uint16_t)(fp8 >> 3) + BF16_EXP_MIN) << 7;
        uint16_t sign  = (uint16_t)(fp8 & 0x04) << 13;
        uint16_t mant2 = (uint16_t)(fp8 & 0x03) << 5;
        p_dst[i] = sign | exp8 | mant2 | 0x10;
    }
}



// p_src 里有 n_bf16 个 BF16 
// 把它们压缩到 p_dst 里，占 n_bf16 字节
// 压缩率固定为 2x
int TunstallCompressBF16 (
    uint8_t        *p_dst,    // 需要 n_bf16*2 字节, 但压缩后起始只有 n_bf16 字节
    const uint16_t *p_src,    //      n_bf16*2 字节
    size_t          n_bf16
) {
    RET_ERROR_IF(R_ERR_UNSUPPORT, n_bf16 == 0 || n_bf16 > 0xFFFFFFFFU);

    if (n_bf16 < MIN_BF16_COUNT || n_bf16 % 32 != 0) { // 数据量不够多，或者不是 32 的倍数
        compress_bf16_to_fp8(p_dst, p_src, n_bf16);    // fallback 到 FP8 压缩
        p_dst[n_bf16 / 2] = 0xFE;                      // 确保 p_dst[n_bf16/2] != TS_MODE_DYNAMIC, 避免被识别成 tunstall 模式
        return R_TS_OK;
    }

    // stage0: 指数重新排布
    size_t   tunstall_len = (n_bf16 / 2);          // Tunstall 压缩流的预算 = n_bf16 / 2 字节
    uint8_t *p_tunstall = p_dst + (n_bf16 / 2);    // Tunstall 压缩流放在          p_dst[n_bf16/2 : n_bf16]
    uint8_t *p_exp      = p_dst + n_bf16;          // 调整数据排布后的 exp 暂时放在 p_dst[n_bf16 : n_bf16*2]
    for (size_t i = 0; i < n_bf16; i += 16) {
        for (size_t lane = 0; lane < 8; lane++) {
            p_exp[i + (lane << 1)]     = bf16_to_exp(p_src[i + lane]);
            p_exp[i + (lane << 1) + 1] = bf16_to_exp(p_src[i + lane + 8]);
        }
    }

    // stage1: 指数 tunstall 压缩
    int err = TunstallCompressDynamic(p_tunstall, &tunstall_len, p_exp, n_bf16);

    if (err || tunstall_len > (n_bf16 / 2)) {        // tunstall 压缩错误，或者无法保证压缩到 n_bf16 / 2 字节内
        compress_bf16_to_fp8(p_dst, p_src, n_bf16);  // fallback 到 FP8 压缩
        p_dst[n_bf16 / 2] = 0xFE;                    // 确保 p_dst[n_bf16/2] != TS_MODE_DYNAMIC, 避免被识别成 tunstall 模式
        return R_TS_OK;
    }

    // stage2: 符号+3bit尾数打包
    for (size_t i = 0; i < n_bf16; i += 32) {
        for (size_t lane = 0; lane < 8; lane++) {
            uint8_t sm_0  = (uint8_t)(bf16_sign_mant8(p_src[i + lane]) >> 4);
            uint8_t sm_8  = (uint8_t)(bf16_sign_mant8(p_src[i + lane + 8]) >> 4);
            uint8_t sm_16 = (uint8_t)(bf16_sign_mant8(p_src[i + lane + 16]) >> 4);
            uint8_t sm_24 = (uint8_t)(bf16_sign_mant8(p_src[i + lane + 24]) >> 4);
            p_dst[(i >> 1) + (lane << 1)]     = (uint8_t)(sm_0  | (sm_8 << 4));
            p_dst[(i >> 1) + (lane << 1) + 1] = (uint8_t)(sm_16 | (sm_24 << 4));
        }
    }

    // stage3: 把 tunstall 流尾部空出来的字节拿来存第4bit尾数
    size_t extra_bytes = (n_bf16 / 2) - tunstall_len;
    if (extra_bytes >= (n_bf16 / 8)) extra_bytes = (n_bf16 / 8);
    uint8_t *p_extra = p_tunstall + tunstall_len;
    for (size_t i = 0; i < extra_bytes; i++) {
        p_extra[i] = bf16_pack_extra8(p_src + (i << 3));
    }

    return R_TS_OK;
}



__attribute__((unused)) static int decompress_tunstall_trunc_v1 (
    uint16_t      *p_dst,
    const uint8_t *p_src,
    size_t         n_bf16
) {
    uint8_t *p_exp = ((uint8_t*)p_dst) + n_bf16;
    uint8_t *p_exp_end = p_exp + n_bf16;
    const uint8_t *p_tunstall = p_src + (n_bf16 / 2);
    const ts_header_t *p_hdr = (const ts_header_t*)p_tunstall;
    const uint8_t *p_mark  = p_tunstall + sizeof(p_hdr->dynamic);
    const uint8_t *p_extra = p_mark + (size_t)p_hdr->dynamic.n_mark;
    const uint8_t *p_extra_final = p_src + n_bf16 - 4;

    RET_ERROR_IF(R_ERR_SYNTAX, (n_bf16 % 32 != 0));
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->mode != TS_MODE_DYNAMIC);
    RET_ERROR_IF(R_ERR_SYNTAX, (p_hdr->dynamic.n_mark & 1) != 0);
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->dynamic.src_len != n_bf16);

    {   // stage1: tunstall decompress
        uint8_t *p_exp_write  = p_exp;

        while ((size_t)(p_exp_write - p_exp) < (n_bf16 - 8)) {
            uint8_t mark = *p_mark++;
            const ts_lut_item_t *it = &(p_hdr->dynamic.lut[mark]);
            memcpy(p_exp_write, it->v, sizeof(ts_lut_item_t));
            p_exp_write += it->c;
        }
        
        // tunstall decompress 末尾，防止踩内存
        while ((size_t)(p_exp_write - p_exp) < n_bf16) {
            uint8_t mark = *p_mark++;
            const ts_lut_item_t *it = &(p_hdr->dynamic.lut[mark]);
            for (uint8_t i=0; i<it->c; i++) {
                if ((size_t)(p_exp_write - p_exp) < n_bf16) {
                    p_exp_write[0] = it->v[i];
                    p_exp_write ++;
                }
            }
        }
    }

    // stage2: BF16 组装
    #if !defined(NEON_DECOMPRESSION)
        for (; p_exp < p_exp_end; p_exp += 32) {
            for (size_t lane = 0; lane < 8; lane++) {
                uint8_t exp_0  = p_exp[(lane << 1)];
                uint8_t exp_8  = p_exp[(lane << 1) + 1];
                uint8_t exp_16 = p_exp[16 + (lane << 1)];
                uint8_t exp_24 = p_exp[16 + (lane << 1) + 1];
                uint8_t sm_lo  = p_src[(lane << 1)];
                uint8_t sm_hi  = p_src[(lane << 1) + 1];
                p_dst[lane]      = ( (sm_lo       & 0x08) << 12) | ((exp_0  + BF16_EXP_MIN) << 7) | ( (sm_lo       & 0x07) << 4);
                p_dst[lane + 8]  = (((sm_lo >> 4) & 0x08) << 12) | ((exp_8  + BF16_EXP_MIN) << 7) | (((sm_lo >> 4) & 0x07) << 4);
                p_dst[lane + 16] = ( (sm_hi       & 0x08) << 12) | ((exp_16 + BF16_EXP_MIN) << 7) | ( (sm_hi       & 0x07) << 4);
                p_dst[lane + 24] = (((sm_hi >> 4) & 0x08) << 12) | ((exp_24 + BF16_EXP_MIN) << 7) | (((sm_hi >> 4) & 0x07) << 4);
                if (p_extra <= p_extra_final) {
                    p_dst[lane]      |= (((p_extra[0] >> (7 - lane)) & 0x01) << 3);
                    p_dst[lane + 8]  |= (((p_extra[1] >> (7 - lane)) & 0x01) << 3);
                    p_dst[lane + 16] |= (((p_extra[2] >> (7 - lane)) & 0x01) << 3);
                    p_dst[lane + 24] |= (((p_extra[3] >> (7 - lane)) & 0x01) << 3);
                    p_dst[lane]      |= 0x03;
                    p_dst[lane + 8]  |= 0x03;
                    p_dst[lane + 16] |= 0x03;
                    p_dst[lane + 24] |= 0x03;
                } else {
                    p_dst[lane]      |= 0x08;
                    p_dst[lane + 8]  |= 0x08;
                    p_dst[lane + 16] |= 0x08;
                    p_dst[lane + 24] |= 0x08;
                }
            }
            
            p_src   += 16;
            p_dst   += 32;
            p_extra += 4;
        }
    #else 
        const uint16x8_t v_exp_bias_hi_lo = vdupq_n_u16(BF16_EXP_MIN + (BF16_EXP_MIN<<8));
        const uint16x8_t v0003 = vdupq_n_u16(0x0003);
        const uint16x8_t v0008 = vdupq_n_u16(0x0008);
        const uint16x8_t v8070 = vdupq_n_u16(0x8070);
        static const int16_t g_extra_shift[8] = {-4, -3, -2, -1, 0, 1, 2, 3};
        const int16x8_t  v_extra_shift = vld1q_s16(g_extra_shift);

        for (; p_exp < p_exp_end; p_exp += 32) {
            uint16x8_t exp_pack0 = vld1q_u16((uint16_t*)p_exp);
            uint16x8_t exp_pack1 = vld1q_u16((uint16_t*)(p_exp + 16));

            exp_pack0 = vaddq_u16(exp_pack0, v_exp_bias_hi_lo);
            exp_pack1 = vaddq_u16(exp_pack1, v_exp_bias_hi_lo);

            uint16x8_t exp_0_7   = vshrq_n_u16(vshlq_n_u16(exp_pack0, 8), 1);
            uint16x8_t exp_8_15  = vshlq_n_u16(vshrq_n_u16(exp_pack0, 8), 7);
            uint16x8_t exp_16_23 = vshrq_n_u16(vshlq_n_u16(exp_pack1, 8), 1);
            uint16x8_t exp_24_31 = vshlq_n_u16(vshrq_n_u16(exp_pack1, 8), 7);

            uint16x8_t sm_pack  = vld1q_u16((uint16_t*)p_src);

            uint16x8_t sm_0_7   = vshlq_n_u16(sm_pack, 12);                              // yyyy000000000000
            uint16x8_t sm_8_15  = vshlq_n_u16(sm_pack, 8);                               // yyyyxxxx00000000
            uint16x8_t sm_16_23 = vshrq_n_u16(sm_pack, 4);                               // 0000xxxxyyyyxxxx
            uint16x8_t sm_24_31 = vshrq_n_u16(sm_pack, 8);                               // 00000000yyyyxxxx
            
            sm_0_7   = vandq_u16(vorrq_u16(vshrq_n_u16(sm_0_7  , 8), sm_0_7  ), v8070);  // yyyy000000000000 -> y00000000yyy0000
            sm_8_15  = vandq_u16(vorrq_u16(vshrq_n_u16(sm_8_15 , 8), sm_8_15 ), v8070);  // yyyy000000000000 -> y00000000yyy0000
            sm_16_23 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_16_23, 8), sm_16_23), v8070);  // 0000xxxxyyyyxxxx -> y00000000yyy0000
            sm_24_31 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_24_31, 8), sm_24_31), v8070);  // 00000000yyyyxxxx -> y00000000yyy0000

            uint16x8_t bf16_0_7   = vorrq_u16(sm_0_7  , exp_0_7  );
            uint16x8_t bf16_8_15  = vorrq_u16(sm_8_15 , exp_8_15 );
            uint16x8_t bf16_16_23 = vorrq_u16(sm_16_23, exp_16_23);
            uint16x8_t bf16_24_31 = vorrq_u16(sm_24_31, exp_24_31);

            if (ENABLE_EXTRA_DECOMPRESSION && p_extra <= p_extra_final) {
                bf16_0_7   = vorrq_u16(bf16_0_7  , vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[0]), v_extra_shift), v0008));
                bf16_8_15  = vorrq_u16(bf16_8_15 , vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[1]), v_extra_shift), v0008));
                bf16_16_23 = vorrq_u16(bf16_16_23, vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[2]), v_extra_shift), v0008));
                bf16_24_31 = vorrq_u16(bf16_24_31, vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[3]), v_extra_shift), v0008));
                bf16_0_7   = vorrq_u16(bf16_0_7  , v0003);
                bf16_8_15  = vorrq_u16(bf16_8_15 , v0003);
                bf16_16_23 = vorrq_u16(bf16_16_23, v0003);
                bf16_24_31 = vorrq_u16(bf16_24_31, v0003);
            } else {
                bf16_0_7   = vorrq_u16(bf16_0_7  , v0008);
                bf16_8_15  = vorrq_u16(bf16_8_15 , v0008);
                bf16_16_23 = vorrq_u16(bf16_16_23, v0008);
                bf16_24_31 = vorrq_u16(bf16_24_31, v0008);
            }

            vst1q_u16(p_dst,      bf16_0_7);
            vst1q_u16(p_dst + 8,  bf16_8_15);
            vst1q_u16(p_dst + 16, bf16_16_23);
            vst1q_u16(p_dst + 24, bf16_24_31);

            p_src   += 16;
            p_dst   += 32;
            p_extra += 4;
        }
    #endif

    return R_TS_OK;
}



static int decompress_tunstall_trunc_v2 (
    uint16_t      *p_dst,
    const uint8_t *p_src,
    size_t         n_bf16
) {
    const uint8_t *p_tunstall = p_src + (n_bf16 / 2);
    const ts_header_t *p_hdr = (const ts_header_t*)p_tunstall;
    const uint8_t *p_mark  = p_tunstall + sizeof(p_hdr->dynamic);
    const uint8_t *p_extra = p_mark + (size_t)p_hdr->dynamic.n_mark;
    const uint8_t *p_extra_final = p_src + n_bf16 - 4;

    RET_ERROR_IF(R_ERR_SYNTAX, (n_bf16 % 32 != 0));
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->mode != TS_MODE_DYNAMIC);
    RET_ERROR_IF(R_ERR_SYNTAX, (p_hdr->dynamic.n_mark & 1) != 0);
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->dynamic.src_len != n_bf16);

    #define   BLOCK_SIZE    8192
    #define   BLOCK_TAIL    8
    uint8_t   buf_exp [BLOCK_SIZE + BLOCK_TAIL];  // ring buffer for exp
    uint8_t * p_exp_write = buf_exp;
    
    #if defined(NEON_DECOMPRESSION)
        const uint16x8_t v_exp_bias_hi_lo = vdupq_n_u16(BF16_EXP_MIN + (BF16_EXP_MIN<<8));
        const uint16x8_t v0003 = vdupq_n_u16(0x0003);
        const uint16x8_t v0008 = vdupq_n_u16(0x0008);
        const uint16x8_t v8070 = vdupq_n_u16(0x8070);
        static const int16_t g_extra_shift[8] = {-4, -3, -2, -1, 0, 1, 2, 3};
        const int16x8_t  v_extra_shift = vld1q_s16(g_extra_shift);
    #endif

    for (size_t i_block = 0; i_block < n_bf16; i_block += BLOCK_SIZE) {

        size_t actual_block_size = (n_bf16 - i_block < BLOCK_SIZE) ? (n_bf16 - i_block) : BLOCK_SIZE;
        const uint8_t *p_exp_end = buf_exp + actual_block_size;

        while (p_exp_write < p_exp_end) {
            const ts_lut_item_t *it = &(p_hdr->dynamic.lut[p_mark[0]]);
            p_mark ++;
            memcpy(p_exp_write, it->v, sizeof(ts_lut_item_t));
            p_exp_write += it->c;
        }

        // stage2: BF16 组装
        for (const uint8_t *p_exp_read = buf_exp; p_exp_read < p_exp_end; p_exp_read += 32) {
            #if !defined(NEON_DECOMPRESSION)
                for (size_t lane = 0; lane < 8; lane++) {
                    uint16_t exp_0  = p_exp_read[(lane << 1)];
                    uint16_t exp_8  = p_exp_read[(lane << 1) + 1];
                    uint16_t exp_16 = p_exp_read[16 + (lane << 1)];
                    uint16_t exp_24 = p_exp_read[16 + (lane << 1) + 1];
                    uint16_t sm_lo  = p_src[(lane << 1)];
                    uint16_t sm_hi  = p_src[(lane << 1) + 1];
                    p_dst[lane]      = ( (sm_lo       & 0x08) << 12) | ((exp_0  + BF16_EXP_MIN) << 7) | ( (sm_lo       & 0x07) << 4);
                    p_dst[lane + 8]  = (((sm_lo >> 4) & 0x08) << 12) | ((exp_8  + BF16_EXP_MIN) << 7) | (((sm_lo >> 4) & 0x07) << 4);
                    p_dst[lane + 16] = ( (sm_hi       & 0x08) << 12) | ((exp_16 + BF16_EXP_MIN) << 7) | ( (sm_hi       & 0x07) << 4);
                    p_dst[lane + 24] = (((sm_hi >> 4) & 0x08) << 12) | ((exp_24 + BF16_EXP_MIN) << 7) | (((sm_hi >> 4) & 0x07) << 4);
                    if (p_extra <= p_extra_final) {
                        p_dst[lane]      |= (((p_extra[0] >> (7 - lane)) & 0x01) << 3);
                        p_dst[lane + 8]  |= (((p_extra[1] >> (7 - lane)) & 0x01) << 3);
                        p_dst[lane + 16] |= (((p_extra[2] >> (7 - lane)) & 0x01) << 3);
                        p_dst[lane + 24] |= (((p_extra[3] >> (7 - lane)) & 0x01) << 3);
                        p_dst[lane]      |= 0x03;
                        p_dst[lane + 8]  |= 0x03;
                        p_dst[lane + 16] |= 0x03;
                        p_dst[lane + 24] |= 0x03;
                    } else {
                        p_dst[lane]      |= 0x08;
                        p_dst[lane + 8]  |= 0x08;
                        p_dst[lane + 16] |= 0x08;
                        p_dst[lane + 24] |= 0x08;
                    }
                }
            #else
                uint16x8_t exp_pack0 = vld1q_u16((uint16_t*)p_exp_read);
                uint16x8_t exp_pack1 = vld1q_u16((uint16_t*)(p_exp_read + 16));

                exp_pack0 = vaddq_u16(exp_pack0, v_exp_bias_hi_lo);
                exp_pack1 = vaddq_u16(exp_pack1, v_exp_bias_hi_lo);

                uint16x8_t exp_0_7   = vshrq_n_u16(vshlq_n_u16(exp_pack0, 8), 1);
                uint16x8_t exp_8_15  = vshlq_n_u16(vshrq_n_u16(exp_pack0, 8), 7);
                uint16x8_t exp_16_23 = vshrq_n_u16(vshlq_n_u16(exp_pack1, 8), 1);
                uint16x8_t exp_24_31 = vshlq_n_u16(vshrq_n_u16(exp_pack1, 8), 7);

                uint16x8_t sm_pack  = vld1q_u16((uint16_t*)p_src);

                uint16x8_t sm_0_7   = vshlq_n_u16(sm_pack, 12);                              // yyyy000000000000
                uint16x8_t sm_8_15  = vshlq_n_u16(sm_pack, 8);                               // yyyyxxxx00000000
                uint16x8_t sm_16_23 = vshrq_n_u16(sm_pack, 4);                               // 0000xxxxyyyyxxxx
                uint16x8_t sm_24_31 = vshrq_n_u16(sm_pack, 8);                               // 00000000yyyyxxxx
                
                sm_0_7   = vandq_u16(vorrq_u16(vshrq_n_u16(sm_0_7  , 8), sm_0_7  ), v8070);  // yyyy000000000000 -> y00000000yyy0000
                sm_8_15  = vandq_u16(vorrq_u16(vshrq_n_u16(sm_8_15 , 8), sm_8_15 ), v8070);  // yyyy000000000000 -> y00000000yyy0000
                sm_16_23 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_16_23, 8), sm_16_23), v8070);  // 0000xxxxyyyyxxxx -> y00000000yyy0000
                sm_24_31 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_24_31, 8), sm_24_31), v8070);  // 00000000yyyyxxxx -> y00000000yyy0000

                uint16x8_t bf16_0_7   = vorrq_u16(sm_0_7  , exp_0_7  );
                uint16x8_t bf16_8_15  = vorrq_u16(sm_8_15 , exp_8_15 );
                uint16x8_t bf16_16_23 = vorrq_u16(sm_16_23, exp_16_23);
                uint16x8_t bf16_24_31 = vorrq_u16(sm_24_31, exp_24_31);

                if (ENABLE_EXTRA_DECOMPRESSION && p_extra <= p_extra_final) {
                    bf16_0_7   = vorrq_u16(bf16_0_7  , vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[0]), v_extra_shift), v0008));
                    bf16_8_15  = vorrq_u16(bf16_8_15 , vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[1]), v_extra_shift), v0008));
                    bf16_16_23 = vorrq_u16(bf16_16_23, vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[2]), v_extra_shift), v0008));
                    bf16_24_31 = vorrq_u16(bf16_24_31, vandq_u16(vshlq_u16(vdupq_n_u16(p_extra[3]), v_extra_shift), v0008));
                    bf16_0_7   = vorrq_u16(bf16_0_7  , v0003);
                    bf16_8_15  = vorrq_u16(bf16_8_15 , v0003);
                    bf16_16_23 = vorrq_u16(bf16_16_23, v0003);
                    bf16_24_31 = vorrq_u16(bf16_24_31, v0003);
                } else {
                    bf16_0_7   = vorrq_u16(bf16_0_7  , v0008);
                    bf16_8_15  = vorrq_u16(bf16_8_15 , v0008);
                    bf16_16_23 = vorrq_u16(bf16_16_23, v0008);
                    bf16_24_31 = vorrq_u16(bf16_24_31, v0008);
                }

                vst1q_u16(p_dst,      bf16_0_7);
                vst1q_u16(p_dst + 8,  bf16_8_15);
                vst1q_u16(p_dst + 16, bf16_16_23);
                vst1q_u16(p_dst + 24, bf16_24_31);
            #endif

            p_src   += 16;
            p_dst   += 32;
            p_extra += 4;
        }

        memcpy(buf_exp, (buf_exp+BLOCK_SIZE), BLOCK_TAIL);
        p_exp_write -= BLOCK_SIZE;
    }

    return R_TS_OK;
}



// p_src 里是压缩流，占 n_bf16 字节
// 把它们解压到 p_dst ，解压出来 n_bf16 个 BF16 
int TunstallDecompressBF16 (
    uint16_t      *p_dst,    //      n_bf16*2 字节
    const uint8_t *p_src,    //      n_bf16 字节
    size_t         n_bf16
) {
    RET_ERROR_IF(R_ERR_UNSUPPORT, n_bf16 == 0 || n_bf16 > 0xFFFFFFFFU);
    
    if ((n_bf16 % 32 != 0) || (p_src[n_bf16/2] != TS_MODE_DYNAMIC)) {   // p_src[src_len / 2] 是 tunstall 流的起始字节, 也即 mode byte ，如果它不是 tunstall dynamic 模式，说明是 fp8 压缩
        decompress_fp8_to_bf16(p_dst, p_src, n_bf16);
        return R_TS_OK;
    } else {
        #if DECOMPRESSION_VERSION == 1
            return decompress_tunstall_trunc_v1(p_dst, p_src, n_bf16);
        #else 
            return decompress_tunstall_trunc_v2(p_dst, p_src, n_bf16);
        #endif
    }
}