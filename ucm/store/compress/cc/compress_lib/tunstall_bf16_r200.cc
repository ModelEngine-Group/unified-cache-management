#include "tunstall_bf16_r200.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define MIN_BF16_COUNT 16384
#define INPLACE_TAIL_BYTES 8192

#if defined(__ARM_NEON) || defined(__aarch64__)
#define NEON_DECOMPRESSION
#include <arm_neon.h>
#endif

#define RET_ERROR_IF(err_code, condition) \
    do {                                  \
        int c = (condition);              \
        int e = (err_code);               \
        if (c) { return e; }              \
    } while (0)

#define BF16_EXP_MIN (128 - (1 << 4))
#define BF16_EXP_MAX (128 + (1 << 4) - 1)

#if defined(NEON_DECOMPRESSION)
using ExtraExpandLut = std::array<std::array<uint16_t, 8>, 256>;

static ExtraExpandLut build_extra_expand_lut()
{
    ExtraExpandLut lut = {};
    for (size_t packed = 0; packed < lut.size(); packed++) {
        for (size_t lane = 0; lane < lut[packed].size(); lane++) {
            lut[packed][lane] =
                static_cast<uint16_t>(((((packed >> (7 - lane)) & 1U) << 3) | 0x03U));
        }
    }
    return lut;
}

alignas(64) static const ExtraExpandLut g_extra_expand_lut = build_extra_expand_lut();
static_assert(sizeof(ExtraExpandLut) == 4096, "R200 extra expansion LUT must stay L1-sized");
#endif

static uint8_t bf16_to_exp(uint16_t x)
{
    uint16_t e = (x >> 7) & 0xFF;
    if (e < BF16_EXP_MIN) { e = BF16_EXP_MIN; }
    if (e > BF16_EXP_MAX) { e = BF16_EXP_MAX; }
    return static_cast<uint8_t>(e - BF16_EXP_MIN);
}

static uint8_t bf16_sign_mant8(uint16_t x)
{
    return static_cast<uint8_t>(((x >> 8) & 0x80) | (x & 0x7F));
}

static uint8_t bf16_pack_extra8(const uint16_t* p_src)
{
    uint8_t extra8 = 0;
    for (size_t lane = 0; lane < 8; lane++) {
        extra8 |= static_cast<uint8_t>(((p_src[lane] >> 3) & 0x01) << (7 - lane));
    }
    return extra8;
}

static void compress_bf16_to_fp8(uint8_t* p_dst, const uint16_t* p_src, size_t n_bf16)
{
    for (size_t i = 0; i < n_bf16; i++) {
        uint16_t x = p_src[i];
        uint8_t exp5 = bf16_to_exp(x);
        uint8_t sign = static_cast<uint8_t>((x >> 15) & 0x01);
        uint8_t mant2 = static_cast<uint8_t>((x >> 5) & 0x03);
        p_dst[i] = static_cast<uint8_t>((exp5 << 3) | (sign << 2) | mant2);
    }
}

static void decompress_fp8_to_bf16(uint16_t* p_dst, const uint8_t* p_src, size_t n_bf16)
{
#if defined(NEON_DECOMPRESSION)
    const uint16x8_t v_exp_min_shifted = vdupq_n_u16(static_cast<uint16_t>(BF16_EXP_MIN) << 7);
    const uint16x8_t mask_exp = vdupq_n_u16(0x00f8);
    const uint16x8_t mask_sign = vdupq_n_u16(0x04);
    const uint16x8_t mask_mant = vdupq_n_u16(0x03);
    const uint16x8_t v_const_10 = vdupq_n_u16(0x10);

    size_t i = 0;
    for (; i + 16 <= n_bf16; i += 16) {
        const uint8x16_t fp8_vec = vld1q_u8(p_src + i);
        const uint16x8_t fp16_low = vmovl_u8(vget_low_u8(fp8_vec));
        const uint16x8_t fp16_high = vmovl_u8(vget_high_u8(fp8_vec));

        const uint16x8_t exp_low =
            vaddq_u16(vshlq_n_u16(vandq_u16(fp16_low, mask_exp), 4), v_exp_min_shifted);
        const uint16x8_t exp_high =
            vaddq_u16(vshlq_n_u16(vandq_u16(fp16_high, mask_exp), 4), v_exp_min_shifted);
        const uint16x8_t sign_low = vshlq_n_u16(vandq_u16(fp16_low, mask_sign), 13);
        const uint16x8_t sign_high = vshlq_n_u16(vandq_u16(fp16_high, mask_sign), 13);
        const uint16x8_t mant_low =
            vorrq_u16(vshlq_n_u16(vandq_u16(fp16_low, mask_mant), 5), v_const_10);
        const uint16x8_t mant_high =
            vorrq_u16(vshlq_n_u16(vandq_u16(fp16_high, mask_mant), 5), v_const_10);

        vst1q_u16(p_dst + i, vorrq_u16(vorrq_u16(exp_low, sign_low), mant_low));
        vst1q_u16(p_dst + i + 8, vorrq_u16(vorrq_u16(exp_high, sign_high), mant_high));
    }
    for (; i < n_bf16; i++) {
#else
    for (size_t i = 0; i < n_bf16; i++) {
#endif
        uint16_t fp8 = p_src[i];
        uint16_t exp8 =
            static_cast<uint16_t>((static_cast<uint16_t>(fp8 >> 3) + BF16_EXP_MIN) << 7);
        uint16_t sign = static_cast<uint16_t>((fp8 & 0x04) << 13);
        uint16_t mant2 = static_cast<uint16_t>((fp8 & 0x03) << 5);
        p_dst[i] = sign | exp8 | mant2 | 0x10;
    }
}

static int decompress_fp8_to_bf16_inplace(uint8_t* p_data, size_t n_bf16)
{
#if !defined(NEON_DECOMPRESSION)
    for (size_t i = n_bf16; i > 0; i--) {
        uint16_t fp8 = p_data[i - 1];
        uint16_t exp8 =
            static_cast<uint16_t>((static_cast<uint16_t>(fp8 >> 3) + BF16_EXP_MIN) << 7);
        uint16_t sign = static_cast<uint16_t>((fp8 & 0x04) << 13);
        uint16_t mant2 = static_cast<uint16_t>((fp8 & 0x03) << 5);
        uint16_t bf16 = sign | exp8 | mant2 | 0x10;
        memcpy(p_data + ((i - 1) << 1), &bf16, sizeof(bf16));
    }
    return R_TS_OK;
#else
    uint16_t* dst = (uint16_t*)p_data;
    const uint8_t* src = (const uint8_t*)p_data;

    // 计算 n_bf16 % 16 的余数
    size_t remainder = n_bf16 & 15;

// --- 标量前导循环：倒序处理余数部分 ---
#pragma GCC ivdep
    for (int i = (int)n_bf16 - 1; i >= (int)(n_bf16 - remainder); i--) {
        uint16_t fp8 = src[i];
        uint16_t exp8 =
            static_cast<uint16_t>((static_cast<uint16_t>(fp8 >> 3) + BF16_EXP_MIN) << 7);
        uint16_t sign = static_cast<uint16_t>((fp8 & 0x04) << 13);
        uint16_t mant2 = static_cast<uint16_t>((fp8 & 0x03) << 5);
        dst[i] = sign | exp8 | mant2 | 0x10;
    }

    // --- NEON 主循环常量 ---
    const uint16x8_t v_exp_min_shifted = vdupq_n_u16(static_cast<uint16_t>(BF16_EXP_MIN) << 7);
    const uint16x8_t mask_exp = vdupq_n_u16(0x00f8);
    const uint16x8_t mask_sign = vdupq_n_u16(0x04);
    const uint16x8_t mask_mant = vdupq_n_u16(0x03);
    const uint16x8_t v_const_10 = vdupq_n_u16(0x10);

    int main_start = (int)(n_bf16 - remainder) - 16;

// --- NEON 主循环：每次处理 16 个 FP8 ---
#pragma GCC ivdep
    for (int i = main_start; i >= 0; i -= 16) {
        // 预取下一个缓存行 (倒序预取，提升命中率)
        __builtin_prefetch(src + i - 64, 0, 0);

        // 1. 加载 16 字节 FP8 并扩展为 16-bit
        uint8x16_t fp8_vec = vld1q_u8(src + i);
        uint16x8_t fp16_low = vmovl_u8(vget_low_u8(fp8_vec));
        uint16x8_t fp16_high = vmovl_u8(vget_high_u8(fp8_vec));

        // --- 指数计算  ---
        uint16x8_t exp_low = vshlq_n_u16(vandq_u16(fp16_low, mask_exp), 4);
        uint16x8_t exp_high = vshlq_n_u16(vandq_u16(fp16_high, mask_exp), 4);
        exp_low = vaddq_u16(exp_low, v_exp_min_shifted);
        exp_high = vaddq_u16(exp_high, v_exp_min_shifted);

        // --- 符号位提取 ---
        uint16x8_t sign_low = vandq_u16(fp16_low, mask_sign);
        uint16x8_t sign_high = vandq_u16(fp16_high, mask_sign);
        sign_low = vshlq_n_u16(sign_low, 13);
        sign_high = vshlq_n_u16(sign_high, 13);

        // --- 尾数提取 + 常数合并 ---
        uint16x8_t mant_low = vandq_u16(fp16_low, mask_mant);
        uint16x8_t mant_high = vandq_u16(fp16_high, mask_mant);
        mant_low = vshlq_n_u16(mant_low, 5);
        mant_high = vshlq_n_u16(mant_high, 5);

        // --- 最终合并  ---
        uint16x8_t res_low = vorrq_u16(mant_low, v_const_10);
        uint16x8_t res_high = vorrq_u16(mant_high, v_const_10);
        res_low = vorrq_u16(res_low, exp_low);
        res_high = vorrq_u16(res_high, exp_high);
        res_low = vorrq_u16(res_low, sign_low);
        res_high = vorrq_u16(res_high, sign_high);

        // ================= 存储结果 =================
        vst1q_u16(dst + i, res_low);
        vst1q_u16(dst + i + 8, res_high);
    }
    return R_TS_OK;
#endif
}

// p_src 里有 n_bf16 个 BF16
// 把它们压缩到 p_dst 里，占 n_bf16 字节
// 压缩率固定为 2x
int TunstallCompressBF16(uint8_t* p_dst,         // 需要 n_bf16*2 字节, 但压缩后起始只有 n_bf16 字节
                         const uint16_t* p_src,  //      n_bf16*2 字节
                         size_t n_bf16)
{
    RET_ERROR_IF(R_ERR_UNSUPPORT, n_bf16 == 0 || n_bf16 > 0xFFFFFFFFU);

    if (n_bf16 < MIN_BF16_COUNT || n_bf16 % 32 != 0) {  // 数据量不够多，或者不是 32 的倍数
        compress_bf16_to_fp8(p_dst, p_src, n_bf16);     // fallback 到 FP8 压缩
        p_dst[n_bf16 / 2] &=
            0xFE;  // 确保 p_dst[n_bf16/2] != TS_MODE_DYNAMIC, 避免被识别成 tunstall 模式
        return R_TS_OK;
    }

    // stage0: 指数重新排布
    size_t tunstall_len = (n_bf16 / 2);  // Tunstall 压缩流的预算 = n_bf16 / 2 字节
    uint8_t* p_tunstall =
        p_dst + (n_bf16 / 2);         // Tunstall 压缩流放在          p_dst[n_bf16/2 : n_bf16]
    uint8_t* p_exp = p_dst + n_bf16;  // 调整数据排布后的 exp 暂时放在 p_dst[n_bf16 : n_bf16*2]
    for (size_t i = 0; i < n_bf16; i += 16) {
        for (size_t lane = 0; lane < 8; lane++) {
            p_exp[i + (lane << 1)] = bf16_to_exp(p_src[i + lane]);
            p_exp[i + (lane << 1) + 1] = bf16_to_exp(p_src[i + lane + 8]);
        }
    }

    // stage1: 指数 tunstall 压缩
    int err = TunstallCompressDynamic(p_tunstall, &tunstall_len, p_exp, n_bf16);

    if (err ||
        tunstall_len > (n_bf16 / 2)) {  // tunstall 压缩错误，或者无法保证压缩到 n_bf16 / 2 字节内
        compress_bf16_to_fp8(p_dst, p_src, n_bf16);  // fallback 到 FP8 压缩
        p_dst[n_bf16 / 2] &=
            0xFE;  // 确保 p_dst[n_bf16/2] != TS_MODE_DYNAMIC, 避免被识别成 tunstall 模式
        return R_TS_OK;
    }

    // stage2: 符号+3bit尾数打包
    for (size_t i = 0; i < n_bf16; i += 32) {
        for (size_t lane = 0; lane < 8; lane++) {
            uint8_t sm_0 = static_cast<uint8_t>(bf16_sign_mant8(p_src[i + lane]) >> 4);
            uint8_t sm_8 = static_cast<uint8_t>(bf16_sign_mant8(p_src[i + lane + 8]) >> 4);
            uint8_t sm_16 = static_cast<uint8_t>(bf16_sign_mant8(p_src[i + lane + 16]) >> 4);
            uint8_t sm_24 = static_cast<uint8_t>(bf16_sign_mant8(p_src[i + lane + 24]) >> 4);
            p_dst[(i >> 1) + (lane << 1)] = static_cast<uint8_t>(sm_0 | (sm_8 << 4));
            p_dst[(i >> 1) + (lane << 1) + 1] = static_cast<uint8_t>(sm_16 | (sm_24 << 4));
        }
    }

    // stage3: 把 tunstall 流尾部空出来的字节拿来存第4bit尾数
    size_t extra_bytes = (n_bf16 / 2) - tunstall_len;
    if (extra_bytes >= (n_bf16 / 8)) { extra_bytes = (n_bf16 / 8); }
    uint8_t* p_extra = p_tunstall + tunstall_len;
    for (size_t i = 0; i < extra_bytes; i++) { p_extra[i] = bf16_pack_extra8(p_src + (i << 3)); }

    return R_TS_OK;
}

static_assert(TS_MARK_BITS == 8, "R200 mark16 decoder expects 8-bit marks");
static_assert(TS_ITEM_SIZE == 7, "R200 mark16 decoder requires 7-symbol LUT items");
static_assert(sizeof(ts_lut_item_t) == 8, "R200 mark16 decoder requires 8-byte LUT items");

#define BLOCK_SIZE 8192
#define BLOCK_TAIL sizeof(ts_lut_item_t)

static int inspect_decode_lut(const ts_lut_item_t* p_lut, bool* p_all_nonzero)
{
    bool all_nonzero = true;
    for (size_t i = 0; i < TS_LUT_SIZE; i++) {
        const uint8_t length = p_lut[i].c;
        RET_ERROR_IF(R_ERR_SYNTAX, length > TS_ITEM_SIZE);
        all_nonzero = all_nonzero && length != 0;
    }
    *p_all_nonzero = all_nonzero;
    return R_TS_OK;
}

// Decode a logical exponent block. The caller provides BLOCK_TAIL writable
// bytes after p_exp_end because every LUT copy is a fixed 8-byte store, while
// only item.c bytes are logically produced.
template <bool ValidateZeroLength>
static int decode_tunstall_marks_mark16(uint8_t** pp_exp_write, const uint8_t* p_exp_end,
                                        const uint8_t** pp_mark, const uint8_t* p_mark_end,
                                        const ts_lut_item_t* p_lut)
{
    uint8_t* p_exp_write = *pp_exp_write;
    const uint8_t* p_mark = *pp_mark;

    RET_ERROR_IF(R_ERR_SYNTAX, p_exp_write > p_exp_end || p_mark > p_mark_end);

    while (p_exp_write < p_exp_end) {
        const size_t output_remaining = (size_t)(p_exp_end - p_exp_write);
        const size_t mark_remaining = (size_t)(p_mark_end - p_mark);

        // BLOCK_TAIL covers the fixed 8-byte store that may cross the logical
        // block end. LUT lengths were capped once before entering this loop.
        if (output_remaining >= 16 * TS_ITEM_SIZE && mark_remaining >= 16) {
            const ts_lut_item_t* i0 = p_lut + p_mark[0];
            const ts_lut_item_t* i1 = p_lut + p_mark[1];
            const ts_lut_item_t* i2 = p_lut + p_mark[2];
            const ts_lut_item_t* i3 = p_lut + p_mark[3];
            const ts_lut_item_t* i4 = p_lut + p_mark[4];
            const ts_lut_item_t* i5 = p_lut + p_mark[5];
            const ts_lut_item_t* i6 = p_lut + p_mark[6];
            const ts_lut_item_t* i7 = p_lut + p_mark[7];
            const ts_lut_item_t* i8 = p_lut + p_mark[8];
            const ts_lut_item_t* i9 = p_lut + p_mark[9];
            const ts_lut_item_t* ia = p_lut + p_mark[10];
            const ts_lut_item_t* ib = p_lut + p_mark[11];
            const ts_lut_item_t* ic = p_lut + p_mark[12];
            const ts_lut_item_t* id = p_lut + p_mark[13];
            const ts_lut_item_t* ie = p_lut + p_mark[14];
            const ts_lut_item_t* if_ = p_lut + p_mark[15];

            const size_t l0 = i0->c, l1 = i1->c, l2 = i2->c, l3 = i3->c;
            const size_t l4 = i4->c, l5 = i5->c, l6 = i6->c, l7 = i7->c;
            const size_t l8 = i8->c, l9 = i9->c, la = ia->c, lb = ib->c;
            const size_t lc = ic->c, ld = id->c, le = ie->c, lf = if_->c;
            if (ValidateZeroLength) {
                RET_ERROR_IF(R_ERR_SYNTAX, l0 == 0 || l1 == 0 || l2 == 0 || l3 == 0 || l4 == 0 ||
                                               l5 == 0 || l6 == 0 || l7 == 0 || l8 == 0 ||
                                               l9 == 0 || la == 0 || lb == 0 || lc == 0 ||
                                               ld == 0 || le == 0 || lf == 0);
            }

            uint8_t* o0 = p_exp_write;
            p_exp_write += l0;
            uint8_t* o1 = p_exp_write;
            p_exp_write += l1;
            uint8_t* o2 = p_exp_write;
            p_exp_write += l2;
            uint8_t* o3 = p_exp_write;
            p_exp_write += l3;
            uint8_t* o4 = p_exp_write;
            p_exp_write += l4;
            uint8_t* o5 = p_exp_write;
            p_exp_write += l5;
            uint8_t* o6 = p_exp_write;
            p_exp_write += l6;
            uint8_t* o7 = p_exp_write;
            p_exp_write += l7;
            uint8_t* o8 = p_exp_write;
            p_exp_write += l8;
            uint8_t* o9 = p_exp_write;
            p_exp_write += l9;
            uint8_t* oa = p_exp_write;
            p_exp_write += la;
            uint8_t* ob = p_exp_write;
            p_exp_write += lb;
            uint8_t* oc = p_exp_write;
            p_exp_write += lc;
            uint8_t* od = p_exp_write;
            p_exp_write += ld;
            uint8_t* oe = p_exp_write;
            p_exp_write += le;
            uint8_t* of = p_exp_write;
            p_exp_write += lf;

            memcpy(o0, i0->v, sizeof(*i0));
            memcpy(o1, i1->v, sizeof(*i1));
            memcpy(o2, i2->v, sizeof(*i2));
            memcpy(o3, i3->v, sizeof(*i3));
            memcpy(o4, i4->v, sizeof(*i4));
            memcpy(o5, i5->v, sizeof(*i5));
            memcpy(o6, i6->v, sizeof(*i6));
            memcpy(o7, i7->v, sizeof(*i7));
            memcpy(o8, i8->v, sizeof(*i8));
            memcpy(o9, i9->v, sizeof(*i9));
            memcpy(oa, ia->v, sizeof(*ia));
            memcpy(ob, ib->v, sizeof(*ib));
            memcpy(oc, ic->v, sizeof(*ic));
            memcpy(od, id->v, sizeof(*id));
            memcpy(oe, ie->v, sizeof(*ie));
            memcpy(of, if_->v, sizeof(*if_));
            p_mark += 16;
            continue;
        }

        // Keep most block-boundary work on a two-mark path. The 8-byte guard
        // safely covers both fixed stores when 14 logical bytes remain.
        if (output_remaining >= 2 * TS_ITEM_SIZE && mark_remaining >= 2) {
            const ts_lut_item_t* i0 = p_lut + p_mark[0];
            const ts_lut_item_t* i1 = p_lut + p_mark[1];
            const size_t l0 = i0->c;
            const size_t l1 = i1->c;
            if (ValidateZeroLength) { RET_ERROR_IF(R_ERR_SYNTAX, l0 == 0 || l1 == 0); }
            memcpy(p_exp_write, i0->v, sizeof(*i0));
            p_exp_write += l0;
            memcpy(p_exp_write, i1->v, sizeof(*i1));
            p_exp_write += l1;
            p_mark += 2;
            continue;
        }

        RET_ERROR_IF(R_ERR_SYNTAX, p_mark == p_mark_end);
        const ts_lut_item_t* item = p_lut + *p_mark++;
        if (ValidateZeroLength) { RET_ERROR_IF(R_ERR_SYNTAX, item->c == 0); }
        memcpy(p_exp_write, item->v, sizeof(*item));
        p_exp_write += item->c;
    }

    *pp_exp_write = p_exp_write;
    *pp_mark = p_mark;
    return R_TS_OK;
}

template <bool HAS_EXTRA>
static inline __attribute__((always_inline)) void join_tunstall_trunc_tile(
    uint16_t* p_dst, const uint8_t* p_exp, const uint8_t* p_sm, const uint8_t* p_extra
#if defined(NEON_DECOMPRESSION)
    ,
    uint16x8_t v_exp_bias_hi_lo, uint16x8_t v0008, uint16x8_t v8070
#endif
)
{
#if !defined(NEON_DECOMPRESSION)
    for (size_t lane = 0; lane < 8; lane++) {
        const uint16_t exp_0 = p_exp[(lane << 1)];
        const uint16_t exp_8 = p_exp[(lane << 1) + 1];
        const uint16_t exp_16 = p_exp[16 + (lane << 1)];
        const uint16_t exp_24 = p_exp[16 + (lane << 1) + 1];
        const uint16_t sm_lo = p_sm[(lane << 1)];
        const uint16_t sm_hi = p_sm[(lane << 1) + 1];
        p_dst[lane] =
            ((sm_lo & 0x08) << 12) | ((exp_0 + BF16_EXP_MIN) << 7) | ((sm_lo & 0x07) << 4);
        p_dst[lane + 8] = (((sm_lo >> 4) & 0x08) << 12) | ((exp_8 + BF16_EXP_MIN) << 7) |
                          (((sm_lo >> 4) & 0x07) << 4);
        p_dst[lane + 16] =
            ((sm_hi & 0x08) << 12) | ((exp_16 + BF16_EXP_MIN) << 7) | ((sm_hi & 0x07) << 4);
        p_dst[lane + 24] = (((sm_hi >> 4) & 0x08) << 12) | ((exp_24 + BF16_EXP_MIN) << 7) |
                           (((sm_hi >> 4) & 0x07) << 4);
        if (HAS_EXTRA) {
            p_dst[lane] |= (uint16_t)(((p_extra[0] >> (7 - lane)) & 0x01) << 3) | 0x03;
            p_dst[lane + 8] |= (uint16_t)(((p_extra[1] >> (7 - lane)) & 0x01) << 3) | 0x03;
            p_dst[lane + 16] |= (uint16_t)(((p_extra[2] >> (7 - lane)) & 0x01) << 3) | 0x03;
            p_dst[lane + 24] |= (uint16_t)(((p_extra[3] >> (7 - lane)) & 0x01) << 3) | 0x03;
        } else {
            p_dst[lane] |= 0x08;
            p_dst[lane + 8] |= 0x08;
            p_dst[lane + 16] |= 0x08;
            p_dst[lane + 24] |= 0x08;
        }
    }
#else
    // Each load contains eight interleaved exponent pairs. Extract their low
    // and high bytes directly into the four BF16 exponent vectors.
    uint16x8_t exp_pack0 = vreinterpretq_u16_u8(vld1q_u8(p_exp));
    uint16x8_t exp_pack1 = vreinterpretq_u16_u8(vld1q_u8(p_exp + 16));
    exp_pack0 = vaddq_u16(exp_pack0, v_exp_bias_hi_lo);
    exp_pack1 = vaddq_u16(exp_pack1, v_exp_bias_hi_lo);
    const uint16x8_t exp_0_7 = vshrq_n_u16(vshlq_n_u16(exp_pack0, 8), 1);
    const uint16x8_t exp_8_15 = vshlq_n_u16(vshrq_n_u16(exp_pack0, 8), 7);
    const uint16x8_t exp_16_23 = vshrq_n_u16(vshlq_n_u16(exp_pack1, 8), 1);
    const uint16x8_t exp_24_31 = vshlq_n_u16(vshrq_n_u16(exp_pack1, 8), 7);

    const uint16x8_t sm_pack = vreinterpretq_u16_u8(vld1q_u8(p_sm));
    uint16x8_t sm_0_7 = vshlq_n_u16(sm_pack, 12);
    uint16x8_t sm_8_15 = vshlq_n_u16(sm_pack, 8);
    uint16x8_t sm_16_23 = vshrq_n_u16(sm_pack, 4);
    uint16x8_t sm_24_31 = vshrq_n_u16(sm_pack, 8);
    sm_0_7 = vandq_u16(vorrq_u16(vshrq_n_u16(sm_0_7, 8), sm_0_7), v8070);
    sm_8_15 = vandq_u16(vorrq_u16(vshrq_n_u16(sm_8_15, 8), sm_8_15), v8070);
    sm_16_23 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_16_23, 8), sm_16_23), v8070);
    sm_24_31 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_24_31, 8), sm_24_31), v8070);

    uint16x8_t bf16_0_7 = vorrq_u16(sm_0_7, exp_0_7);
    uint16x8_t bf16_8_15 = vorrq_u16(sm_8_15, exp_8_15);
    uint16x8_t bf16_16_23 = vorrq_u16(sm_16_23, exp_16_23);
    uint16x8_t bf16_24_31 = vorrq_u16(sm_24_31, exp_24_31);

    if (HAS_EXTRA) {
        // Each table entry expands one packed M3 plane and the fixed low bits
        // into eight BF16 lanes.
        bf16_0_7 = vorrq_u16(bf16_0_7, vld1q_u16(g_extra_expand_lut[p_extra[0]].data()));
        bf16_8_15 = vorrq_u16(bf16_8_15, vld1q_u16(g_extra_expand_lut[p_extra[1]].data()));
        bf16_16_23 = vorrq_u16(bf16_16_23, vld1q_u16(g_extra_expand_lut[p_extra[2]].data()));
        bf16_24_31 = vorrq_u16(bf16_24_31, vld1q_u16(g_extra_expand_lut[p_extra[3]].data()));
    } else {
        bf16_0_7 = vorrq_u16(bf16_0_7, v0008);
        bf16_8_15 = vorrq_u16(bf16_8_15, v0008);
        bf16_16_23 = vorrq_u16(bf16_16_23, v0008);
        bf16_24_31 = vorrq_u16(bf16_24_31, v0008);
    }

    vst1q_u16(p_dst, bf16_0_7);
    vst1q_u16(p_dst + 8, bf16_8_15);
    vst1q_u16(p_dst + 16, bf16_16_23);
    vst1q_u16(p_dst + 24, bf16_24_31);
#endif
}

static int decompress_tunstall_trunc(uint16_t* p_dst, const uint8_t* p_src, size_t n_bf16)
{
    RET_ERROR_IF(R_ERR_SYNTAX, (n_bf16 % 32 != 0));
    const size_t tunstall_capacity = n_bf16 / 2;
    const uint8_t* p_tunstall = p_src + tunstall_capacity;
    const ts_header_t* p_hdr = (const ts_header_t*)p_tunstall;
    RET_ERROR_IF(R_ERR_SYNTAX, tunstall_capacity < sizeof(p_hdr->dynamic));
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->mode != TS_MODE_DYNAMIC);
    RET_ERROR_IF(R_ERR_SYNTAX, (p_hdr->dynamic.n_mark & 1) != 0);
    RET_ERROR_IF(R_ERR_SYNTAX, p_hdr->dynamic.src_len != n_bf16);
    RET_ERROR_IF(R_ERR_SYNTAX,
                 (size_t)p_hdr->dynamic.n_mark > tunstall_capacity - sizeof(p_hdr->dynamic));

    const uint8_t* p_mark = p_tunstall + sizeof(p_hdr->dynamic);
    const uint8_t* p_mark_end = p_mark + (size_t)p_hdr->dynamic.n_mark;
    const uint8_t* p_extra = p_mark_end;
    const uint8_t* p_payload_end = p_src + n_bf16;
    const size_t total_tiles = n_bf16 / 32;
    const size_t available_extra_tiles = (size_t)(p_payload_end - p_extra) / 4;
    const size_t extra_tiles =
        available_extra_tiles < total_tiles ? available_extra_tiles : total_tiles;

    bool lut_all_nonzero = false;
    int err = inspect_decode_lut(p_hdr->dynamic.lut, &lut_all_nonzero);
    if (err) { return err; }

    uint8_t buf_exp[BLOCK_SIZE + BLOCK_TAIL];  // ring buffer for exp
    uint8_t* p_exp_write = buf_exp;
    size_t tiles_done = 0;

#if defined(NEON_DECOMPRESSION)
    const uint16x8_t v_exp_bias_hi_lo = vdupq_n_u16(BF16_EXP_MIN + (BF16_EXP_MIN << 8));
    const uint16x8_t v0008 = vdupq_n_u16(0x0008);
    const uint16x8_t v8070 = vdupq_n_u16(0x8070);
#endif

    for (size_t i_block = 0; i_block < n_bf16; i_block += BLOCK_SIZE) {
        size_t actual_block_size =
            (n_bf16 - i_block < BLOCK_SIZE) ? (n_bf16 - i_block) : BLOCK_SIZE;
        const uint8_t* p_exp_end = buf_exp + actual_block_size;

        const bool final_block = i_block + actual_block_size == n_bf16;
        err = lut_all_nonzero
                  ? decode_tunstall_marks_mark16<false>(&p_exp_write, p_exp_end, &p_mark,
                                                        p_mark_end, p_hdr->dynamic.lut)
                  : decode_tunstall_marks_mark16<true>(&p_exp_write, p_exp_end, &p_mark, p_mark_end,
                                                       p_hdr->dynamic.lut);
        if (err) { return err; }
        if (final_block) {
            // The encoder emits marks in pairs. Its second final mark may be
            // pure padding and is intentionally left unread once src_len has
            // been reconstructed.
            RET_ERROR_IF(R_ERR_SYNTAX, p_exp_write < p_exp_end ||
                                           (size_t)(p_exp_write - p_exp_end) > TS_ITEM_SIZE - 1 ||
                                           p_mark > p_mark_end ||
                                           (size_t)(p_mark_end - p_mark) > 1);
        }

        // The extra plane always describes a prefix. Split it from the
        // no-extra suffix once per block so the tile join has no data-dependent
        // branch and p_extra never advances beyond the payload object.
        const size_t block_tiles = actual_block_size / 32;
        const size_t remaining_extra_tiles =
            tiles_done < extra_tiles ? extra_tiles - tiles_done : 0;
        const size_t block_extra_tiles =
            remaining_extra_tiles < block_tiles ? remaining_extra_tiles : block_tiles;
        const uint8_t* p_exp_read = buf_exp;
        for (size_t tile = 0; tile < block_extra_tiles; tile++) {
            join_tunstall_trunc_tile<true>(p_dst, p_exp_read, p_src, p_extra
#if defined(NEON_DECOMPRESSION)
                                           ,
                                           v_exp_bias_hi_lo, v0008, v8070
#endif
            );
            p_exp_read += 32;
            p_src += 16;
            p_dst += 32;
            p_extra += 4;
        }
        for (size_t tile = block_extra_tiles; tile < block_tiles; tile++) {
            join_tunstall_trunc_tile<false>(p_dst, p_exp_read, p_src, nullptr
#if defined(NEON_DECOMPRESSION)
                                            ,
                                            v_exp_bias_hi_lo, v0008, v8070
#endif
            );
            p_exp_read += 32;
            p_src += 16;
            p_dst += 32;
        }
        tiles_done += block_tiles;

        if (actual_block_size == BLOCK_SIZE) {
            memcpy(buf_exp, buf_exp + BLOCK_SIZE, BLOCK_TAIL);
            p_exp_write -= BLOCK_SIZE;
        }
    }

    return R_TS_OK;
}

static int compact_tunstall_inplace_streams(uint8_t* p_stream_end, uint8_t** pp_sm,
                                            uint8_t** pp_sm_end, uint8_t** pp_mark,
                                            uint8_t** pp_mark_end, uint8_t** pp_extra,
                                            uint8_t** pp_extra_end)
{
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_mark > *pp_mark_end);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_mark_end > *pp_extra);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_extra > *pp_extra_end);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_extra_end > *pp_sm);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_sm > *pp_sm_end);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_sm_end != p_stream_end);

    size_t sm_len = (size_t)(*pp_sm_end - *pp_sm);
    size_t mark_len = (size_t)(*pp_mark_end - *pp_mark);
    size_t extra_len = (size_t)(*pp_extra_end - *pp_extra);
    RET_ERROR_IF(R_ERR_SYNTAX, sm_len + mark_len + extra_len > (size_t)(p_stream_end - *pp_mark));

    uint8_t* p_sm_new = p_stream_end - sm_len;
    uint8_t* p_extra_new = p_sm_new - extra_len;
    uint8_t* p_mark_new = p_extra_new - mark_len;

    if (extra_len > 0) { memmove(p_extra_new, *pp_extra, extra_len); }
    if (mark_len > 0) { memmove(p_mark_new, *pp_mark, mark_len); }

    *pp_mark = p_mark_new;
    *pp_mark_end = p_extra_new;
    *pp_extra = p_extra_new;
    *pp_extra_end = p_sm_new;
    *pp_sm = p_sm_new;
    *pp_sm_end = p_stream_end;

    return R_TS_OK;
}

static int spill_tunstall_inplace_streams(uint8_t* p_tail, size_t tail_size, uint8_t** pp_sm,
                                          uint8_t** pp_sm_end, uint8_t** pp_mark,
                                          uint8_t** pp_mark_end, uint8_t** pp_extra,
                                          uint8_t** pp_extra_end)
{
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_mark > *pp_mark_end);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_mark_end > *pp_extra);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_extra > *pp_extra_end);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_extra_end > *pp_sm);
    RET_ERROR_IF(R_ERR_SYNTAX, *pp_sm > *pp_sm_end);

    size_t tail_len = (size_t)(*pp_sm_end - *pp_mark);
    size_t mark_end_off = (size_t)(*pp_mark_end - *pp_mark);
    size_t extra_off = (size_t)(*pp_extra - *pp_mark);
    size_t extra_end_off = (size_t)(*pp_extra_end - *pp_mark);
    size_t sm_off = (size_t)(*pp_sm - *pp_mark);

    RET_ERROR_IF(R_ERR_SYNTAX, tail_len > tail_size);

    memcpy(p_tail, *pp_mark, tail_len);

    *pp_mark = p_tail;
    *pp_mark_end = p_tail + mark_end_off;
    *pp_extra = p_tail + extra_off;
    *pp_extra_end = p_tail + extra_end_off;
    *pp_sm = p_tail + sm_off;
    *pp_sm_end = p_tail + tail_len;

    return R_TS_OK;
}

static int decompress_tunstall_trunc_inplace(uint8_t* p_data, size_t n_bf16)
{
    RET_ERROR_IF(R_ERR_SYNTAX, (n_bf16 % 32 != 0));

    uint8_t* p_src = p_data + n_bf16;
    uint8_t* p_src_end = p_src + n_bf16;
    size_t sm_total = n_bf16 / 2;
    uint8_t* p_tunstall_src = p_data + sm_total;

    ts_header_t hdr = {};
    RET_ERROR_IF(R_ERR_SYNTAX, (n_bf16 / 2) < sizeof(hdr.dynamic));
    memcpy(&hdr.dynamic, p_tunstall_src, sizeof(hdr.dynamic));

    RET_ERROR_IF(R_ERR_SYNTAX, hdr.mode != TS_MODE_DYNAMIC);
    RET_ERROR_IF(R_ERR_SYNTAX, (hdr.dynamic.n_mark & 1) != 0);
    RET_ERROR_IF(R_ERR_SYNTAX, hdr.dynamic.src_len != n_bf16);

    size_t tunstall_len = sizeof(hdr.dynamic) + (size_t)hdr.dynamic.n_mark;
    RET_ERROR_IF(R_ERR_SYNTAX, tunstall_len > sm_total);

    bool lut_all_nonzero = false;
    int lut_err = inspect_decode_lut(hdr.dynamic.lut, &lut_all_nonzero);
    if (lut_err) { return lut_err; }

    // Repack upper half as {tunstall, extra, sm}; compact then moves only tunstall/extra.
    memcpy(p_src, p_tunstall_src, sm_total);
    memcpy(p_src + sm_total, p_data, sm_total);

    uint8_t* p_mark = p_src + sizeof(hdr.dynamic);
    uint8_t* p_mark_end = p_mark + (size_t)hdr.dynamic.n_mark;
    uint8_t* p_extra = p_mark_end;
    uint8_t* p_extra_end = p_src + sm_total;
    uint8_t* p_sm = p_extra_end;
    uint8_t* p_sm_end = p_src_end;
    uint8_t* p_dst = p_data;

    uint8_t buf_exp[BLOCK_SIZE + BLOCK_TAIL];
    uint8_t* p_exp_write = buf_exp;
    uint8_t tail[INPLACE_TAIL_BYTES];
    int using_tail = 0;
    const size_t dst_block_bytes = 32 * sizeof(uint16_t);

#if defined(NEON_DECOMPRESSION)
    const uint16x8_t v_exp_bias_hi_lo = vdupq_n_u16(BF16_EXP_MIN + (BF16_EXP_MIN << 8));
    const uint16x8_t v0008 = vdupq_n_u16(0x0008);
    const uint16x8_t v8070 = vdupq_n_u16(0x8070);
#endif

    for (size_t i_block = 0; i_block < n_bf16; i_block += BLOCK_SIZE) {
        size_t actual_block_size =
            (n_bf16 - i_block < BLOCK_SIZE) ? (n_bf16 - i_block) : BLOCK_SIZE;
        const uint8_t* p_exp_end = buf_exp + actual_block_size;

        const uint8_t* p_mark_read = p_mark;
        const bool final_block = i_block + actual_block_size == n_bf16;
        int decode_err =
            lut_all_nonzero
                ? decode_tunstall_marks_mark16<false>(&p_exp_write, p_exp_end, &p_mark_read,
                                                      p_mark_end, hdr.dynamic.lut)
                : decode_tunstall_marks_mark16<true>(&p_exp_write, p_exp_end, &p_mark_read,
                                                     p_mark_end, hdr.dynamic.lut);
        p_mark = const_cast<uint8_t*>(p_mark_read);
        if (decode_err) { return decode_err; }
        if (final_block) {
            RET_ERROR_IF(R_ERR_SYNTAX, p_exp_write < p_exp_end ||
                                           (size_t)(p_exp_write - p_exp_end) > TS_ITEM_SIZE - 1 ||
                                           p_mark > p_mark_end ||
                                           (size_t)(p_mark_end - p_mark) > 1);
        }

        for (const uint8_t* p_exp_read = buf_exp; p_exp_read < p_exp_end; p_exp_read += 32) {
            int err = R_TS_OK;
            if (!using_tail && (size_t)(p_sm_end - p_mark) <= sizeof(tail)) {
                err = spill_tunstall_inplace_streams(tail, sizeof(tail), &p_sm, &p_sm_end, &p_mark,
                                                     &p_mark_end, &p_extra, &p_extra_end);
                if (err) { return err; }
                using_tail = 1;
            }

            uint8_t sm[16];
            uint8_t extra[4] = {0};

            RET_ERROR_IF(R_ERR_SYNTAX, (size_t)(p_sm_end - p_sm) < sizeof(sm));
            memcpy(sm, p_sm, sizeof(sm));
            p_sm += sizeof(sm);

            int has_extra = 0;
            if ((size_t)(p_extra_end - p_extra) >= sizeof(extra)) {
                memcpy(extra, p_extra, sizeof(extra));
                p_extra += sizeof(extra);
                has_extra = 1;
            } else {
                p_extra = p_extra_end;
            }

            if (!using_tail) {
                if ((size_t)(p_sm_end - p_mark) <= sizeof(tail)) {
                    err = spill_tunstall_inplace_streams(tail, sizeof(tail), &p_sm, &p_sm_end,
                                                         &p_mark, &p_mark_end, &p_extra,
                                                         &p_extra_end);
                    if (err) { return err; }
                    using_tail = 1;
                } else {
                    uint8_t* p_first_src =
                        (p_mark < p_mark_end) ? p_mark : ((p_extra < p_extra_end) ? p_extra : p_sm);
                    if (p_dst + dst_block_bytes > p_first_src) {
                        err = compact_tunstall_inplace_streams(p_sm_end, &p_sm, &p_sm_end, &p_mark,
                                                               &p_mark_end, &p_extra, &p_extra_end);
                        if (err) { return err; }
                        p_first_src = (p_mark < p_mark_end)
                                          ? p_mark
                                          : ((p_extra < p_extra_end) ? p_extra : p_sm);
                        RET_ERROR_IF(R_ERR_SYNTAX, p_dst + dst_block_bytes > p_first_src);
                    }
                }
            }

            if (has_extra) {
                join_tunstall_trunc_tile<true>((uint16_t*)p_dst, p_exp_read, sm, extra
#if defined(NEON_DECOMPRESSION)
                                               ,
                                               v_exp_bias_hi_lo, v0008, v8070
#endif
                );
            } else {
                join_tunstall_trunc_tile<false>((uint16_t*)p_dst, p_exp_read, sm, nullptr
#if defined(NEON_DECOMPRESSION)
                                                ,
                                                v_exp_bias_hi_lo, v0008, v8070
#endif
                );
            }

            p_dst += dst_block_bytes;
        }

        if (actual_block_size == BLOCK_SIZE) {
            memcpy(buf_exp, buf_exp + BLOCK_SIZE, BLOCK_TAIL);
            p_exp_write -= BLOCK_SIZE;
        }
    }

    return R_TS_OK;
}

// p_data[0:n_bf16] is compressed input, p_data[0:n_bf16*2] is BF16 output.
int TunstallDecompressBF16Inplace(
    uint8_t* p_data,  // 可访问范围是 n_bf16*2 字节，解压前的数据占据 p_data[0:n_bf16]
                      // 字节，解压后占据 p_data[0:n_bf16*2] 字节
    size_t n_bf16)
{
    const R200PayloadMode mode = TunstallGetBF16R200Mode(p_data, n_bf16);
    RET_ERROR_IF(R_ERR_UNSUPPORT, mode == R200PayloadMode::INVALID);
    return mode == R200PayloadMode::TUNSTALL ? decompress_tunstall_trunc_inplace(p_data, n_bf16)
                                             : decompress_fp8_to_bf16_inplace(p_data, n_bf16);
}

R200PayloadMode TunstallGetBF16R200Mode(const uint8_t* p_src, size_t payload_bytes) noexcept
{
    if (p_src == nullptr || payload_bytes == 0 || payload_bytes > 0xFFFFFFFFU) {
        return R200PayloadMode::INVALID;
    }
    if ((payload_bytes % 32 != 0) || (p_src[payload_bytes / 2] != TS_MODE_DYNAMIC)) {
        return R200PayloadMode::FP8_FALLBACK;
    }
    return R200PayloadMode::TUNSTALL;
}

// p_src 里是压缩流，占 n_bf16 字节
// 把它们解压到 p_dst ，解压出来 n_bf16 个 BF16
int TunstallDecompressBF16(uint16_t* p_dst,       // n_bf16*2 字节
                           const uint8_t* p_src,  // n_bf16 字节
                           size_t n_bf16)
{
    const R200PayloadMode mode = TunstallGetBF16R200Mode(p_src, n_bf16);
    RET_ERROR_IF(R_ERR_UNSUPPORT, p_dst == nullptr || mode == R200PayloadMode::INVALID);
    if (mode == R200PayloadMode::FP8_FALLBACK) {
        decompress_fp8_to_bf16(p_dst, p_src, n_bf16);
        return R_TS_OK;
    }
    return decompress_tunstall_trunc(p_dst, p_src, n_bf16);
}
