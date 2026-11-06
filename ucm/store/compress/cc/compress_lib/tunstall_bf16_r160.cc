#include "tunstall_bf16_r160.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

constexpr size_t VALUES_PER_BLOCK = 32;
constexpr size_t E8_BLOCK_VALUES = 8192;
static_assert(E8_BLOCK_VALUES % VALUES_PER_BLOCK == 0,
              "E8 decode blocks must preserve the 32-value stream layout");
constexpr size_t TUNSTALL_GUARD_BYTES = sizeof(uint32_t);
constexpr size_t E8_DECODE_GUARD_BYTES = sizeof(ts_lut_item_t);
constexpr uint8_t BF16_QUANT_EXP_MIN = 128 - (1 << 4);
constexpr uint8_t BF16_QUANT_EXP_MAX = 128 + (1 << 4) - 1;

#if defined(__aarch64__)
using M1ExpandLut = std::array<std::array<uint16_t, 8>, 256>;

M1ExpandLut BuildM1ExpandLut()
{
    M1ExpandLut lut{};
    for (size_t packed = 0; packed < lut.size(); ++packed) {
        for (size_t lane = 0; lane < lut[packed].size(); ++lane) {
            lut[packed][lane] = static_cast<uint16_t>(((packed >> (7 - lane)) & 1U) << 1);
        }
    }
    return lut;
}

alignas(64) const M1ExpandLut M1_EXPAND_LUT = BuildM1ExpandLut();
static_assert(sizeof(M1ExpandLut) == 4096, "M1 expansion LUT must remain 4 KiB");
#endif

struct DynamicE8View {
    uint32_t markCount{0};
    size_t payloadBytes{0};
    bool allLutEntriesPopulated{false};
};

size_t SM4Bytes(size_t n_bf16) { return n_bf16 / 2; }

size_t M32Bytes(size_t n_bf16) { return n_bf16 / 4; }

size_t PlaneBytes(size_t n_bf16) { return n_bf16 / 8; }

size_t HighPrecisionBaseBytes(size_t n_bf16) { return SM4Bytes(n_bf16) + M32Bytes(n_bf16); }

bool CheckedR160Input(size_t n_bf16, size_t& maximum_bytes)
{
    if (n_bf16 < 2 * VALUES_PER_BLOCK || n_bf16 % VALUES_PER_BLOCK != 0 ||
        n_bf16 > std::numeric_limits<size_t>::max() / 5) {
        return false;
    }
    maximum_bytes = n_bf16 * 5 / 4;
    return maximum_bytes <= std::numeric_limits<uint32_t>::max();
}

bool CheckedR160StreamSize(size_t n_bf16, size_t stored_bytes)
{
    size_t maximum_bytes = 0;
    return CheckedR160Input(n_bf16, maximum_bytes) && stored_bytes >= n_bf16 &&
           stored_bytes <= maximum_bytes;
}

R160PayloadMode ClassifyMode(uint8_t mode)
{
    if (mode == TS_MODE_DYNAMIC) { return R160PayloadMode::HIGH_PRECISION; }
    return (mode & 1U) == 0 ? R160PayloadMode::QUANTIZED : R160PayloadMode::INVALID;
}

R160PayloadMode ClassifyR160Payload(const uint8_t* src, size_t stored_bytes, size_t n_bf16)
{
    if (src == nullptr || !CheckedR160StreamSize(n_bf16, stored_bytes)) {
        return R160PayloadMode::INVALID;
    }
    return ClassifyMode(src[HighPrecisionBaseBytes(n_bf16)]);
}

bool EncodeDynamicE8(std::vector<uint8_t>& encoded, const uint16_t* src, size_t n_bf16,
                     size_t max_bytes)
{
    if (max_bytes <= TUNSTALL_GUARD_BYTES) { return false; }

    thread_local std::vector<uint8_t> dense;
    dense.resize(n_bf16);
    std::array<int16_t, 256> e8_to_index;
    e8_to_index.fill(-1);
    std::array<uint8_t, TS_N_SYMB> index_to_e8{};
    size_t symbol_count = 0;
    // Match the R200 32-value tile: E0,E8,E1,E9,...,E7,E15, then
    // E16,E24,...,E23,E31.  The decoded E8 vectors then have the same lane
    // shape as SM4 and can be split with one NEON structure load.
    for (size_t i = 0; i < n_bf16; i += 16) {
        for (size_t lane = 0; lane < 8; ++lane) {
            for (size_t half = 0; half < 2; ++half) {
                const size_t source_index = i + lane + half * 8;
                const size_t dense_index = i + lane * 2 + half;
                const uint8_t e8 = static_cast<uint8_t>((src[source_index] >> 7) & 0xffU);
                int16_t index = e8_to_index[e8];
                if (index < 0) {
                    if (symbol_count == TS_N_SYMB) { return false; }
                    index = static_cast<int16_t>(symbol_count);
                    e8_to_index[e8] = index;
                    index_to_e8[symbol_count++] = e8;
                }
                dense[dense_index] = static_cast<uint8_t>(index);
            }
        }
    }

    encoded.resize(max_bytes);
    size_t encoded_bytes = max_bytes - TUNSTALL_GUARD_BYTES;
    const int err =
        TunstallCompressDynamic(encoded.data(), &encoded_bytes, dense.data(), dense.size());
    if (err != R_TS_OK || encoded_bytes > max_bytes - TUNSTALL_GUARD_BYTES) { return false; }

    auto* header = reinterpret_cast<ts_header_t*>(encoded.data());
    if (encoded_bytes < sizeof(header->dynamic) || header->mode != TS_MODE_DYNAMIC ||
        header->dynamic.src_len != n_bf16 || header->dynamic.n_mark == 0 ||
        (header->dynamic.n_mark & 1U) != 0) {
        return false;
    }
    for (size_t i = 0; i < TS_LUT_SIZE; ++i) {
        if (header->dynamic.lut[i].c > TS_ITEM_SIZE) { return false; }
        for (size_t j = 0; j < header->dynamic.lut[i].c; ++j) {
            const uint8_t index = header->dynamic.lut[i].v[j];
            if (index >= symbol_count) { return false; }
            header->dynamic.lut[i].v[j] = index_to_e8[index];
        }
    }

    std::memset(encoded.data() + encoded_bytes, 0, TUNSTALL_GUARD_BYTES);
    encoded.resize(encoded_bytes + TUNSTALL_GUARD_BYTES);
    return true;
}

bool ParseDynamicE8(const uint8_t* payload, size_t available, size_t n_bf16, DynamicE8View& view)
{
    if (payload == nullptr || TS_MARK_BITS != 8) { return false; }
    const auto* header = reinterpret_cast<const ts_header_t*>(payload);
    if (available < sizeof(header->dynamic)) { return false; }
    if (header->mode != TS_MODE_DYNAMIC || header->dynamic.src_len != n_bf16 ||
        header->dynamic.n_mark == 0 || (header->dynamic.n_mark & 1U) != 0) {
        return false;
    }
    bool all_lut_entries_populated = true;
    for (size_t i = 0; i < TS_LUT_SIZE; ++i) {
        const uint8_t length = header->dynamic.lut[i].c;
        if (length > TS_ITEM_SIZE) { return false; }
        all_lut_entries_populated &= length != 0;
    }

    view.markCount = header->dynamic.n_mark;
    const size_t mark_bytes = static_cast<size_t>(view.markCount);
    const size_t header_bytes = sizeof(header->dynamic);
    if (mark_bytes > available - header_bytes ||
        TUNSTALL_GUARD_BYTES > available - header_bytes - mark_bytes) {
        return false;
    }
    view.payloadBytes = header_bytes + mark_bytes + TUNSTALL_GUARD_BYTES;
    view.allLutEntriesPopulated = all_lut_entries_populated;
    return true;
}

struct DynamicE8Decoder {
    const uint8_t* marks{nullptr};
    size_t marksRemaining{0};
    const ts_lut_item_t* lut{nullptr};
};

template <bool ValidateZeroLength>
bool DecodeDynamicE8Block(uint8_t*& output, uint8_t* logical_end, uint8_t* padded_end,
                          DynamicE8Decoder& decoder)
{
    static_assert(TS_MARK_BITS == 8, "R160 mark16 decoder requires 8-bit Tunstall marks");
    static_assert(TS_ITEM_SIZE == 7, "R160 mark16 decoder requires 7-symbol LUT items");
    static_assert(sizeof(ts_lut_item_t) == 8,
                  "R160 mark16 decoder requires 8-byte Tunstall LUT items");

    while (output < logical_end) {
        const size_t output_remaining = static_cast<size_t>(logical_end - output);
        if (decoder.marksRemaining >= 16 && output_remaining >= 16 * TS_ITEM_SIZE) {
            const uint8_t* marks = decoder.marks;
            const ts_lut_item_t* lut = decoder.lut;
            const ts_lut_item_t* i0 = lut + marks[0];
            const ts_lut_item_t* i1 = lut + marks[1];
            const ts_lut_item_t* i2 = lut + marks[2];
            const ts_lut_item_t* i3 = lut + marks[3];
            const ts_lut_item_t* i4 = lut + marks[4];
            const ts_lut_item_t* i5 = lut + marks[5];
            const ts_lut_item_t* i6 = lut + marks[6];
            const ts_lut_item_t* i7 = lut + marks[7];
            const ts_lut_item_t* i8 = lut + marks[8];
            const ts_lut_item_t* i9 = lut + marks[9];
            const ts_lut_item_t* ia = lut + marks[10];
            const ts_lut_item_t* ib = lut + marks[11];
            const ts_lut_item_t* ic = lut + marks[12];
            const ts_lut_item_t* id = lut + marks[13];
            const ts_lut_item_t* ie = lut + marks[14];
            const ts_lut_item_t* if_ = lut + marks[15];

            const size_t l0 = i0->c, l1 = i1->c, l2 = i2->c, l3 = i3->c;
            const size_t l4 = i4->c, l5 = i5->c, l6 = i6->c, l7 = i7->c;
            const size_t l8 = i8->c, l9 = i9->c, la = ia->c, lb = ib->c;
            const size_t lc = ic->c, ld = id->c, le = ie->c, lf = if_->c;
            if (ValidateZeroLength &&
                (l0 == 0 || l1 == 0 || l2 == 0 || l3 == 0 || l4 == 0 || l5 == 0 || l6 == 0 ||
                 l7 == 0 || l8 == 0 || l9 == 0 || la == 0 || lb == 0 || lc == 0 || ld == 0 ||
                 le == 0 || lf == 0)) {
                return false;
            }
            uint8_t* o0 = output;
            output += l0;
            uint8_t* o1 = output;
            output += l1;
            uint8_t* o2 = output;
            output += l2;
            uint8_t* o3 = output;
            output += l3;
            uint8_t* o4 = output;
            output += l4;
            uint8_t* o5 = output;
            output += l5;
            uint8_t* o6 = output;
            output += l6;
            uint8_t* o7 = output;
            output += l7;
            uint8_t* o8 = output;
            output += l8;
            uint8_t* o9 = output;
            output += l9;
            uint8_t* oa = output;
            output += la;
            uint8_t* ob = output;
            output += lb;
            uint8_t* oc = output;
            output += lc;
            uint8_t* od = output;
            output += ld;
            uint8_t* oe = output;
            output += le;
            uint8_t* of = output;
            output += lf;
            std::memcpy(o0, i0->v, sizeof(*i0));
            std::memcpy(o1, i1->v, sizeof(*i1));
            std::memcpy(o2, i2->v, sizeof(*i2));
            std::memcpy(o3, i3->v, sizeof(*i3));
            std::memcpy(o4, i4->v, sizeof(*i4));
            std::memcpy(o5, i5->v, sizeof(*i5));
            std::memcpy(o6, i6->v, sizeof(*i6));
            std::memcpy(o7, i7->v, sizeof(*i7));
            std::memcpy(o8, i8->v, sizeof(*i8));
            std::memcpy(o9, i9->v, sizeof(*i9));
            std::memcpy(oa, ia->v, sizeof(*ia));
            std::memcpy(ob, ib->v, sizeof(*ib));
            std::memcpy(oc, ic->v, sizeof(*ic));
            std::memcpy(od, id->v, sizeof(*id));
            std::memcpy(oe, ie->v, sizeof(*ie));
            std::memcpy(of, if_->v, sizeof(*if_));
            decoder.marks += 16;
            decoder.marksRemaining -= 16;
            continue;
        }

        // Keep most block-boundary work on a two-mark path. ParseDynamicE8
        // has already capped every LUT item at TS_ITEM_SIZE, so 14 logical
        // bytes plus the fixed decode guard safely cover both 8-byte stores.
        if (decoder.marksRemaining >= 2 && output_remaining >= 2 * TS_ITEM_SIZE) {
            const ts_lut_item_t* first = decoder.lut + decoder.marks[0];
            const ts_lut_item_t* second = decoder.lut + decoder.marks[1];
            const size_t first_length = first->c;
            const size_t second_length = second->c;
            if ((ValidateZeroLength && (first_length == 0 || second_length == 0)) ||
                first_length > TS_ITEM_SIZE || second_length > TS_ITEM_SIZE) {
                return false;
            }
            uint8_t* first_output = output;
            output += first_length;
            uint8_t* second_output = output;
            output += second_length;
            std::memcpy(first_output, first->v, sizeof(*first));
            std::memcpy(second_output, second->v, sizeof(*second));
            decoder.marks += 2;
            decoder.marksRemaining -= 2;
            continue;
        }

        if (decoder.marksRemaining == 0) { return false; }
        const ts_lut_item_t* item = decoder.lut + *decoder.marks;
        const size_t length = item->c;
        if ((ValidateZeroLength && length == 0) || length > TS_ITEM_SIZE ||
            static_cast<size_t>(padded_end - output) < sizeof(*item)) {
            return false;
        }
        std::memcpy(output, item->v, sizeof(*item));
        output += length;
        ++decoder.marks;
        --decoder.marksRemaining;
    }
    return true;
}

uint8_t SignMantissaHigh4(uint16_t value)
{
    return static_cast<uint8_t>(((value >> 12) & 0x08U) | ((value >> 4) & 0x07U));
}

uint8_t PackMantissaPlane(const uint16_t* src, unsigned bit)
{
    uint8_t packed = 0;
    for (size_t lane = 0; lane < 8; ++lane) {
        packed |= static_cast<uint8_t>(((src[lane] >> bit) & 1U) << (7 - lane));
    }
    return packed;
}

void PackSM4(uint8_t* dst, const uint16_t* src, size_t n_bf16)
{
    for (size_t i = 0; i < n_bf16; i += VALUES_PER_BLOCK) {
        uint8_t* block = dst + i / 2;
        for (size_t lane = 0; lane < 8; ++lane) {
            const uint8_t sm0 = SignMantissaHigh4(src[i + lane]);
            const uint8_t sm8 = SignMantissaHigh4(src[i + lane + 8]);
            const uint8_t sm16 = SignMantissaHigh4(src[i + lane + 16]);
            const uint8_t sm24 = SignMantissaHigh4(src[i + lane + 24]);
            block[lane * 2] = static_cast<uint8_t>(sm0 | (sm8 << 4));
            block[lane * 2 + 1] = static_cast<uint8_t>(sm16 | (sm24 << 4));
        }
    }
}

// Match the R200 lane-major tile.  One byte holds M3:M2 for values at the same
// lane in the four 8-value groups: [lane, lane+8, lane+16, lane+24].
void PackM32(uint8_t* dst, const uint16_t* src, size_t n_bf16)
{
    for (size_t i = 0; i < n_bf16; i += VALUES_PER_BLOCK) {
        uint8_t* block = dst + i / 4;
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t group = 0; group < 4; ++group) {
                packed |= static_cast<uint8_t>((src[i + group * 8 + lane] >> 2) & 0x03U)
                          << (group * 2);
            }
            block[lane] = packed;
        }
    }
}

void PackPlaneBytes(uint8_t* dst, const uint16_t* src, size_t packed_bytes, unsigned bit)
{
    for (size_t byte = 0; byte < packed_bytes; ++byte) {
        dst[byte] = PackMantissaPlane(src + byte * 8, bit);
    }
}

uint16_t LoadLE16(const uint8_t* src)
{
    return static_cast<uint16_t>(src[0]) | static_cast<uint16_t>(src[1]) << 8;
}

void StoreLE16(uint8_t* dst, uint16_t value)
{
    dst[0] = static_cast<uint8_t>(value);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

// Optional low bits keep the original precision priority while changing only
// their physical layout. Groups with both M1 and M0 are stored as M10. Four
// groups form one lane-major 8-byte tile; the final 1..3 groups use one LE16
// per group. The remaining groups contain only the original M1 bit-plane.
void PackOptionalBits(uint8_t* dst, const uint16_t* src, size_t m1_groups, size_t m0_groups)
{
    const size_t full_m10_groups = m0_groups & ~size_t{3};
    for (size_t group = 0; group < full_m10_groups; group += 4) {
        uint8_t* tile = dst + group * 2;
        for (size_t lane = 0; lane < 8; ++lane) {
            uint8_t packed = 0;
            for (size_t tile_group = 0; tile_group < 4; ++tile_group) {
                packed |= static_cast<uint8_t>(src[(group + tile_group) * 8 + lane] & 0x03U)
                          << (tile_group * 2);
            }
            tile[lane] = packed;
        }
    }
    for (size_t group = full_m10_groups; group < m0_groups; ++group) {
        uint16_t packed = 0;
        for (size_t lane = 0; lane < 8; ++lane) {
            packed |= static_cast<uint16_t>(src[group * 8 + lane] & 0x03U) << (lane * 2);
        }
        StoreLE16(dst + group * 2, packed);
    }

    PackPlaneBytes(dst + m0_groups * 2, src + m0_groups * 8, m1_groups - m0_groups, 1);
}

uint8_t QuantizeBF16(uint16_t value)
{
    uint16_t exponent = static_cast<uint16_t>((value >> 7) & 0xffU);
    exponent = std::max<uint16_t>(exponent, BF16_QUANT_EXP_MIN);
    exponent = std::min<uint16_t>(exponent, BF16_QUANT_EXP_MAX);
    const uint8_t exp5 = static_cast<uint8_t>(exponent - BF16_QUANT_EXP_MIN);
    const uint8_t sign = static_cast<uint8_t>((value >> 15) & 1U);
    const uint8_t mantissa2 = static_cast<uint8_t>((value >> 5) & 3U);
    return static_cast<uint8_t>((exp5 << 3) | (sign << 2) | mantissa2);
}

struct ParsedR160 {
    R160PayloadMode mode{R160PayloadMode::INVALID};
    const uint8_t* sm4{nullptr};
    const uint8_t* m32{nullptr};
    const uint8_t* e8{nullptr};
    DynamicE8View dynamicE8{};
    const uint8_t* optional{nullptr};
    size_t m1Groups{0};
    size_t m0Groups{0};
    const uint8_t* quantized{nullptr};
    const uint8_t* quantM4{nullptr};
    const uint8_t* quantM3{nullptr};
    size_t quantM4Groups{0};
    size_t quantM3Groups{0};
};

int ParseLayout(const uint8_t* src, size_t src_len, size_t n_bf16, ParsedR160& layout)
{
    if (!CheckedR160StreamSize(n_bf16, src_len)) { return R_ERR_R160_STREAM_SIZE; }

    layout.mode = ClassifyMode(src[HighPrecisionBaseBytes(n_bf16)]);
    if (layout.mode == R160PayloadMode::INVALID) { return R_ERR_R160_E8_TAG; }

    if (layout.mode == R160PayloadMode::QUANTIZED) {
        layout.quantized = src;
        const size_t plane_bytes = PlaneBytes(n_bf16);
        layout.quantM4 = layout.quantized + n_bf16;
        const size_t optional_bytes = src_len - n_bf16;
        layout.quantM4Groups = std::min(plane_bytes, optional_bytes);
        layout.quantM3 = layout.quantM4 + layout.quantM4Groups;
        layout.quantM3Groups = std::min(plane_bytes, optional_bytes - layout.quantM4Groups);
        return R_TS_OK;
    }

    const size_t sm4_bytes = SM4Bytes(n_bf16);
    const size_t m32_bytes = M32Bytes(n_bf16);
    size_t offset = 0;
    layout.sm4 = src + offset;
    offset += sm4_bytes;
    layout.m32 = src + offset;
    offset += m32_bytes;

    layout.e8 = src + offset;
    const size_t available = src_len - offset;
    if (!ParseDynamicE8(layout.e8, available, n_bf16, layout.dynamicE8)) {
        return R_ERR_R160_E8_METADATA;
    }
    const size_t optional_bytes = available - layout.dynamicE8.payloadBytes;
    const size_t plane_bytes = PlaneBytes(n_bf16);
    layout.optional = layout.e8 + layout.dynamicE8.payloadBytes;
    layout.m1Groups = std::min(plane_bytes, optional_bytes);
    layout.m0Groups = std::min(plane_bytes, optional_bytes - layout.m1Groups);
    return R_TS_OK;
}

uint8_t ReadOptionalLow2(const ParsedR160& layout, size_t group, size_t lane)
{
    if (group < layout.m0Groups) {
        const size_t full_m10_groups = layout.m0Groups & ~size_t{3};
        if (group < full_m10_groups) {
            const uint8_t packed = layout.optional[(group / 4) * 8 + lane];
            return static_cast<uint8_t>((packed >> ((group & 3U) * 2)) & 0x03U);
        }
        const uint16_t packed = LoadLE16(layout.optional + group * 2);
        return static_cast<uint8_t>((packed >> (lane * 2)) & 0x03U);
    }
    if (group < layout.m1Groups) {
        const uint8_t* m1_only = layout.optional + layout.m0Groups * 2;
        return static_cast<uint8_t>(((m1_only[group - layout.m0Groups] >> (7 - lane)) & 1U) << 1);
    }
    return 0;
}

#if defined(__aarch64__)
void BuildOptionalM10Tile(uint8_t* dst, const ParsedR160& layout, size_t first_group)
{
    for (size_t lane = 0; lane < 8; ++lane) {
        uint8_t packed = 0;
        for (size_t group = 0; group < 4; ++group) {
            packed |= ReadOptionalLow2(layout, first_group + group, lane) << (group * 2);
        }
        dst[lane] = packed;
    }
}
#endif

#if !defined(__aarch64__)
void JoinHighPrecisionScalar(uint16_t* dst, const uint8_t* e8, const ParsedR160& layout,
                             size_t n_bf16)
{
    for (size_t i = 0; i < n_bf16; i += VALUES_PER_BLOCK) {
        const uint8_t* sm4 = layout.sm4 + i / 2;

        for (size_t lane = 0; lane < VALUES_PER_BLOCK; ++lane) {
            const size_t group = lane / 8;
            const size_t group_lane = lane % 8;
            const size_t sm_index = group_lane * 2 + group / 2;
            const unsigned sm_shift = (group & 1U) != 0 ? 4 : 0;
            const uint8_t sm = static_cast<uint8_t>((sm4[sm_index] >> sm_shift) & 0x0fU);
            const uint8_t m32 = layout.m32[i / 4 + group_lane];
            const uint8_t m32_value = static_cast<uint8_t>((m32 >> (group * 2)) & 0x03U);
            const uint16_t sign = static_cast<uint16_t>(sm & 0x08U) << 12;
            const size_t e8_index = (group / 2) * 16 + group_lane * 2 + (group & 1U);
            const uint16_t exponent = static_cast<uint16_t>(e8[i + e8_index]) << 7;
            uint16_t mantissa = static_cast<uint16_t>(sm & 0x07U) << 4;
            mantissa |= static_cast<uint16_t>(m32_value) << 2;
            mantissa |= ReadOptionalLow2(layout, i / 8 + group, group_lane);
            dst[i + lane] = sign | exponent | mantissa;
        }
    }
}

void JoinQuantizedScalar(uint16_t* dst, const ParsedR160& layout, size_t n_bf16)
{
    for (size_t i = 0; i < n_bf16; ++i) {
        const uint8_t quantized = layout.quantized[i];
        const uint16_t exponent = static_cast<uint16_t>((quantized >> 3) + BF16_QUANT_EXP_MIN) << 7;
        const uint16_t sign = static_cast<uint16_t>(quantized & 0x04U) << 13;
        uint16_t mantissa = static_cast<uint16_t>(quantized & 0x03U) << 5;
        const size_t group = i / 8;
        if (group < layout.quantM4Groups) {
            mantissa |= static_cast<uint16_t>((layout.quantM4[group] >> (7 - i % 8)) & 1U) << 4;
        }
        if (group < layout.quantM3Groups) {
            mantissa |= static_cast<uint16_t>((layout.quantM3[group] >> (7 - i % 8)) & 1U) << 3;
        }
        mantissa |= 0x0003U;
        dst[i] = sign | exponent | mantissa;
    }
}
#endif

#if defined(__aarch64__)
inline __attribute__((always_inline)) uint32_t LoadLE32(const uint8_t* src)
{
    uint32_t value = 0;
    std::memcpy(&value, src, sizeof(value));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = __builtin_bswap32(value);
#endif
    return value;
}

uint16x8_t ExpandPlane(uint8_t packed, int16x8_t shifts, uint16_t mask)
{
    return vandq_u16(vshlq_u16(vdupq_n_u16(packed), shifts), vdupq_n_u16(mask));
}

enum class OptionalJoinMode {
    NONE,
    M1_ONLY,
    M10,
};

template <OptionalJoinMode Mode>
inline __attribute__((always_inline)) void JoinHighPrecisionBlockNEON(uint16_t* dst,
                                                                      const uint8_t* e8,
                                                                      const uint8_t* sm4,
                                                                      const uint8_t* m32,
                                                                      const uint8_t* optional)
{
    // E8 is interleaved exactly like R200. LD2 produces
    // {E0..E7,E16..E23} and {E8..E15,E24..E31}.
    const uint8x16x2_t e8_pack = vld2q_u8(e8);
    const uint16x8_t e8_0_7 = vshll_n_u8(vget_low_u8(e8_pack.val[0]), 7);
    const uint16x8_t e8_8_15 = vshll_n_u8(vget_low_u8(e8_pack.val[1]), 7);
    const uint16x8_t e8_16_23 = vshll_high_n_u8(e8_pack.val[0], 7);
    const uint16x8_t e8_24_31 = vshll_high_n_u8(e8_pack.val[1], 7);

    // SM4 uses the same lane pairing as R200, so one 16-byte load yields all
    // four S+M6:M4 vectors without byte unzips.
    const uint16x8_t sm_pack = vreinterpretq_u16_u8(vld1q_u8(sm4));
    const uint16x8_t sm_mask = vdupq_n_u16(0x8070U);
    uint16x8_t sm_0_7 = vshlq_n_u16(sm_pack, 12);
    uint16x8_t sm_8_15 = vshlq_n_u16(sm_pack, 8);
    uint16x8_t sm_16_23 = vshrq_n_u16(sm_pack, 4);
    uint16x8_t sm_24_31 = vshrq_n_u16(sm_pack, 8);
    sm_0_7 = vandq_u16(vorrq_u16(vshrq_n_u16(sm_0_7, 8), sm_0_7), sm_mask);
    sm_8_15 = vandq_u16(vorrq_u16(vshrq_n_u16(sm_8_15, 8), sm_8_15), sm_mask);
    sm_16_23 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_16_23, 8), sm_16_23), sm_mask);
    sm_24_31 = vandq_u16(vorrq_u16(vshlq_n_u16(sm_24_31, 8), sm_24_31), sm_mask);

    // M32 is also lane-major: bits 1:0, 3:2, 5:4 and 7:6 belong to
    // the four 8-value output groups respectively.
    const uint16x8_t m32_pack = vmovl_u8(vld1_u8(m32));
    const uint16x8_t m32_mask = vdupq_n_u16(0x000cU);
    const uint16x8_t m32_0_7 = vandq_u16(vshlq_n_u16(m32_pack, 2), m32_mask);
    const uint16x8_t m32_8_15 = vandq_u16(m32_pack, m32_mask);
    const uint16x8_t m32_16_23 = vandq_u16(vshrq_n_u16(m32_pack, 2), m32_mask);
    const uint16x8_t m32_24_31 = vandq_u16(vshrq_n_u16(m32_pack, 4), m32_mask);

    uint16x8_t value0 = vorrq_u16(vorrq_u16(e8_0_7, sm_0_7), m32_0_7);
    uint16x8_t value1 = vorrq_u16(vorrq_u16(e8_8_15, sm_8_15), m32_8_15);
    uint16x8_t value2 = vorrq_u16(vorrq_u16(e8_16_23, sm_16_23), m32_16_23);
    uint16x8_t value3 = vorrq_u16(vorrq_u16(e8_24_31, sm_24_31), m32_24_31);

    if (Mode == OptionalJoinMode::M10) {
        const uint16x8_t m10_pack = vmovl_u8(vld1_u8(optional));
        const uint16x8_t m10_mask = vdupq_n_u16(0x0003U);
        value0 = vorrq_u16(value0, vandq_u16(m10_pack, m10_mask));
        value1 = vorrq_u16(value1, vandq_u16(vshrq_n_u16(m10_pack, 2), m10_mask));
        value2 = vorrq_u16(value2, vandq_u16(vshrq_n_u16(m10_pack, 4), m10_mask));
        value3 = vorrq_u16(value3, vshrq_n_u16(m10_pack, 6));
    }
    if (Mode == OptionalJoinMode::M1_ONLY) {
        const uint32_t m1 = LoadLE32(optional);
        value0 = vorrq_u16(value0, vld1q_u16(M1_EXPAND_LUT[m1 & 0xffU].data()));
        value1 = vorrq_u16(value1, vld1q_u16(M1_EXPAND_LUT[(m1 >> 8) & 0xffU].data()));
        value2 = vorrq_u16(value2, vld1q_u16(M1_EXPAND_LUT[(m1 >> 16) & 0xffU].data()));
        value3 = vorrq_u16(value3, vld1q_u16(M1_EXPAND_LUT[m1 >> 24].data()));
    }

    vst1q_u16(dst, value0);
    vst1q_u16(dst + 8, value1);
    vst1q_u16(dst + 16, value2);
    vst1q_u16(dst + 24, value3);
}

void JoinHighPrecisionNEON(uint16_t* dst, const uint8_t* e8, const ParsedR160& layout,
                           size_t n_bf16)
{
    const size_t block_count = n_bf16 / VALUES_PER_BLOCK;
    size_t block = 0;

    const size_t full_m10_blocks = layout.m0Groups / 4;
    for (; block < full_m10_blocks; ++block) {
        JoinHighPrecisionBlockNEON<OptionalJoinMode::M10>(
            dst + block * 32, e8 + block * 32, layout.sm4 + block * 16, layout.m32 + block * 8,
            layout.optional + block * 8);
    }
    if ((layout.m0Groups & 3U) != 0) {
        uint8_t mixed_m10[8];
        BuildOptionalM10Tile(mixed_m10, layout, block * 4);
        JoinHighPrecisionBlockNEON<OptionalJoinMode::M10>(dst + block * 32, e8 + block * 32,
                                                          layout.sm4 + block * 16,
                                                          layout.m32 + block * 8, mixed_m10);
        ++block;
    }

    const uint8_t* m1_only = layout.optional + layout.m0Groups * 2;
    const size_t full_m1_blocks = layout.m1Groups / 4;
    for (; block < full_m1_blocks; ++block) {
        JoinHighPrecisionBlockNEON<OptionalJoinMode::M1_ONLY>(
            dst + block * 32, e8 + block * 32, layout.sm4 + block * 16, layout.m32 + block * 8,
            m1_only + (block * 4 - layout.m0Groups));
    }
    if (block * 4 < layout.m1Groups) {
        uint8_t partial_m1[4] = {0, 0, 0, 0};
        std::memcpy(partial_m1, m1_only + (block * 4 - layout.m0Groups),
                    layout.m1Groups - block * 4);
        JoinHighPrecisionBlockNEON<OptionalJoinMode::M1_ONLY>(dst + block * 32, e8 + block * 32,
                                                              layout.sm4 + block * 16,
                                                              layout.m32 + block * 8, partial_m1);
        ++block;
    }

    for (; block < block_count; ++block) {
        JoinHighPrecisionBlockNEON<OptionalJoinMode::NONE>(dst + block * 32, e8 + block * 32,
                                                           layout.sm4 + block * 16,
                                                           layout.m32 + block * 8, nullptr);
    }
}

void JoinQuantizedGroupNEON(uint16_t* dst, const uint8_t* quantized, uint8_t m4, uint8_t m3,
                            int16x8_t shift_m4, int16x8_t shift_m3)
{
    const uint16x8_t packed = vmovl_u8(vld1_u8(quantized));
    uint16x8_t value = vaddq_u16(vshlq_n_u16(vandq_u16(packed, vdupq_n_u16(0x00f8U)), 4),
                                 vdupq_n_u16(static_cast<uint16_t>(BF16_QUANT_EXP_MIN) << 7));
    value = vorrq_u16(value, vshlq_n_u16(vandq_u16(packed, vdupq_n_u16(0x0004U)), 13));
    value = vorrq_u16(value, vshlq_n_u16(vandq_u16(packed, vdupq_n_u16(0x0003U)), 5));
    value = vorrq_u16(value, ExpandPlane(m4, shift_m4, 0x0010U));
    value = vorrq_u16(value, ExpandPlane(m3, shift_m3, 0x0008U));
    value = vorrq_u16(value, vdupq_n_u16(0x0003U));
    vst1q_u16(dst, value);
}

void JoinQuantizedNEON(uint16_t* dst, const ParsedR160& layout, size_t n_bf16)
{
    const int16_t shifts_m4_data[8] = {-3, -2, -1, 0, 1, 2, 3, 4};
    const int16_t shifts_m3_data[8] = {-4, -3, -2, -1, 0, 1, 2, 3};
    const int16x8_t shifts_m4 = vld1q_s16(shifts_m4_data);
    const int16x8_t shifts_m3 = vld1q_s16(shifts_m3_data);

    const size_t groups = PlaneBytes(n_bf16);
    size_t group = 0;
    for (; group < layout.quantM3Groups; ++group) {
        JoinQuantizedGroupNEON(dst + group * 8, layout.quantized + group * 8, layout.quantM4[group],
                               layout.quantM3[group], shifts_m4, shifts_m3);
    }
    for (; group < layout.quantM4Groups; ++group) {
        JoinQuantizedGroupNEON(dst + group * 8, layout.quantized + group * 8, layout.quantM4[group],
                               0, shifts_m4, shifts_m3);
    }
    for (; group < groups; ++group) {
        JoinQuantizedGroupNEON(dst + group * 8, layout.quantized + group * 8, 0, 0, shifts_m4,
                               shifts_m3);
    }
}
#endif

ParsedR160 SliceHighPrecisionLayout(const ParsedR160& layout, size_t value_offset,
                                    size_t value_count)
{
    ParsedR160 sliced = layout;
    sliced.sm4 = layout.sm4 + value_offset / 2;
    sliced.m32 = layout.m32 + value_offset / 4;

    const size_t plane_offset = value_offset / 8;
    const size_t plane_count = value_count / 8;
    if (plane_offset < layout.m0Groups) {
        sliced.optional = layout.optional + plane_offset * 2;
    } else {
        const size_t m1_offset = std::min(plane_offset, layout.m1Groups);
        sliced.optional = layout.optional + layout.m0Groups * 2 + (m1_offset - layout.m0Groups);
    }
    sliced.m0Groups =
        plane_offset < layout.m0Groups ? std::min(plane_count, layout.m0Groups - plane_offset) : 0;
    sliced.m1Groups =
        plane_offset < layout.m1Groups ? std::min(plane_count, layout.m1Groups - plane_offset) : 0;
    return sliced;
}

template <bool ValidateZeroLength>
bool DecodeAndJoinHighPrecision(uint16_t* dst, size_t n_bf16, const ParsedR160& layout,
                                uint8_t* e8_scratch)
{
    const auto* header = reinterpret_cast<const ts_header_t*>(layout.e8);
    DynamicE8Decoder decoder;
    decoder.marks = layout.e8 + sizeof(header->dynamic);
    decoder.marksRemaining = layout.dynamicE8.markCount;
    decoder.lut = header->dynamic.lut;
    uint8_t* output = e8_scratch;

    for (size_t offset = 0; offset < n_bf16; offset += E8_BLOCK_VALUES) {
        const size_t block_values = std::min(E8_BLOCK_VALUES, n_bf16 - offset);
        uint8_t* const logical_end = e8_scratch + block_values;
        uint8_t* const padded_end = logical_end + E8_DECODE_GUARD_BYTES;
        if (!DecodeDynamicE8Block<ValidateZeroLength>(output, logical_end, padded_end, decoder)) {
            return false;
        }
        const size_t carry = static_cast<size_t>(output - logical_end);
        if (carry > TS_ITEM_SIZE - 1) { return false; }

        const ParsedR160 block_layout = SliceHighPrecisionLayout(layout, offset, block_values);
#if defined(__aarch64__)
        JoinHighPrecisionNEON(dst + offset, e8_scratch, block_layout, block_values);
#else
        JoinHighPrecisionScalar(dst + offset, e8_scratch, block_layout, block_values);
#endif

        if (offset + block_values != n_bf16) {
            std::memmove(e8_scratch, logical_end, carry);
            output = e8_scratch + carry;
        }
    }
    // The encoder emits marks in pairs. If the first mark of the final pair
    // reaches the logical E8 length, the second padding mark is intentionally
    // left unread; anything beyond that is a malformed trailing mark stream.
    return decoder.marksRemaining <= 1;
}

int DecompressWithE8Scratch(uint16_t* dst, size_t n_bf16, const uint8_t* src, size_t src_len,
                            uint8_t* e8_scratch)
{
    if (dst == nullptr || src == nullptr || e8_scratch == nullptr) { return R_ERR_SRC_OVERFLOW; }

    ParsedR160 layout;
    const int layout_err = ParseLayout(src, src_len, n_bf16, layout);
    if (layout_err != R_TS_OK) { return layout_err; }

    if (layout.mode == R160PayloadMode::QUANTIZED) {
#if defined(__aarch64__)
        JoinQuantizedNEON(dst, layout, n_bf16);
#else
        JoinQuantizedScalar(dst, layout, n_bf16);
#endif
        return R_TS_OK;
    }

    const bool expanded = layout.dynamicE8.allLutEntriesPopulated
                              ? DecodeAndJoinHighPrecision<false>(dst, n_bf16, layout, e8_scratch)
                              : DecodeAndJoinHighPrecision<true>(dst, n_bf16, layout, e8_scratch);
    if (!expanded) { return R_ERR_R160_E8_EXPANSION; }
    return R_TS_OK;
}

bool CompressHighPrecision(uint8_t* dst, size_t stored_bytes, const uint16_t* src, size_t n_bf16,
                           std::vector<uint8_t>& encoded_e8)
{
    const size_t base_bytes = HighPrecisionBaseBytes(n_bf16);
    const size_t max_e8_region = stored_bytes - base_bytes;
    if (!EncodeDynamicE8(encoded_e8, src, n_bf16, max_e8_region)) { return false; }

    uint8_t* sm4 = dst;
    uint8_t* m32 = sm4 + SM4Bytes(n_bf16);
    uint8_t* e8_payload = m32 + M32Bytes(n_bf16);
    PackSM4(sm4, src, n_bf16);
    PackM32(m32, src, n_bf16);
    std::memcpy(e8_payload, encoded_e8.data(), encoded_e8.size());

    uint8_t* optional = e8_payload + encoded_e8.size();
    const size_t used_before_extra = static_cast<size_t>(optional - dst);
    const size_t remaining = stored_bytes - used_before_extra;
    const size_t plane_bytes = PlaneBytes(n_bf16);
    const size_t m1_groups = std::min(plane_bytes, remaining);
    const size_t m0_groups = std::min(plane_bytes, remaining - m1_groups);
    PackOptionalBits(optional, src, m1_groups, m0_groups);

    const size_t actual_bytes = used_before_extra + m1_groups + m0_groups;
    std::memset(dst + actual_bytes, 0, stored_bytes - actual_bytes);
    return true;
}

void CompressQuantized(uint8_t* dst, size_t stored_bytes, const uint16_t* src, size_t n_bf16)
{
    uint8_t* quantized = dst;
    for (size_t i = 0; i < n_bf16; ++i) { quantized[i] = QuantizeBF16(src[i]); }

    const size_t plane_bytes = PlaneBytes(n_bf16);
    const size_t optional_bytes = stored_bytes - n_bf16;
    const size_t m4_bytes = std::min(plane_bytes, optional_bytes);
    const size_t m3_bytes = std::min(plane_bytes, optional_bytes - m4_bytes);
    uint8_t* m4 = quantized + n_bf16;
    uint8_t* m3 = m4 + m4_bytes;
    PackPlaneBytes(m4, src, m4_bytes, 4);
    PackPlaneBytes(m3, src, m3_bytes, 3);

    // Reuse the high-precision mode position as the shard-mode discriminator.
    // Clearing this quantized-core bit loses M5 for exactly one BF16 value.
    dst[HighPrecisionBaseBytes(n_bf16)] &= 0xfeU;
}

}  // namespace

R160PayloadMode TunstallGetBF16R160Mode(const uint8_t* p_src, size_t src_len, size_t n_bf16)
{
    return ClassifyR160Payload(p_src, src_len, n_bf16);
}

int TunstallCompressBF16R160(uint8_t* p_dst, size_t* p_dst_len, const uint16_t* p_src,
                             size_t n_bf16)
{
    if (p_dst == nullptr || p_dst_len == nullptr || p_src == nullptr) { return R_ERR_SRC_OVERFLOW; }

    const size_t stored_bytes = *p_dst_len;
    if (!CheckedR160StreamSize(n_bf16, stored_bytes)) { return R_ERR_UNSUPPORT; }

    thread_local std::vector<uint8_t> encoded_e8;
    if (!CompressHighPrecision(p_dst, stored_bytes, p_src, n_bf16, encoded_e8)) {
        CompressQuantized(p_dst, stored_bytes, p_src, n_bf16);
    }

    *p_dst_len = stored_bytes;
    return R_TS_OK;
}

int TunstallDecompressBF16R160(const uint8_t* p_src, size_t src_len, uint16_t* p_dst, size_t n_bf16)
{
    if (p_src == nullptr || p_dst == nullptr ||
        reinterpret_cast<uintptr_t>(p_dst) % alignof(uint16_t) != 0) {
        return R_ERR_SRC_OVERFLOW;
    }
    if (!CheckedR160StreamSize(n_bf16, src_len)) { return R_ERR_R160_STREAM_SIZE; }
    thread_local std::vector<uint8_t> e8_scratch;
    e8_scratch.resize(std::min(E8_BLOCK_VALUES, n_bf16) + E8_DECODE_GUARD_BYTES);
    return DecompressWithE8Scratch(p_dst, n_bf16, p_src, src_len, e8_scratch.data());
}

int TunstallDecompressBF16R160Inplace(uint8_t* p_data, size_t n_bf16, size_t src_len)
{
    if (p_data == nullptr || reinterpret_cast<uintptr_t>(p_data) % alignof(uint16_t) != 0) {
        return R_ERR_SRC_OVERFLOW;
    }
    if (!CheckedR160StreamSize(n_bf16, src_len)) { return R_ERR_R160_STREAM_SIZE; }
    const size_t e8_scratch_bytes = std::min(E8_BLOCK_VALUES, n_bf16) + E8_DECODE_GUARD_BYTES;
    if (src_len > std::numeric_limits<size_t>::max() - e8_scratch_bytes) {
        return R_ERR_SRC_OVERFLOW;
    }

    thread_local std::vector<uint8_t> scratch;
    scratch.resize(src_len + e8_scratch_bytes);
    std::memcpy(scratch.data(), p_data, src_len);
    uint8_t* e8_scratch = scratch.data() + src_len;
    return DecompressWithE8Scratch(reinterpret_cast<uint16_t*>(p_data), n_bf16, scratch.data(),
                                   src_len, e8_scratch);
}
