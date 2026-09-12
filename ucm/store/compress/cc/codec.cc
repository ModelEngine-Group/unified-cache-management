/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "codec.h"
#include <limits>
#include "compress_lib/tunstall_bf16_r160.h"
#include "compress_lib/tunstall_bf16_r200.h"

namespace UC::Compressor {

const char* CodecErrorName(int error) noexcept
{
    switch (error) {
        case R_TS_OK: return "success";
        case R_ERR_UNSUPPORT: return "unsupported codec input";
        case R_ERR_SYNTAX: return "invalid compressed stream";
        case R_ERR_SYMB_RANGE: return "symbol out of range";
        case R_ERR_SYMB_RANGE_PREDEF: return "predefined-table symbol out of range";
        case R_ERR_DST_OVERFLOW: return "destination buffer overflow";
        case R_ERR_SRC_OVERFLOW: return "source buffer invalid";
        case R_ERR_LUT_BUILD: return "Tunstall LUT build failed";
        case R_ERR_LUT_CHECK: return "Tunstall LUT validation failed";
        case R_ERR_LUT_MISMATCH: return "Tunstall LUT mismatch";
        case R_ERR_NORMALIZE: return "histogram normalization failed";
        case R_ERR_PREDEF_UNINIT: return "predefined tables not initialized";
        case R_ERR_LARGER: return "compression budget not met";
        case R_ERR_R160_STREAM_SIZE: return "R160 stream size/input length mismatch";
        case R_ERR_R160_E8_TAG: return "R160 fixed mode discriminator invalid";
        case R_ERR_R160_E8_METADATA: return "R160 E8 stream metadata invalid";
        case R_ERR_R160_E8_EXPANSION: return "R160 Tunstall expansion exceeds output";
        default: return "unknown codec error";
    }
}

// ===================================================================
// NoopCodec — R100 (1.00x), 无压缩
// ===================================================================
class NoopCodec : public Codec {
public:
    explicit NoopCodec(size_t compressedBytes) : compressedBytes_(compressedBytes) {}

    bool NeedsCompress() const override { return false; }
    bool NeedsDecompress() const override { return false; }

    size_t CompressedSize(size_t originalBytes) const override
    {
        return compressedBytes_ == originalBytes ? compressedBytes_ : 0;
    }
    size_t CompressScratchSize(size_t) const override { return 0; }

    int DecompressInplace(void*, size_t) const override { return 0; }

    size_t Compress(void*, const void*, size_t) const override { return 0; }

private:
    size_t compressedBytes_;
};

// ===================================================================
// Bf16R160Codec — R160 (1.6x), shard 级高精度/量化回退 BF16 算法
// ===================================================================
class Bf16R160Codec : public Codec {
public:
    explicit Bf16R160Codec(size_t compressedBytes) : compressedBytes_(compressedBytes) {}

    bool NeedsCompress() const override { return true; }
    bool NeedsDecompress() const override { return true; }

    size_t CompressedSize(size_t originalBytes) const override
    {
        constexpr size_t MIN_INPUT_BYTES = 128;
        constexpr size_t INPUT_BYTES_PER_GROUP = 64;
        if (originalBytes < MIN_INPUT_BYTES || originalBytes % INPUT_BYTES_PER_GROUP != 0 ||
            originalBytes > std::numeric_limits<size_t>::max() / 5) {
            return 0;
        }
        const size_t maximumBytes = originalBytes / 8 * 5;
        const size_t minimumBytes = originalBytes >> 1;
        if (compressedBytes_ < minimumBytes || compressedBytes_ > maximumBytes ||
            compressedBytes_ > std::numeric_limits<uint32_t>::max()) {
            return 0;
        }
        return compressedBytes_;
    }

    size_t CompressScratchSize(size_t originalBytes) const override
    {
        return CompressedSize(originalBytes);
    }

    int DecompressInplace(void* data, size_t originalBytes) const override
    {
        const size_t storedBytes = CompressedSize(originalBytes);
        return TunstallDecompressBF16R160Inplace(static_cast<uint8_t*>(data), originalBytes >> 1,
                                                 storedBytes);
    }

    size_t Compress(void* dst, const void* src, size_t originalBytes) const override
    {
        const size_t storedBytes = CompressedSize(originalBytes);
        size_t payloadBytes = storedBytes;
        const int err =
            TunstallCompressBF16R160(static_cast<uint8_t*>(dst), &payloadBytes,
                                     static_cast<const uint16_t*>(src), originalBytes >> 1);
        if (err != R_TS_OK) { return 0; }
        return payloadBytes;
    }

    CodecPayloadMode GetPayloadMode(const void* data, size_t payloadBytes,
                                    size_t originalBytes) const override
    {
        switch (TunstallGetBF16R160Mode(static_cast<const uint8_t*>(data), payloadBytes,
                                        originalBytes >> 1)) {
            case R160PayloadMode::HIGH_PRECISION: return CodecPayloadMode::R160_HIGH_PRECISION;
            case R160PayloadMode::QUANTIZED: return CodecPayloadMode::R160_QUANTIZED;
            case R160PayloadMode::INVALID: return CodecPayloadMode::INVALID;
        }
        return CodecPayloadMode::INVALID;
    }

private:
    size_t compressedBytes_;
};

// ===================================================================
// Bf16R200Codec — R200 (2.0x), BF16 Tunstall 算法
// ===================================================================
class Bf16R200Codec : public Codec {
public:
    explicit Bf16R200Codec(size_t compressedBytes) : compressedBytes_(compressedBytes) {}

    bool NeedsCompress() const override { return true; }
    bool NeedsDecompress() const override { return true; }

    size_t CompressedSize(size_t originalBytes) const override
    {
        // 2x 压缩: 原始 N 字节 → N/2 字节
        const size_t expectedBytes = originalBytes >> 1;
        return compressedBytes_ == expectedBytes ? compressedBytes_ : 0;
    }

    size_t CompressScratchSize(size_t originalBytes) const override
    {
        // TunstallCompressBF16 的 dst 需要 n_bf16*2 (= originalBytes) 字节
        // +4096 为对齐安全余量
        return originalBytes + 4096;
    }

    int DecompressInplace(void* data, size_t originalBytes) const override
    {
        // n_bf16 = originalBytes / 2 (每个 BF16 2 字节)
        return TunstallDecompressBF16Inplace(static_cast<uint8_t*>(data), originalBytes >> 1);
    }

    size_t Compress(void* dst, const void* src, size_t originalBytes) const override
    {
        size_t n_bf16 = originalBytes >> 1;
        int err = TunstallCompressBF16(static_cast<uint8_t*>(dst),
                                       static_cast<const uint16_t*>(src), n_bf16);
        if (err != 0) { return 0; }
        // 压缩后实际占用 n_bf16 字节
        return n_bf16;
    }

    CodecPayloadMode GetPayloadMode(const void* data, size_t payloadBytes, size_t) const override
    {
        switch (TunstallGetBF16R200Mode(static_cast<const uint8_t*>(data), payloadBytes)) {
            case R200PayloadMode::TUNSTALL: return CodecPayloadMode::R200_TUNSTALL;
            case R200PayloadMode::FP8_FALLBACK: return CodecPayloadMode::R200_FP8_FALLBACK;
            case R200PayloadMode::INVALID: return CodecPayloadMode::INVALID;
        }
        return CodecPayloadMode::INVALID;
    }

private:
    size_t compressedBytes_;
};

// ===================================================================
// MakeCodec — 工厂函数
// ===================================================================
std::unique_ptr<Codec> MakeCodec(FixedRatio ratio, DataType dataType, size_t compressedBytes)
{
    if (dataType != DT_BF16) { return nullptr; }
    if (ratio == R100) { return std::make_unique<NoopCodec>(compressedBytes); }

    switch (ratio) {
        case R160: return std::make_unique<Bf16R160Codec>(compressedBytes);
        case R200: return std::make_unique<Bf16R200Codec>(compressedBytes);
        default: return nullptr;
    }
}

}  // namespace UC::Compressor
