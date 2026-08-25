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
#ifndef UNIFIEDCACHE_COMPRESSOR_CC_CODEC_H
#define UNIFIEDCACHE_COMPRESSOR_CC_CODEC_H

#include <cstddef>
#include <memory>
#include "compress_lib/compress_types.h"

namespace UC::Compressor {

enum class CodecPayloadMode {
    NOT_APPLICABLE,
    R160_HIGH_PRECISION,
    R160_QUANTIZED,
    R200_TUNSTALL,
    R200_FP8_FALLBACK,
    INVALID,
};

/**
 * @brief 编解码器抽象接口。
 *
 * 封装 (压缩率, 数据类型) 组合的所有差异化行为：
 *  - 是否需要压缩 / 解压
 *  - 压缩后大小计算
 *  - 压缩临时缓冲区大小
 *  - 原地解压
 *  - 压缩
 *
 * 新增压缩率或数据类型只需实现新的子类并在 MakeCodec() 中注册。
 */
class Codec {
public:
    virtual ~Codec() = default;

    /// 是否需要压缩 / 解压 (R100 返回 false)
    virtual bool NeedsCompress() const = 0;
    virtual bool NeedsDecompress() const = 0;

    /**
     * @brief 根据原始字节数计算压缩后字节数。
     * @param originalBytes 原始未压缩数据大小 (shardSize)
     */
    virtual size_t CompressedSize(size_t originalBytes) const = 0;

    /**
     * @brief 压缩操作所需的临时缓冲区字节数。
     * @param originalBytes 原始未压缩数据大小
     */
    virtual size_t CompressScratchSize(size_t originalBytes) const = 0;

    /**
     * @brief 原地解压。
     *
     * 调用前: data[0 : CompressedSize(originalBytes)] 是压缩数据,
     *         data 总容量为 originalBytes 字节。
     * 调用后: data[0 : originalBytes] 是解压后的数据。
     *
     * @param data          缓冲区指针
     * @param originalBytes 解压后的原始数据大小
     * @return 0 成功，非 0 失败
     */
    virtual int DecompressInplace(void* data, size_t originalBytes) const = 0;

    /**
     * @brief 压缩数据到目标缓冲区。
     *
     * @param dst           目标缓冲区，至少有 CompressScratchSize(originalBytes) 字节
     * @param src           源数据，originalBytes 字节
     * @param originalBytes 原始未压缩数据大小
     * @return 实际压缩后字节数，0 表示失败
     */
    virtual size_t Compress(void* dst, const void* src, size_t originalBytes) const = 0;

    /// Infer the codec-specific mode from the compressed payload.
    virtual CodecPayloadMode GetPayloadMode(const void*, size_t, size_t) const
    {
        return CodecPayloadMode::NOT_APPLICABLE;
    }
};

/**
 * @brief 根据压缩率和数据类型创建对应的编解码器。
 *
 * @return 成功返回编解码器实例，不支持的组合返回 nullptr。
 */
std::unique_ptr<Codec> MakeCodec(FixedRatio ratio, DataType dataType, size_t compressedBytes);

/// Human-readable name for codec and R160 decode errors.
const char* CodecErrorName(int error) noexcept;

}  // namespace UC::Compressor

#endif
