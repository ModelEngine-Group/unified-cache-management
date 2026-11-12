/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string_view>
#include <vector>

namespace UC {

/**
 * @brief Content-addressed identifier of a KV cache block, carried as a 16-byte hash.
 */
struct BlockId {
    std::array<std::byte, 16> bytes{};
    bool operator==(const BlockId& other) const noexcept { return bytes == other.bytes; }
    bool operator!=(const BlockId& other) const noexcept { return !(*this == other); }
};

/**
 * @brief Resolved content of a KV cache block, ready for transfer.
 */
struct BlockPayload {
    /** Resolved xPU byte pointers of the block. */
    std::vector<void*> pointers;
    /** Byte length of each pointer. */
    std::vector<std::size_t> sizes;
    /** Offset of each pointer within the object, one-to-one with pointers/sizes. */
    std::vector<std::size_t> offsets;
};

}  // namespace UC

namespace std {

/** @brief Enables BlockId as a key in std::unordered_map / std::unordered_set. */
template <>
struct hash<UC::BlockId> {
    static std::uint64_t Mix64(std::uint64_t v) noexcept
    {
        v ^= v >> 33;
        v *= 0xff51afd7ed558ccdULL;
        v ^= v >> 33;
        v *= 0xc4ceb9fe1a85ec53ULL;
        v ^= v >> 33;
        return v;
    }

    std::size_t operator()(const UC::BlockId& blockId) const noexcept
    {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
        std::memcpy(&lo, blockId.bytes.data(), sizeof(lo));
        std::memcpy(&hi, blockId.bytes.data() + sizeof(lo), sizeof(hi));
        return static_cast<std::size_t>(Mix64(lo ^ (Mix64(hi) + 0x9e3779b97f4a7c15ULL)));
    }
};

}  // namespace std
