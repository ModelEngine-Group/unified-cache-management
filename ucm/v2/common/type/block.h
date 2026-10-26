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
    /**
     * Optional: when MemoryRegions were registered at initialization, the index of the
     * MemoryRegion each pointer belongs to.
     */
    std::vector<std::size_t> memoryRegionIndices;
    /**
     * Optional: when groups were registered at initialization, the number of pointers
     * in each group.
     */
    std::vector<std::size_t> groupSizes;
};

}  // namespace UC

namespace std {

/** @brief Enables BlockId as a key in std::unordered_map / std::unordered_set. */
template <>
struct hash<UC::BlockId> {
    std::size_t operator()(const UC::BlockId& blockId) const noexcept
    {
        std::string_view sv(reinterpret_cast<const char*>(blockId.bytes.data()),
                            blockId.bytes.size());
        return std::hash<std::string_view>{}(sv);
    }
};

}  // namespace std
