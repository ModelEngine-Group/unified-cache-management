/*
 * Copyright (c) 2023 Sebastian Aaltonen
 * Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Derived from https://github.com/sebbbi/OffsetAllocator
 * Upstream commit: 3610a7377088b1e8c8f1525f458c96038a4e6fc0
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
 */
#pragma once

#include <limits>

namespace OffsetAllocator {

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;

// 16-bit node indices halve the metadata storage cost, but limit the node table to 65535 slots.
#ifdef USE_16_BIT_NODE_INDICES
using NodeIndex = uint16;
static constexpr uint32 DEFAULT_MAX_ALLOCS = 65535;
#else
using NodeIndex = uint32;
static constexpr uint32 DEFAULT_MAX_ALLOCS = 128 * 1024;
#endif

static constexpr uint32 NUM_TOP_BINS = 32;
static constexpr uint32 BINS_PER_LEAF = 8;
static constexpr uint32 TOP_BINS_INDEX_SHIFT = 3;
static constexpr uint32 LEAF_BINS_INDEX_MASK = 0x7;
static constexpr uint32 NUM_LEAF_BINS = NUM_TOP_BINS * BINS_PER_LEAF;

struct Allocation {
    static constexpr uint32 NO_SPACE = 0xffffffff;
    static constexpr NodeIndex NO_SPACE_METADATA = std::numeric_limits<NodeIndex>::max();

    uint32 offset = NO_SPACE;
    NodeIndex metadata = NO_SPACE_METADATA;
};

struct StorageReport {
    uint32 totalFreeSpace;
    uint32 largestFreeRegion;
};

struct StorageReportFull {
    struct Region {
        uint32 size;
        uint32 count;
    };

    Region freeRegions[NUM_LEAF_BINS];
};

class Allocator {
public:
    Allocator(uint32 size, uint32 maxAllocs = DEFAULT_MAX_ALLOCS);
    ~Allocator();

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;
    // Allocator ownership remains bound to the arena that provides synchronization.
    Allocator(Allocator&&) = delete;
    Allocator& operator=(Allocator&&) = delete;

    void reset();

    Allocation allocate(uint32 size);
    bool free(Allocation allocation);

    uint32 allocationSize(Allocation allocation) const;
    StorageReport storageReport() const;
    StorageReportFull storageReportFull() const;

private:
    NodeIndex insertNodeIntoBin(uint32 size, uint32 dataOffset);
    void unlinkNodeFromBin(uint32 binIndex, NodeIndex nodeIndex);
    void removeNodeFromBin(NodeIndex nodeIndex);

    struct Node {
        static constexpr NodeIndex unused = std::numeric_limits<NodeIndex>::max();

        uint32 dataOffset = 0;
        uint32 dataSize = 0;
        NodeIndex binListPrev = unused;
        NodeIndex binListNext = unused;
        NodeIndex neighborPrev = unused;
        NodeIndex neighborNext = unused;
        bool used = false;
    };

    uint32 m_size;
    uint32 m_maxAllocs;
    uint32 m_freeStorage;

    uint32 m_usedBinsTop;
    uint8 m_usedBins[NUM_TOP_BINS];
    NodeIndex m_binIndices[NUM_LEAF_BINS];

    Node* m_nodes;
    NodeIndex* m_freeNodes;
    uint32 m_freeOffset;
};

}  // namespace OffsetAllocator
