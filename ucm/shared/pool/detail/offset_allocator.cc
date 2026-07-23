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
#include "pool/detail/offset_allocator.h"
#include <cassert>
#ifdef DEBUG_VERBOSE
#include <cstdio>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <limits>
#include <memory>
#include <stdexcept>

namespace OffsetAllocator {

static constexpr uint32 BIN_SCAN_LIMIT = 8;

inline uint32 lzcnt_nonzero(uint32 value)
{
#ifdef _MSC_VER
    unsigned long result;
    _BitScanReverse(&result, value);
    return 31 - result;
#else
    return static_cast<uint32>(__builtin_clz(value));
#endif
}

inline uint32 tzcnt_nonzero(uint32 value)
{
#ifdef _MSC_VER
    unsigned long result;
    _BitScanForward(&result, value);
    return result;
#else
    return static_cast<uint32>(__builtin_ctz(value));
#endif
}

namespace SmallFloat {

static constexpr uint32 MANTISSA_BITS = 3;
static constexpr uint32 MANTISSA_VALUE = 1 << MANTISSA_BITS;
static constexpr uint32 MANTISSA_MASK = MANTISSA_VALUE - 1;

// Bin sizes use a piecewise-linear logarithmic distribution.
uint32 uintToFloatRoundUp(uint32 size)
{
    uint32 exponent = 0;
    uint32 mantissa = 0;

    if (size < MANTISSA_VALUE) {
        mantissa = size;
    } else {
        const uint32 leadingZeros = lzcnt_nonzero(size);
        const uint32 highestSetBit = 31 - leadingZeros;
        const uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
        exponent = mantissaStartBit + 1;
        mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;

        const uint32 lowBitsMask = (1U << mantissaStartBit) - 1;
        if ((size & lowBitsMask) != 0) { ++mantissa; }
    }

    return (exponent << MANTISSA_BITS) + mantissa;
}

uint32 uintToFloatRoundDown(uint32 size)
{
    uint32 exponent = 0;
    uint32 mantissa = 0;

    if (size < MANTISSA_VALUE) {
        mantissa = size;
    } else {
        const uint32 leadingZeros = lzcnt_nonzero(size);
        const uint32 highestSetBit = 31 - leadingZeros;
        const uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
        exponent = mantissaStartBit + 1;
        mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;
    }

    return (exponent << MANTISSA_BITS) | mantissa;
}

uint32 floatToUint(uint32 floatValue)
{
    const uint32 exponent = floatValue >> MANTISSA_BITS;
    const uint32 mantissa = floatValue & MANTISSA_MASK;
    if (exponent == 0) { return mantissa; }
    return (mantissa | MANTISSA_VALUE) << (exponent - 1);
}

}  // namespace SmallFloat

uint32 findLowestSetBitAfter(uint32 bitMask, uint32 startBitIndex)
{
    const uint32 maskBeforeStartIndex = (1U << startBitIndex) - 1;
    const uint32 maskAfterStartIndex = ~maskBeforeStartIndex;
    const uint32 bitsAfter = bitMask & maskAfterStartIndex;
    if (bitsAfter == 0) { return Allocation::NO_SPACE; }
    return tzcnt_nonzero(bitsAfter);
}

Allocator::Allocator(uint32 size, uint32 maxAllocs)
    : m_size(size), m_maxAllocs(maxAllocs), m_nodes(nullptr), m_freeNodes(nullptr)
{
    // Reject invalid metadata layouts before reset performs unsigned index arithmetic.
    if (size == 0) { throw std::invalid_argument("allocator size must be non-zero"); }
    if (maxAllocs < 3) { throw std::invalid_argument("maxAllocs must be at least 3"); }
    if (maxAllocs > static_cast<uint32>(std::numeric_limits<NodeIndex>::max())) {
        throw std::invalid_argument("maxAllocs exceeds NodeIndex range");
    }
    reset();
}

Allocator::~Allocator()
{
    delete[] m_nodes;
    delete[] m_freeNodes;
}

void Allocator::reset()
{
    // Allocate replacements first so allocation failure leaves the current state intact.
    auto newNodes = std::make_unique<Node[]>(m_maxAllocs);
    auto newFreeNodes = std::make_unique<NodeIndex[]>(m_maxAllocs);
    for (uint32 index = 0; index < m_maxAllocs; ++index) {
        newFreeNodes[index] = static_cast<NodeIndex>(m_maxAllocs - index - 1);
    }

    delete[] m_nodes;
    delete[] m_freeNodes;
    m_nodes = newNodes.release();
    m_freeNodes = newFreeNodes.release();

    m_freeStorage = 0;
    m_usedBinsTop = 0;
    m_freeOffset = m_maxAllocs - 1;

    for (uint32 index = 0; index < NUM_TOP_BINS; ++index) { m_usedBins[index] = 0; }
    for (uint32 index = 0; index < NUM_LEAF_BINS; ++index) { m_binIndices[index] = Node::unused; }

    // Initially the complete storage is one free node.
    insertNodeIntoBin(m_size, 0);
}

Allocation Allocator::allocate(uint32 size)
{
    // Zero-sized allocations would consume metadata while repeatedly returning offset zero.
    if (size == 0 || size > m_size) {
        return {Allocation::NO_SPACE, Allocation::NO_SPACE_METADATA};
    }
    uint32 binIndex = Allocation::NO_SPACE;
    NodeIndex nodeIndex = Node::unused;
    if (m_freeOffset == 0) {
        // Metadata exhausted: only an exact-fit node can be allocated without splitting.
        const uint32 exactBinIndex = SmallFloat::uintToFloatRoundDown(size);
        NodeIndex candidateIndex = m_binIndices[exactBinIndex];
        for (uint32 checked = 0; candidateIndex != Node::unused && checked < BIN_SCAN_LIMIT;
             ++checked) {
            if (m_nodes[candidateIndex].dataSize == size) {
                binIndex = exactBinIndex;
                nodeIndex = candidateIndex;
                break;
            }
            candidateIndex = m_nodes[candidateIndex].binListNext;
        }
    } else {
        const uint32 minBinIndex = SmallFloat::uintToFloatRoundUp(size);
        const uint32 minTopBinIndex = minBinIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 minLeafBinIndex = minBinIndex & LEAF_BINS_INDEX_MASK;

        uint32 topBinIndex = minTopBinIndex;
        uint32 leafBinIndex = Allocation::NO_SPACE;
        if (m_usedBinsTop & (1U << topBinIndex)) {
            leafBinIndex = findLowestSetBitAfter(m_usedBins[topBinIndex], minLeafBinIndex);
        }
        if (leafBinIndex == Allocation::NO_SPACE) {
            topBinIndex = findLowestSetBitAfter(m_usedBinsTop, minTopBinIndex + 1);
            if (topBinIndex != Allocation::NO_SPACE) {
                leafBinIndex = tzcnt_nonzero(m_usedBins[topBinIndex]);
            }
        }
        if (topBinIndex != Allocation::NO_SPACE && leafBinIndex != Allocation::NO_SPACE) {
            binIndex = (topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex;
            nodeIndex = m_binIndices[binIndex];
        }

        // Bounded lower-bin fallback: recover fitting nodes hidden by round-down insertion.
        if (nodeIndex == Node::unused) {
            const uint32 lowerBinIndex = SmallFloat::uintToFloatRoundDown(size);
            if (lowerBinIndex != minBinIndex) {
                NodeIndex candidateIndex = m_binIndices[lowerBinIndex];
                for (uint32 checked = 0; candidateIndex != Node::unused && checked < BIN_SCAN_LIMIT;
                     ++checked) {
                    if (m_nodes[candidateIndex].dataSize >= size) {
                        binIndex = lowerBinIndex;
                        nodeIndex = candidateIndex;
                        break;
                    }
                    candidateIndex = m_nodes[candidateIndex].binListNext;
                }
            }
        }
    }

    if (nodeIndex == Node::unused) { return {Allocation::NO_SPACE, Allocation::NO_SPACE_METADATA}; }

    Node& node = m_nodes[nodeIndex];
    const uint32 nodeTotalSize = node.dataSize;
    if (nodeTotalSize < size) { return {Allocation::NO_SPACE, Allocation::NO_SPACE_METADATA}; }

    const uint32 remainderSize = nodeTotalSize - size;
    // Only splitting a free region consumes a spare metadata node.
    if (remainderSize > 0 && m_freeOffset == 0) {
        return {Allocation::NO_SPACE, Allocation::NO_SPACE_METADATA};
    }

    node.dataSize = size;
    node.used = true;
    unlinkNodeFromBin(binIndex, nodeIndex);
    m_freeStorage -= nodeTotalSize;
#ifdef DEBUG_VERBOSE
    std::printf("Free storage: %u (-%u) (allocate)\n", m_freeStorage, nodeTotalSize);
#endif

    if (remainderSize > 0) {
        const NodeIndex newNodeIndex = insertNodeIntoBin(remainderSize, node.dataOffset + size);
        if (node.neighborNext != Node::unused) {
            m_nodes[node.neighborNext].neighborPrev = newNodeIndex;
        }
        m_nodes[newNodeIndex].neighborPrev = nodeIndex;
        m_nodes[newNodeIndex].neighborNext = node.neighborNext;
        node.neighborNext = newNodeIndex;
    }

    return {node.dataOffset, static_cast<NodeIndex>(nodeIndex)};
}

void Allocator::unlinkNodeFromBin(uint32 binIndex, NodeIndex nodeIndex)
{
    Node& node = m_nodes[nodeIndex];
    if (node.binListPrev != Node::unused) {
        m_nodes[node.binListPrev].binListNext = node.binListNext;
    } else {
        assert(m_binIndices[binIndex] == nodeIndex);
        m_binIndices[binIndex] = node.binListNext;
    }
    if (node.binListNext != Node::unused) {
        m_nodes[node.binListNext].binListPrev = node.binListPrev;
    }
    node.binListPrev = Node::unused;
    node.binListNext = Node::unused;

    // Bounded lower-bin fallback may remove a node from the middle of this list.
    if (m_binIndices[binIndex] == Node::unused) {
        const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;
        m_usedBins[topBinIndex] &= ~(1U << leafBinIndex);
        if (m_usedBins[topBinIndex] == 0) { m_usedBinsTop &= ~(1U << topBinIndex); }
    }
}

bool Allocator::free(Allocation allocation)
{
    // Validate the caller-owned handle before using metadata as an array index.
    if (!m_nodes || allocation.metadata == Allocation::NO_SPACE_METADATA ||
        static_cast<uint32>(allocation.metadata) >= m_maxAllocs) {
        return false;
    }

    const NodeIndex nodeIndex = allocation.metadata;
    Node& node = m_nodes[nodeIndex];
    if (!node.used || node.dataOffset != allocation.offset) { return false; }

    uint32 offset = node.dataOffset;
    uint32 size = node.dataSize;

    if (node.neighborPrev != Node::unused && !m_nodes[node.neighborPrev].used) {
        Node& previousNode = m_nodes[node.neighborPrev];
        offset = previousNode.dataOffset;
        size += previousNode.dataSize;
        removeNodeFromBin(node.neighborPrev);
        assert(previousNode.neighborNext == nodeIndex);
        node.neighborPrev = previousNode.neighborPrev;
    }

    if (node.neighborNext != Node::unused && !m_nodes[node.neighborNext].used) {
        Node& nextNode = m_nodes[node.neighborNext];
        size += nextNode.dataSize;
        removeNodeFromBin(node.neighborNext);
        assert(nextNode.neighborPrev == nodeIndex);
        node.neighborNext = nextNode.neighborNext;
    }

    const NodeIndex neighborNext = node.neighborNext;
    const NodeIndex neighborPrev = node.neighborPrev;

#ifdef DEBUG_VERBOSE
    std::printf("Putting node %u into freelist[%u] (free)\n", static_cast<uint32>(nodeIndex),
                m_freeOffset + 1);
#endif
    m_freeNodes[++m_freeOffset] = nodeIndex;

    const NodeIndex combinedNodeIndex = insertNodeIntoBin(size, offset);

    if (neighborNext != Node::unused) {
        m_nodes[combinedNodeIndex].neighborNext = neighborNext;
        m_nodes[neighborNext].neighborPrev = combinedNodeIndex;
    }
    if (neighborPrev != Node::unused) {
        m_nodes[combinedNodeIndex].neighborPrev = neighborPrev;
        m_nodes[neighborPrev].neighborNext = combinedNodeIndex;
    }

    return true;
}

NodeIndex Allocator::insertNodeIntoBin(uint32 size, uint32 dataOffset)
{
    assert(m_freeOffset > 0);
    const uint32 binIndex = SmallFloat::uintToFloatRoundDown(size);
    const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
    const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

    if (m_binIndices[binIndex] == Node::unused) {
        m_usedBins[topBinIndex] |= 1U << leafBinIndex;
        m_usedBinsTop |= 1U << topBinIndex;
    }

    const NodeIndex topNodeIndex = m_binIndices[binIndex];
    const NodeIndex nodeIndex = m_freeNodes[m_freeOffset--];
#ifdef DEBUG_VERBOSE
    std::printf("Getting node %u from freelist[%u]\n", static_cast<uint32>(nodeIndex),
                m_freeOffset + 1);
#endif
    Node newNode;
    newNode.dataOffset = dataOffset;
    newNode.dataSize = size;
    newNode.binListNext = topNodeIndex;
    m_nodes[nodeIndex] = newNode;
    if (topNodeIndex != Node::unused) { m_nodes[topNodeIndex].binListPrev = nodeIndex; }
    m_binIndices[binIndex] = nodeIndex;

    m_freeStorage += size;
#ifdef DEBUG_VERBOSE
    std::printf("Free storage: %u (+%u) (insertNodeIntoBin)\n", m_freeStorage, size);
#endif

    return nodeIndex;
}

void Allocator::removeNodeFromBin(NodeIndex nodeIndex)
{
    Node& node = m_nodes[nodeIndex];

    if (node.binListPrev != Node::unused) {
        m_nodes[node.binListPrev].binListNext = node.binListNext;
        if (node.binListNext != Node::unused) {
            m_nodes[node.binListNext].binListPrev = node.binListPrev;
        }
    } else {
        const uint32 binIndex = SmallFloat::uintToFloatRoundDown(node.dataSize);
        const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

        m_binIndices[binIndex] = node.binListNext;
        if (node.binListNext != Node::unused) {
            m_nodes[node.binListNext].binListPrev = Node::unused;
        }

        if (m_binIndices[binIndex] == Node::unused) {
            m_usedBins[topBinIndex] &= ~(1U << leafBinIndex);
            if (m_usedBins[topBinIndex] == 0) { m_usedBinsTop &= ~(1U << topBinIndex); }
        }
    }

#ifdef DEBUG_VERBOSE
    std::printf("Putting node %u into freelist[%u] (removeNodeFromBin)\n",
                static_cast<uint32>(nodeIndex), m_freeOffset + 1);
#endif
    m_freeNodes[++m_freeOffset] = nodeIndex;
    m_freeStorage -= node.dataSize;
#ifdef DEBUG_VERBOSE
    std::printf("Free storage: %u (-%u) (removeNodeFromBin)\n", m_freeStorage, node.dataSize);
#endif
}

uint32 Allocator::allocationSize(Allocation allocation) const
{
    if (allocation.metadata == Allocation::NO_SPACE_METADATA || !m_nodes ||
        static_cast<uint32>(allocation.metadata) >= m_maxAllocs) {
        return 0;
    }
    const Node& node = m_nodes[allocation.metadata];
    if (!node.used || node.dataOffset != allocation.offset) { return 0; }
    return node.dataSize;
}

StorageReport Allocator::storageReport() const
{
    uint32 largestFreeRegion = 0;
    const uint32 freeStorage = m_freeStorage;

    if (m_usedBinsTop) {
        const uint32 topBinIndex = 31 - lzcnt_nonzero(m_usedBinsTop);
        const uint32 leafBinIndex = 31 - lzcnt_nonzero(m_usedBins[topBinIndex]);
        largestFreeRegion =
            SmallFloat::floatToUint((topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex);
        assert(freeStorage >= largestFreeRegion);
    }

    return {freeStorage, largestFreeRegion};
}

StorageReportFull Allocator::storageReportFull() const
{
    StorageReportFull report{};
    for (uint32 index = 0; index < NUM_LEAF_BINS; ++index) {
        uint32 count = 0;
        NodeIndex nodeIndex = m_binIndices[index];
        while (nodeIndex != Node::unused) {
            nodeIndex = m_nodes[nodeIndex].binListNext;
            ++count;
        }
        report.freeRegions[index] = {SmallFloat::floatToUint(index), count};
    }
    return report;
}

}  // namespace OffsetAllocator
