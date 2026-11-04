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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <limits>
#include <memory>
#include <stdexcept>
#include "logger.h"

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
    if (bitsAfter == 0) { return NO_SPACE; }
    return tzcnt_nonzero(bitsAfter);
}

Allocator::Allocator(uint32 size, uint32 maxAllocs)
    : size_(size), maxAllocs_(maxAllocs), nodes_(nullptr), freeNodes_(nullptr)
{
    // Reject invalid metadata layouts before reset performs unsigned index arithmetic.
    if (size == 0) { throw std::invalid_argument("allocator size must be non-zero"); }
    if (maxAllocs < 3) { throw std::invalid_argument("maxAllocs must be at least 3"); }
    if (maxAllocs > static_cast<uint32>(std::numeric_limits<NodeIndex>::max())) {
        throw std::invalid_argument("maxAllocs exceeds NodeIndex range");
    }
    Reset();
}

Allocator::~Allocator()
{
    delete[] nodes_;
    delete[] freeNodes_;
}

void Allocator::Reset()
{
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Allocate replacements first so allocation failure leaves the current state intact.
    auto newNodes = std::make_unique<Node[]>(maxAllocs_);
    auto newFreeNodes = std::make_unique<NodeIndex[]>(maxAllocs_);
    for (uint32 index = 0; index < maxAllocs_; ++index) {
        newFreeNodes[index] = static_cast<NodeIndex>(maxAllocs_ - index - 1);
    }

    delete[] nodes_;
    delete[] freeNodes_;
    nodes_ = newNodes.release();
    freeNodes_ = newFreeNodes.release();

    freeStorage_ = 0;
    usedBinsTop_ = 0;
    freeOffset_ = maxAllocs_ - 1;

    for (uint32 index = 0; index < NUM_TOP_BINS; ++index) { usedBins_[index] = 0; }
    for (uint32 index = 0; index < NUM_LEAF_BINS; ++index) { binIndices_[index] = Node::unused; }

    // Initially the complete storage is one free node.
    InsertNodeIntoBin(size_, 0);
}

Allocation Allocator::Allocate(uint32 size)
{
    // Zero-sized allocations would consume metadata while repeatedly returning offset zero.
    if (size == 0 || size > size_) { return {NO_SPACE, NO_SPACE_NODE_INDEX}; }

    std::unique_lock<std::shared_mutex> lock(mutex_);

    uint32 binIndex = NO_SPACE;
    NodeIndex nodeIndex = Node::unused;
    if (freeOffset_ == 0) {
        // Metadata exhausted: only an exact-fit node can be allocated without splitting.
        const uint32 exactBinIndex = SmallFloat::uintToFloatRoundDown(size);
        NodeIndex candidateIndex = binIndices_[exactBinIndex];
        for (uint32 checked = 0; candidateIndex != Node::unused && checked < BIN_SCAN_LIMIT;
             ++checked) {
            if (nodes_[candidateIndex].dataSize == size) {
                binIndex = exactBinIndex;
                nodeIndex = candidateIndex;
                break;
            }
            candidateIndex = nodes_[candidateIndex].binListNext;
        }
    } else {
        const uint32 minBinIndex = SmallFloat::uintToFloatRoundUp(size);
        const uint32 minTopBinIndex = minBinIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 minLeafBinIndex = minBinIndex & LEAF_BINS_INDEX_MASK;

        uint32 topBinIndex = minTopBinIndex;
        uint32 leafBinIndex = NO_SPACE;
        if (usedBinsTop_ & (1U << topBinIndex)) {
            leafBinIndex = findLowestSetBitAfter(usedBins_[topBinIndex], minLeafBinIndex);
        }
        if (leafBinIndex == NO_SPACE) {
            topBinIndex = findLowestSetBitAfter(usedBinsTop_, minTopBinIndex + 1);
            if (topBinIndex != NO_SPACE) { leafBinIndex = tzcnt_nonzero(usedBins_[topBinIndex]); }
        }
        if (topBinIndex != NO_SPACE && leafBinIndex != NO_SPACE) {
            binIndex = (topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex;
            nodeIndex = binIndices_[binIndex];
        }

        // Bounded lower-bin fallback: recover fitting nodes hidden by round-down insertion.
        if (nodeIndex == Node::unused) {
            const uint32 lowerBinIndex = SmallFloat::uintToFloatRoundDown(size);
            if (lowerBinIndex != minBinIndex) {
                NodeIndex candidateIndex = binIndices_[lowerBinIndex];
                for (uint32 checked = 0; candidateIndex != Node::unused && checked < BIN_SCAN_LIMIT;
                     ++checked) {
                    if (nodes_[candidateIndex].dataSize >= size) {
                        binIndex = lowerBinIndex;
                        nodeIndex = candidateIndex;
                        break;
                    }
                    candidateIndex = nodes_[candidateIndex].binListNext;
                }
            }
        }
    }

    if (nodeIndex == Node::unused) { return {NO_SPACE, NO_SPACE_NODE_INDEX}; }

    Node& node = nodes_[nodeIndex];
    const uint32 nodeTotalSize = node.dataSize;
    if (nodeTotalSize < size) { return {NO_SPACE, NO_SPACE_NODE_INDEX}; }

    const uint32 remainderSize = nodeTotalSize - size;
    // Only splitting a free region consumes a spare metadata node.
    if (remainderSize > 0 && freeOffset_ == 0) { return {NO_SPACE, NO_SPACE_NODE_INDEX}; }

    node.dataSize = size;
    node.used = true;
    UnlinkNodeFromBin(binIndex, nodeIndex);
    freeStorage_ -= nodeTotalSize;
    UC_DEBUG("Free storage: {} (-{}) (allocate)", freeStorage_, nodeTotalSize);

    if (remainderSize > 0) {
        const NodeIndex newNodeIndex = InsertNodeIntoBin(remainderSize, node.dataOffset + size);
        if (node.neighborNext != Node::unused) {
            nodes_[node.neighborNext].neighborPrev = newNodeIndex;
        }
        nodes_[newNodeIndex].neighborPrev = nodeIndex;
        nodes_[newNodeIndex].neighborNext = node.neighborNext;
        node.neighborNext = newNodeIndex;
    }

    return {node.dataOffset, static_cast<NodeIndex>(nodeIndex)};
}

void Allocator::UnlinkNodeFromBin(uint32 binIndex, NodeIndex nodeIndex)
{
    Node& node = nodes_[nodeIndex];
    if (node.binListPrev != Node::unused) {
        nodes_[node.binListPrev].binListNext = node.binListNext;
    } else {
        assert(binIndices_[binIndex] == nodeIndex);
        binIndices_[binIndex] = node.binListNext;
    }
    if (node.binListNext != Node::unused) {
        nodes_[node.binListNext].binListPrev = node.binListPrev;
    }
    node.binListPrev = Node::unused;
    node.binListNext = Node::unused;

    // Bounded lower-bin fallback may remove a node from the middle of this list.
    if (binIndices_[binIndex] == Node::unused) {
        const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;
        usedBins_[topBinIndex] &= ~(1U << leafBinIndex);
        if (usedBins_[topBinIndex] == 0) { usedBinsTop_ &= ~(1U << topBinIndex); }
    }
}

bool Allocator::Free(Allocation allocation)
{
    // Validate the caller-owned handle before using its node index as an array index.
    if (allocation.nodeIndex == NO_SPACE_NODE_INDEX ||
        static_cast<uint32>(allocation.nodeIndex) >= maxAllocs_) {
        return false;
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!nodes_) { return false; }

    const NodeIndex nodeIndex = allocation.nodeIndex;
    Node& node = nodes_[nodeIndex];
    if (!node.used || node.dataOffset != allocation.offset) { return false; }

    uint32 offset = node.dataOffset;
    uint32 size = node.dataSize;

    if (node.neighborPrev != Node::unused && !nodes_[node.neighborPrev].used) {
        Node& previousNode = nodes_[node.neighborPrev];
        offset = previousNode.dataOffset;
        size += previousNode.dataSize;
        RemoveNodeFromBin(node.neighborPrev);
        assert(previousNode.neighborNext == nodeIndex);
        node.neighborPrev = previousNode.neighborPrev;
    }

    if (node.neighborNext != Node::unused && !nodes_[node.neighborNext].used) {
        Node& nextNode = nodes_[node.neighborNext];
        size += nextNode.dataSize;
        RemoveNodeFromBin(node.neighborNext);
        assert(nextNode.neighborPrev == nodeIndex);
        node.neighborNext = nextNode.neighborNext;
    }

    const NodeIndex neighborNext = node.neighborNext;
    const NodeIndex neighborPrev = node.neighborPrev;

    UC_DEBUG("Putting node {} into freelist[{}] (free)", static_cast<uint32>(nodeIndex),
             freeOffset_ + 1);
    freeNodes_[++freeOffset_] = nodeIndex;

    const NodeIndex combinedNodeIndex = InsertNodeIntoBin(size, offset);

    if (neighborNext != Node::unused) {
        nodes_[combinedNodeIndex].neighborNext = neighborNext;
        nodes_[neighborNext].neighborPrev = combinedNodeIndex;
    }
    if (neighborPrev != Node::unused) {
        nodes_[combinedNodeIndex].neighborPrev = neighborPrev;
        nodes_[neighborPrev].neighborNext = combinedNodeIndex;
    }

    return true;
}

NodeIndex Allocator::InsertNodeIntoBin(uint32 size, uint32 dataOffset)
{
    assert(freeOffset_ > 0);
    const uint32 binIndex = SmallFloat::uintToFloatRoundDown(size);
    const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
    const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

    if (binIndices_[binIndex] == Node::unused) {
        usedBins_[topBinIndex] |= 1U << leafBinIndex;
        usedBinsTop_ |= 1U << topBinIndex;
    }

    const NodeIndex topNodeIndex = binIndices_[binIndex];
    const NodeIndex nodeIndex = freeNodes_[freeOffset_--];
    UC_DEBUG("Getting node {} from freelist[{}]", static_cast<uint32>(nodeIndex), freeOffset_ + 1);
    Node newNode;
    newNode.dataOffset = dataOffset;
    newNode.dataSize = size;
    newNode.binListNext = topNodeIndex;
    nodes_[nodeIndex] = newNode;
    if (topNodeIndex != Node::unused) { nodes_[topNodeIndex].binListPrev = nodeIndex; }
    binIndices_[binIndex] = nodeIndex;

    freeStorage_ += size;
    UC_DEBUG("Free storage: {} (+{}) (insertNodeIntoBin)", freeStorage_, size);

    return nodeIndex;
}

void Allocator::RemoveNodeFromBin(NodeIndex nodeIndex)
{
    Node& node = nodes_[nodeIndex];

    if (node.binListPrev != Node::unused) {
        nodes_[node.binListPrev].binListNext = node.binListNext;
        if (node.binListNext != Node::unused) {
            nodes_[node.binListNext].binListPrev = node.binListPrev;
        }
    } else {
        const uint32 binIndex = SmallFloat::uintToFloatRoundDown(node.dataSize);
        const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
        const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

        binIndices_[binIndex] = node.binListNext;
        if (node.binListNext != Node::unused) {
            nodes_[node.binListNext].binListPrev = Node::unused;
        }

        if (binIndices_[binIndex] == Node::unused) {
            usedBins_[topBinIndex] &= ~(1U << leafBinIndex);
            if (usedBins_[topBinIndex] == 0) { usedBinsTop_ &= ~(1U << topBinIndex); }
        }
    }

    UC_DEBUG("Putting node {} into freelist[{}] (removeNodeFromBin)",
             static_cast<uint32>(nodeIndex), freeOffset_ + 1);
    freeNodes_[++freeOffset_] = nodeIndex;
    freeStorage_ -= node.dataSize;
    UC_DEBUG("Free storage: {} (-{}) (removeNodeFromBin)", freeStorage_, node.dataSize);
}

uint32 Allocator::GetAllocationSize(Allocation allocation) const
{
    if (allocation.nodeIndex == NO_SPACE_NODE_INDEX ||
        static_cast<uint32>(allocation.nodeIndex) >= maxAllocs_) {
        return 0;
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!nodes_) { return 0; }

    const Node& node = nodes_[allocation.nodeIndex];
    if (!node.used || node.dataOffset != allocation.offset) { return 0; }
    return node.dataSize;
}

StorageReport Allocator::GetStorageReport() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    uint32 largestFreeRegion = 0;
    const uint32 freeStorage = freeStorage_;

    if (usedBinsTop_) {
        const uint32 topBinIndex = 31 - lzcnt_nonzero(usedBinsTop_);
        const uint32 leafBinIndex = 31 - lzcnt_nonzero(usedBins_[topBinIndex]);
        largestFreeRegion =
            SmallFloat::floatToUint((topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex);
        assert(freeStorage >= largestFreeRegion);
    }

    return {freeStorage, largestFreeRegion};
}

StorageReportFull Allocator::GetStorageReportFull() const
{
    std::shared_lock<std::shared_mutex> lock(mutex_);

    StorageReportFull report{};
    for (uint32 index = 0; index < NUM_LEAF_BINS; ++index) {
        uint32 count = 0;
        NodeIndex nodeIndex = binIndices_[index];
        while (nodeIndex != Node::unused) {
            nodeIndex = nodes_[nodeIndex].binListNext;
            ++count;
        }
        report.freeRegions[index] = {SmallFloat::floatToUint(index), count};
    }
    return report;
}

}  // namespace OffsetAllocator
