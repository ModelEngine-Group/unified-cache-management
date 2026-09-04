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

#include <cstdint>
#include <string>
#include <unordered_map>
#include "types.h"

namespace UC::Router {

constexpr std::uint64_t kDefaultVirtualNodeCount = 128;
constexpr std::uint64_t kDefaultMaglevTableSize = 65537;

// RouterType selects the routing strategy implementation.
enum class RouterType {
    RING_HASH_FULL_SPREAD = 0,
    MAGLEV_FULL_SPREAD = 1,
    CONTIGUOUS_BLOCK_AFFINITY = 2,
    BATCH_TOPK_AFFINITY = 3,
    RING_HASH = RING_HASH_FULL_SPREAD,
    MAGLEV = MAGLEV_FULL_SPREAD,
};

// RingHashConfig controls the full-spread ring-hash strategy.
struct RingHashConfig {
    std::uint64_t virtualNodeCount{kDefaultVirtualNodeCount};
};

// MaglevConfig controls the full-spread Maglev strategy.
struct MaglevConfig {
    std::uint64_t tableSize{kDefaultMaglevTableSize};
};

// ContiguousBlockAffinityConfig keeps each K-sized key range on the same node.
struct ContiguousBlockAffinityConfig {
    std::uint64_t blockCount{1};
    RouterType fullSpreadType{RouterType::RING_HASH_FULL_SPREAD};
    bool dynamicAdjustEnabled{false};
};

// BatchTopKAffinityConfig limits each batch to a TopK node candidate set.
struct BatchTopKAffinityConfig {
    std::uint64_t topK{1};
    bool dynamicAdjustEnabled{false};
};

// RouterConfig controls router construction and routing strategy parameters.
struct RouterConfig {
    RouterType type{RouterType::RING_HASH_FULL_SPREAD};
    RingHashConfig ringHash;
    MaglevConfig maglev;
    ContiguousBlockAffinityConfig contiguousBlockAffinity;
    BatchTopKAffinityConfig batchTopKAffinity;
};

// Builds a RouterConfig from a string-keyed attribute map.
// Returns ::UC::ASU::Status to report invalid attribute values; this cross-module
// dependency on ASU's Status type is a transitional layering wrinkle.
::UC::ASU::Status BuildRouterConfigFromAttrs(const std::unordered_map<std::string, std::string>& attrs,
                                             RouterConfig& config);

}  // namespace UC::Router
