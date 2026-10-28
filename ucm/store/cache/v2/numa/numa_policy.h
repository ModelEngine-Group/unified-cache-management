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

#include <cstddef>
#include <cstdint>
#include <optional>
#include "status/status.h"

namespace UC::Cache2::Numa {

/* How the local data segment's node was chosen. This is an internal decision
 * result, never a user-facing policy: there is no config switch and no node
 * list. */
enum class Placement {
    Affinity,  /* device affinity detected: place on that node */
    RankSlice, /* no affinity: contiguous worker-rank slices per NUMA node */
};

/* Resolved placement of one local data segment. The whole segment goes on a
 * single node; it is never split per page or per block. */
struct Plan {
    Placement placement{Placement::Affinity};
    int32_t node{-1};
};

/* Picks the node for the local segment before BindLocal. deviceNode is the
 * device-layer probe result and stays empty when unknown -- this module never
 * fabricates an affinity. fallbackNumaRank is the connector-derived local
 * worker rank, not the control-plane myRank; a missing one is an error, never
 * a silent 0. rankCount is the number of workers on the machine and defines
 * the contiguous rank slices; it can be larger than the current Cache domain.
 * The result stays fixed for the segment's lifetime. */
Expected<size_t> RankSliceIndex(size_t rank, size_t rankCount, size_t nodeCount);

Expected<Plan> Resolve(std::optional<int32_t> deviceNode, std::optional<size_t> fallbackNumaRank,
                       size_t rankCount);

/* Binds the whole local segment to plan.node before its first touch and before
 * device registration. Registration pins pages, so binding afterwards would
 * lock them onto the wrong node. Memfd path. */
Status BindBeforeTouch(void* base, size_t bytes, const Plan& plan);

/* Bounded sampling of actual page placement, for diagnostics only: a failed
 * query must not affect service. */
Status Verify(void* base, size_t bytes, const Plan& plan);

}  // namespace UC::Cache2::Numa
