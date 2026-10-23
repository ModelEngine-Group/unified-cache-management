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
#include <vector>
#include "status/status.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

class TokenAddressed {
public:
    virtual ~TokenAddressed() = default;
    /**
     * @brief Pins the given blocks so they cannot be evicted, and mints
     *        their read-end token.
     *
     * @param blockIds Blocks to pin; each must be present.
     *
     * @return Expected the read-end token on success: keys echoes blockIds,
     *         payload packs the transfer-end addressing the worker side
     *         needs — empty when it resolves blockIds on its own — and
     *         stays valid while the pin is held. Error status (e.g.
     *         NotFound) otherwise.
     *
     * @note Implementations must guarantee atomicity: if Pin fails, none of
     *       the listed blocks remain pinned.
     */
    virtual Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) = 0;
    /**
     * @brief Releases pins previously taken via Pin.
     *
     * @param blockIds Blocks to unpin.
     *
     * @note Must be idempotent: unpinning blocks that are not pinned is a
     *       no-op, hence there is no Status return.
     */
    virtual void Unpin(const std::vector<BlockId>& blockIds) = 0;
    /**
     * @brief Reserves capacity for blocks that are about to enter this Store.
     *
     * @param blockIds Blocks to reserve space for.
     * @param context  Request facts of the requesting context (see
     *                 RequestContext): retention-governing stores consume
     *                 the fields they understand and ignore the rest.
     *                 Write-through demotions of an already-concluded window
     *                 admit with a default context (no TTL fact).
     *
     * @return Expected the write-end token on success: keys echoes blockIds,
     *         payload packs the write-end addressing consumed by the
     *         matching Dump job — empty when the worker side resolves
     *         blockIds on its own. Error status (e.g. NoSpace) otherwise.
     *
     * @note Offload reservations are NOT visible to Lookup until the
     *       matching Commit. Implementations must guarantee atomicity: if
     *       Alloc fails, none of the listed blocks remain reserved.
     *       Every successful Alloc must be concluded by exactly one
     *       Commit of the same blocks (true admits the data, false
     *       releases the reservation); until then the reserved slots
     *       are immune to eviction. Concurrent Allocs of the same block
     *       may share one reservation: each holds one reference and is
     *       concluded by its own Commit of the same blocks, and the
     *       reservation concludes -- becoming visible or released --
     *       when the last reference arrives.
     */
    virtual Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                           const RequestContext& context) = 0;
    /**
     * @brief Finalizes or releases the reservation.
     *
     * @param blockIds Blocks of the admission being concluded.
     * @param success  true commits the blocks behind the admission and
     *                 makes them visible to Lookup; false releases the
     *                 reservation.
     *
     * @note Commit addresses this store's live admission of exactly these
     *       blocks. Concurrent admissions of the same block (different
     *       requests) are serialized by the store; on data conflicts
     *       last-writer-wins, matching the window protocol's re-put rule.
     *       Commit(blockIds, false) may only be issued once the matching
     *       Dump jobs have finished and their completions have been consumed
     *       via GetFinished()/Wait(): a released reservation's slots may be
     *       handed to a new admission immediately, and a still-flying Dump
     *       would corrupt the new occupant's data.
     */
    virtual void Commit(const std::vector<BlockId>& blockIds, bool success) = 0;
    /**
     * @brief Token-bearing reload job: copies blocks from this Store to
     *        the inference side, as an asynchronous job.
     *
     * @param jobId Caller-allocated handle of the asynchronous job; must be
     *              unique among in-flight jobs.
     * @param job   Transfer job descriptor: job.blockIds[i] with the token
     *              resolves to the read source of blockIds[i] on this
     *              store's worker side; job.payloads[i] is the destination.
     * @param token Read end minted by the matching Pin; its payload is
     *              empty when this store resolves blockIds on its own
     *              (such implementations typically delegate to the base
     *              Store::Load).
     *
     * @return Status indicating whether the job was accepted (and started);
     *         error status otherwise, with nothing started. Completion
     *         arrives via the base Store's GetFinished()/Wait().
     */
    virtual Status Load(std::size_t jobId, JobDescriptor job, AddressedToken token) = 0;
    /**
     * @brief Token-bearing offload job: copies blocks from the inference
     *        side into this Store, as an asynchronous job.
     *
     * @param jobId  Caller-allocated handle of the asynchronous job; must be
     *               unique among in-flight jobs.
     * @param job    Transfer job descriptor: job.blockIds[i] with the token
     *               resolves to the write destination of blockIds[i] on
     *               this store's worker side; job.payloads[i] is the
     *               source. The data becomes visible to Lookup only after
     *               the matching Commit.
     * @param token  Write end minted by the matching Alloc(...); its
     *               payload is empty when this store resolves blockIds on
     *               its own.
     *
     * @return Status indicating whether the job was accepted (and started);
     *         error status otherwise, with nothing started. Completion
     *         arrives via the base Store's GetFinished()/Wait().
     */
    virtual Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken token) = 0;
};

}  // namespace UC::Store
