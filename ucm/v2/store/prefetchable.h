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
#include <utility>
#include <vector>
#include "status/status.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief Store facet that fetches its absent blocks from the tier below.
 *
 * @note The requestId is the protocol's join key. The request-scoped
 *       state it addresses is created by the first Lookup and
 *       reclaimed by ReleaseRequest.
 */
class Prefetchable {
public:
    virtual ~Prefetchable() = default;
    /**
     * @brief Probes the presence of the given blocks in this Store, and
     * signals whether the absent ones can be fetched from the lower tier.
     *
     * Overloads Store::Lookup: the two are distinguished by the requestId
     * parameter and the pair result.
     *
     * @param requestId Correlation id of one logical query.
     * @param blockIds  Blocks to probe.
     *
     * @return Expected {hits, prefetchable} pair on success: hits[i] is
     *         non-zero iff blockIds[i] is present (hits.size() ==
     *         blockIds.size(), populated even when the signal is set);
     *         prefetchable signals that the query contains absent blocks
     *         this Store can prefetch from the lower tier. An error
     *         status means the probe failed.
     *
     * @note Finality: prefetchable == false makes hits the final answer;
     *       prefetchable == true is advisory — committed prefetches
     *       change the snapshot, and a retry with the same requestId
     *       returns the authoritative final hit vector.
     */
    virtual Expected<std::pair<std::vector<std::uint8_t>, bool>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& blockIds) = 0;
    /**
     * @brief Prefix-oriented variant of Lookup: reports how far the
     * request is present from its head.
     *
     * @param requestId See Lookup().
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected {index, prefetchable} pair: index of the last
     *         block of the leading run of present blocks, -1 when the
     *         leading run is empty; prefetchable as in Lookup. Error
     *         status otherwise.
     */
    virtual Expected<std::pair<std::int64_t, bool>> LookupOnPrefix(
        const RequestId& requestId, const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(requestId, blockIds);
        if (!res) { return res.Error(); }
        const auto& [hits, prefetchable] = res.Value();
        for (std::size_t i = 0; i < hits.size(); i++) {
            if (hits[i] == 0) {
                return std::make_pair(static_cast<std::int64_t>(i) - 1, prefetchable);
            }
        }
        return std::make_pair(static_cast<std::int64_t>(hits.size()) - 1, prefetchable);
    }
    /**
     * @brief Reverse variant of Lookup: reports the rightmost present
     * block of the request.
     *
     * @param requestId See Lookup().
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected {index, prefetchable} pair: index of the rightmost
     *         present block, -1 when none is present; blocks before it
     *         may be absent. prefetchable as in Lookup. Error status
     *         otherwise.
     */
    virtual Expected<std::pair<std::int64_t, bool>> LookupOnReverse(
        const RequestId& requestId, const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(requestId, blockIds);
        if (!res) { return res.Error(); }
        const auto& [hits, prefetchable] = res.Value();
        for (auto i = hits.size(); i > 0; i--) {
            if (hits[i - 1] != 0) {
                return std::make_pair(static_cast<std::int64_t>(i) - 1, prefetchable);
            }
        }
        return std::make_pair(-1, prefetchable);
    }
    /**
     * @brief Pin the sources and reserve the destinations for the given
     *        blocks of the request.
     *
     * @param keys    Absent blocks to admit: a key outside the request's
     *                probed set is a protocol violation (InvalidParam), a
     *                key already present is filtered out, an empty list
     *                yields an empty plan.
     * @param context Request facts for the destination admission: the
     *                internal Store::Alloc admits plan.keys with this
     *                context.
     *
     * @return The prefetch plan; token.keys echo the admitted subset.
     *
     * @note One prefetch window per request: a second call while one is
     *       open is rejected (InvalidParam), a concluded window frees
     *       the request for the next, and a failed call opens no window.
     */
    virtual Expected<AddressedToken> PreparePrefetch(const std::vector<BlockId>& keys,
                                                     const RequestContext& context) = 0;
    /**
     * @brief Conclude the plan opened by PreparePrefetch.
     *
     * On success=true the destination admission is committed (Commit of
     * plan.keys) and the keys become HIT; on success=false the admission
     * is rolled back.
     *
     * @param success Transfer verdict of the jobs that executed the plan,
     *                consumed from their completion events.
     *
     * @note Idempotent: concluding an already-concluded or unknown plan
     *       is a no-op.
     */
    virtual void CompletePrefetch(const AddressedToken& plan, bool success) = 0;
    /**
     * @brief Worker-side job: copy the plan's sources into this store's
     *        destinations.
     *
     * The payload is the envelope this store's PreparePrefetch minted:
     * conventionally the source metas minted by the lower tier's Pin
     * plus this store's own destination metas. The job drives the lower
     * tier's Load with the resolved destinations as payloads, passing
     * the lower tier's read-end token along — keys-only when the lower
     * tier uses default addressing.
     *
     * @return Status indicating acceptance only; transfer failures are
     *         reported via GetFinished()/Wait().
     */
    virtual Status Prefetch(std::size_t jobId, AddressedToken plan) = 0;
    /**
     * @brief Releases the request-scoped state of a finished request.
     *
     * @param requestId The finished request.
     *
     * @note Releasing an unknown request is a no-op.
     */
    virtual void ReleaseRequest(const RequestId& requestId) = 0;
};

}  // namespace UC::Store
