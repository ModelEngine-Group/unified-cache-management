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
#include <optional>
#include <vector>
#include "status/status.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

class Prefetchable {
public:
    virtual ~Prefetchable() = default;
    /**
     * @brief Probes the presence of the given blocks in this Store; a query
     * that contains absent blocks prefetchable from the lower tier is
     * answered with the prefetchable signal instead.
     *
     * Overloads Store::Lookup: the two are distinguished by the requestId
     * parameter and the optional result.
     *
     * @param requestId Correlation id of one logical query. It doubles as the
     *                  join key of the prefetch protocol: the query issued
     *                  here is the one PreparePrefetch(requestId) concludes.
     * @param blockIds  Content-addressed ids of the blocks to probe, ordered by
     *                  their position within the request.
     *
     * @return Expected optional per-block hit vector: nullopt as a
     *         request-level prefetchable signal — the query contains absent
     *         blocks this Store can prefetch from the lower tier (see
     *         PreparePrefetch); otherwise the per-block hit vector, where
     *         hits[i] is non-zero iff blockIds[i] is present, and
     *         hits.size() == blockIds.size(). Error status otherwise.
     *
     * @note The two result channels are distinct: nullopt is the prefetchable
     *       signal, not an error; an Error status means the query itself
     *       failed.
     * @note Resolution protocol: upon nullopt the caller drives
     *       PreparePrefetch/Prefetch/CompletePrefetch, then retries Lookup
     *       with the same requestId; committed keys become HIT (see
     *       CompletePrefetch) and the retry yields the final hit vector.
     * @note requestId lifecycle: a requestId identifies exactly one logical
     *       query and must not be reused for a different block set. The
     *       request-scoped state lives from the first Lookup through the
     *       prefetch protocol; once a final hit vector or error has been
     *       returned, the request is concluded and implementations may
     *       reclaim its state.
     */
    virtual Expected<std::optional<std::vector<std::uint8_t>>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& blockIds) = 0;
    /**
     * @brief Prefix-oriented variant of Lookup: reports how far the request
     * is present from its head.
     *
     * @param requestId See Lookup(); the prefetchable signal is
     *                  inherited: the same requestId is passed through and
     *                  nullopt is propagated when the underlying query is
     *                  answered as prefetchable.
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected optional index of the last block of the leading run
     *         of present blocks: nullopt as the request-level prefetchable
     *         signal (see Lookup); -1 when the first block is absent
     *         (empty input yields -1 as well); blockIds.size() - 1 when
     *         every block is present. Error status otherwise.
     *
     * @note The default implementation derives the answer from a single
     *       Lookup call; override to exploit ordered medium layouts, e.g.
     *       by answering from metadata without probing each block.
     */
    virtual Expected<std::optional<std::int64_t>> LookupOnPrefix(
        const RequestId& requestId, const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(requestId, blockIds);
        if (!res) { return res.Error(); }
        if (!res.Value().has_value()) { return std::nullopt; }
        const auto& hits = res.Value().value();
        for (std::size_t i = 0; i < hits.size(); i++) {
            if (hits[i] == 0) { return static_cast<std::int64_t>(i) - 1; }
        }
        return static_cast<std::int64_t>(hits.size()) - 1;
    }
    /**
     * @brief Reverse variant of Lookup: reports the rightmost present
     * block of the request.
     *
     * @param requestId See Lookup(); the prefetchable signal is
     *                  inherited: the same requestId is passed through and
     *                  nullopt is propagated when the underlying query is
     *                  answered as prefetchable.
     * @param blockIds  Blocks ordered by their position within the request.
     *
     * @return Expected optional index of the rightmost present block:
     *         nullopt as the request-level prefetchable signal (see
     *         Lookup); -1 when none is present (empty input yields -1
     *         as well). Error status otherwise.
     *         Contiguity is neither required nor checked: blocks before the
     *         returned index may be absent.
     *
     * @note The default implementation derives the answer from a single
     *       Lookup call; override only if the Store can answer reverse
     *       queries cheaper.
     */
    virtual Expected<std::optional<std::int64_t>> LookupOnReverse(
        const RequestId& requestId, const std::vector<BlockId>& blockIds)
    {
        auto res = Lookup(requestId, blockIds);
        if (!res) { return res.Error(); }
        if (!res.Value().has_value()) { return std::nullopt; }
        const auto& hits = res.Value().value();
        for (auto i = hits.size(); i > 0; i--) {
            if (hits[i - 1] != 0) { return static_cast<std::int64_t>(i) - 1; }
        }
        return -1;
    }
    /**
     * @brief Pin the sources and reserve the destinations for the
     *        prefetchable subset of keys — the subset whose absence made
     *        the matching Prefetchable::Lookup answer nullopt.
     *
     * @param requestId Correlation id of the query to prefetch.
     * @param context   Request facts for the destination admission: the
     *                  internal Store::Alloc admits plan.keys with this
     *                  context, e.g. carrying the current window's request
     *                  TTL so prefetched blocks inherit it (see
     *                  RequestContext::ttl).
     *
     * @return The prefetch plan, or an error Status.
     */
    virtual Expected<AddressedToken> PreparePrefetch(const RequestId& requestId,
                                                     const RequestContext& context) = 0;
    /**
     * @brief Conclude the plan opened by PreparePrefetch.
     *
     * On success=true the destination admission is committed (Commit of
     * plan.keys) and the keys become HIT; on success=false the admission
     * is rolled back.
     */
    virtual Status CompletePrefetch(const AddressedToken& plan, bool success) = 0;
    /**
     * @brief Worker-side job: copy the plan's sources into this store's
     *        destinations, as an asynchronous job.
     *
     * Split out of the Store base: plain Stores never receive it. Consumes
     * the plan's self-addressed payload verbatim: implementations unpack
     * what they packed in PreparePrefetch (conventionally the source metas
     * minted by the lower tier's Pin and their own destination metas) and
     * drive the lower tier's Load with the resolved destinations as
     * payloads, passing the lower tier's read-end token along — keys-only
     * when the lower tier uses default addressing.
     *
     * @return Status indicating acceptance only; transfer failures are
     *         reported via GetFinished()/Wait(), like every other job.
     */
    virtual Status Prefetch(std::size_t jobId, AddressedToken plan) = 0;
};

}  // namespace UC::Store
