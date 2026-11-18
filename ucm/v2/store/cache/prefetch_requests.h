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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "control_codec.h"
#include "status/status.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The carrying channel to this tier's engine: one control-plane
 *        request in, its response out.
 *
 * The implementer stamps the request's sequence number and owns the
 * transport. The request is moved from; nothing is read back.
 */
class ControlEndpoint {
public:
    virtual ~ControlEndpoint() = default;

    /**
     * @return Expected the engine's response on success; the carrying
     *         channel's error Status otherwise.
     */
    virtual Expected<ControlResponse> Execute(ControlRequest request) = 0;
};

/**
 * @brief The request glue of the Prefetchable facet: holds the
 *        request-scoped bookkeeping around the engine's window half
 *        and the lower tier's presence probe.
 *
 * The control-plane wire carries no requestId, so the per-request
 * state lives here, beside its caller. Each record owns the keys its
 * request's lookups have probed -- the admission scope of the
 * request's prefetches -- and its one open window, held by the
 * handle the window's plan carries. A record is created by the
 * request's first successful Lookup and dropped by ReleaseRequest.
 *
 * Lookup answers from two sources: the engine holds this store's
 * presence snapshot, and the wired lower tier is probed for the
 * query's absent blocks -- any presence there sets the prefetchable
 * signal. PreparePrefetch guards its keys against the request's
 * probed set and the one-window rule before the engine mints.
 * CompletePrefetch locates the open window by its plan's handle and
 * concludes it on the engine, freeing the request for the next
 * window. ReleaseRequest concludes a still-open window as failed
 * before dropping the record, so the engine's reserved destinations
 * are released.
 *
 * @note Threading contract: single-threaded like the facet it backs
 *       -- the methods are called from one thread or are externally
 *       serialized.
 * @note Lifetime: the endpoint and the lower tier must outlive this
 *       object.
 */
class PrefetchRequests {
public:
    /**
     * @param engine The engine's carrying channel.
     * @param lower  The lower tier to probe, null when none is wired.
     */
    PrefetchRequests(ControlEndpoint& engine, Store* lower);

    PrefetchRequests(PrefetchRequests&&) = delete;
    PrefetchRequests& operator=(PrefetchRequests&&) = delete;
    PrefetchRequests(const PrefetchRequests&) = delete;
    PrefetchRequests& operator=(const PrefetchRequests&) = delete;

    /**
     * @brief Probes this store's presence for the query and signals
     *        whether the lower tier can serve its absent blocks.
     *
     * The absent blocks are probed against the wired lower tier
     * unless nothing is absent or no lower tier is wired; the signal
     * is set when the probe reports any of them present. On success
     * the query's keys join the request's probed set -- a failed
     * exchange or probe changes nothing, and the request's first
     * successful Lookup creates its record.
     *
     * @param requestId Correlation id of one logical query.
     * @param blockIds  Blocks to probe.
     *
     * @return Expected {hits, prefetchable} pair on success: hits
     *         mirrors the engine's presence snapshot, hits[i]
     *         non-zero iff blockIds[i] is present; prefetchable
     *         tells whether the lower tier holds any of the absent
     *         blocks. The failing step's error Status otherwise.
     */
    [[nodiscard]] Expected<std::pair<std::vector<std::uint8_t>, bool>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& blockIds);

    /**
     * @brief Opens the request's prefetch window on the engine.
     *
     * @param requestId The request whose window is opened.
     * @param keys      Blocks to admit; each must sit inside the
     *                  request's probed set.
     * @param context   Request facts for the engine's admission.
     *
     * @return Expected the minted plan token on success -- an idle
     *         token when the engine admits nothing. InvalidParam
     *         when the request has no record, a key falls outside
     *         its probed set, or its window is already open; the
     *         engine's error Status when the admission fails. No
     *         rejection records a window or consumes a handle, and
     *         the idle plan records none either.
     */
    [[nodiscard]] Expected<AddressedToken> PreparePrefetch(const RequestId& requestId,
                                                           const std::vector<BlockId>& keys,
                                                           const RequestContext& context);

    /**
     * @brief Concludes the window identified by the plan's handle:
     *        forwards the verdict to the engine and frees the request
     *        for its next window.
     *
     * An idle plan, or a handle matching no open window -- unknown,
     * already concluded, or released -- concludes nothing.
     *
     * @param plan    The plan token whose window is concluded.
     * @param success Transfer verdict of the jobs that executed the
     *                plan.
     */
    void CompletePrefetch(const AddressedToken& plan, bool success);

    /**
     * @brief Drops the request's record; a window still open is
     *        concluded as failed first.
     *
     * @param requestId The finished request. An unknown request is a
     *                  no-op.
     */
    void ReleaseRequest(const RequestId& requestId);

private:
    /** One request's state: the keys its lookups have probed, and its
     *  open window's handle -- 0 when none. */
    struct RequestRecord {
        std::unordered_set<BlockId> probed;
        std::uint64_t window{0};
    };

    /** Runs one request through the engine endpoint, unwrapping the
     *  channel's error. */
    [[nodiscard]] Expected<ControlResponse> Exchange(ControlRequest request);

    /** Forwards one window conclusion to the engine. */
    void ConcludeWindow(std::uint64_t batchId, bool success);

    ControlEndpoint& engine_; /* The engine's carrying channel; non-owning. */
    Store* lower_;            /* The tier to probe; null when none is wired. */
    std::unordered_map<RequestId, RequestRecord> requests_; /* State by its join key. */
};

}  // namespace UC::Store
