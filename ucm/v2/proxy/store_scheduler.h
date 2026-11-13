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

#include <utility>
#include <vector>
#include "status/status.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Proxy {

/**
 * @brief Control-plane facet of one assembled store topology: the
 *        request-scoped queries, the transfer windows, and the offload
 *        drain loop.
 *
 * @note Window protocol, per request and per kind (Get/Put/prefetch,
 *       independently): Get and Put windows are multi-open -- PrepareGet()
 *       and PreparePut() may open any number of concurrent windows per
 *       request -- while a failed PrepareX() opens nothing and may be
 *       retried. Each open window is identified by the AddressedToken its
 *       PrepareX() minted: CompleteGet()/CompletePut() conclude exactly
 *       the window their token addresses and leave the others open, and a
 *       token matching no open window (already concluded, foreign, or
 *       belonging to a finished request) is a silent no-op that never
 *       reaches the store. Prefetch windows stay single-open: a second
 *       PreparePrefetch() while one is open is rejected (InvalidParam).
 *       SubmitX() (StoreWorker) may be called any number of times against
 *       each open window's token. CompleteX() must be issued only after
 *       the completions of the jobs submitted against its window have
 *       been consumed via GetFinished()/Wait(): concluding earlier is
 *       undefined, since releasing a pin or reservation while its
 *       transfer job still flies can corrupt the slot's next occupant
 *       (see Store::Commit and Pin).
 * @note OnRequestFinished() requires every window concluded and every
 *       in-flight job drained (the Store destruction contract applies);
 *       windows left open are concluded best-effort as a safety net
 *       (Unpin / Commit(false) / CompletePrefetch(false)), each open
 *       Get/Put window individually.
 * @note Offload drain loop, independent of requests: PrepareOffload()
 *       drains the plans the topology's own concluded admissions have
 *       queued; SubmitOffload() (StoreWorker) executes a drained plan as
 *       a job; CompleteOffload() concludes it once the job's completion
 *       has been consumed. The loop belongs to the topology, not to any
 *       request: it opens no window and OnRequestFinished() never
 *       concludes it.
 */
class StoreScheduler {
public:
    virtual ~StoreScheduler() = default;

    /** Registers a request and its facts; rejects a duplicate live id. */
    virtual Status OnNewRequest(const RequestContext& request) = 0;
    /**
     * @brief Erases the request and concludes windows left open
     *        best-effort (see the window protocol note above).
     */
    virtual void OnRequestFinished(const RequestId& requestId) = 0;

    virtual Expected<std::pair<std::vector<std::uint8_t>, bool>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& keys) = 0;
    virtual Expected<std::pair<std::int64_t, bool>> LookupOnPrefix(
        const RequestId& requestId, const std::vector<BlockId>& keys) = 0;
    virtual Expected<std::pair<std::int64_t, bool>> LookupOnReverse(
        const RequestId& requestId, const std::vector<BlockId>& keys) = 0;

    virtual void Touch(const RequestId& requestId, const std::vector<BlockId>& keys) = 0;

    virtual Expected<AddressedToken> PreparePrefetch(const RequestId& requestId) = 0;
    virtual void CompletePrefetch(const RequestId& requestId, bool success) = 0;

    virtual Expected<AddressedToken> PrepareGet(const RequestId& requestId,
                                                const std::vector<BlockId>& keys) = 0;
    virtual void CompleteGet(const RequestId& requestId, const AddressedToken& token) = 0;

    virtual Expected<AddressedToken> PreparePut(const RequestId& requestId,
                                                const std::vector<BlockId>& keys) = 0;
    virtual void CompletePut(const RequestId& requestId, const AddressedToken& token,
                             bool success) = 0;

    virtual std::vector<AddressedToken> PrepareOffload() = 0;
    virtual void CompleteOffload(const AddressedToken& plan) = 0;
};

}  // namespace UC::Proxy
