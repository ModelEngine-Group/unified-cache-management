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
#include "status/status.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief Cascaded offload facet of a caching Store tier: copies committed
 *        blocks down into the wired lower tier, so they survive this
 *        tier's eviction and expiry.
 *
 * @note Trigger: the conclusion of a successful admission -- the Commit
 *       that arrives when the reservation's last reference lands and some
 *       sharer concluded with success -- never the Dump completion, since
 *       several workers may dump into the same slot and only the concluded
 *       Commit establishes that the block's data is complete. Every
 *       successfully concluded admission therefore enters the offload
 *       pending set exactly once.
 * @note Split halves: the engine half (PrepareOffload/CompleteOffload)
 *       is served by the scheduler-role instance that concludes
 *       admissions; the executor half (Offload) by the worker-role
 *       instances, each executing plans against its own lower-tier
 *       instance. Plans travel between the halves over the deployment's
 *       channel of choice -- the facet fixes the plan's encoding,
 *       execution, and receipt, not the transport. An instance serving
 *       both roles can close the loop internally.
 * @note Scenario precondition: the deployment never re-admits a committed
 *       block -- a key whose admission has concluded is never Alloc'd
 *       again while it remains resident (the window protocol does not
 *       re-put a concluded window). The facet relies on this for its
 *       exactly-once guarantee and for torn-read freedom: a source slot
 *       pinned by an in-flight plan must not be rewritten mid-copy.
 *       Implementations are encouraged to tripwire the violation -- fail
 *       the re-admission Alloc of a key still pinned by an in-flight plan,
 *       a protocol-legal refusal -- rather than silently copy torn data.
 * @note Threading contract: control-plane methods are called from a
 *       single thread or are externally serialized. The engine half
 *       runs on the scheduler's driving thread; the executor half is
 *       called from the worker's submit path.
 */
class Offloadable {
public:
    virtual ~Offloadable() = default;

    /*== Engine half ==*/
    /**
     * @brief Mint the next offload plan: pin the source slots of every
     *        block whose admission concluded successfully since the last
     *        plan, and encode them into one self-addressed plan token.
     *
     * The pins defer eviction and expiry of the sources while the plan
     * is in flight: the lower-tier copies are not committed yet, so the
     * sources must stay resident until the plan's receipt.
     *
     * @return The plan token, or an empty token (empty keys) when nothing
     *         is pending. The payload is the self-addressed envelope
     *         consumed verbatim by the executor half -- conventionally
     *         the plan's batch id plus the addressed source slots of
     *         plan.keys.
     *
     * @note Exactly-once per key: a key enters at most one plan over the
     *       Store's lifetime; every minted plan must be concluded by
     *       exactly one receipt.
     * @note Cadence: the caller drains this at its own pace (e.g. once per
     *       scheduling turn); a plan covers whatever has concluded, and an
     *       empty token simply means idle.
     */
    virtual AddressedToken PrepareOffload() = 0;
    /**
     * @brief Receipt of a plan executed to conclusion: release the source
     *        pins minted by the matching PrepareOffload, regardless of the
     *        plan's outcome.
     *
     * The offload outcome is not reported here: it already reached the
     * caller as the executing worker's (jobId, success) completion
     * event, and the pin release has no use for the verdict anyway --
     * a failed offload leaves the source committed in this tier
     * either way.
     *
     * @param plan The concluded plan, as minted by PrepareOffload.
     *
     * @note The receipt is the pin-release license: it must not be issued
     *       before the plan's execution has fully concluded -- pins keep
     *       the sources alive until every lower-tier Commit of the plan
     *       has landed.
     * @note Idempotent: receiving an already-received or unknown plan is
     *       a no-op.
     */
    virtual void CompleteOffload(const AddressedToken& plan) = 0;

    /*== Executor half ==*/
    /**
     * @brief Worker-side job: copy the plan's blocks down into the lower
     *        tier, as an asynchronous job.
     *
     * Split out of the Store base: plain Stores never receive it.
     * Consumes the plan's self-addressed payload verbatim and drives the
     * lower tier's admission protocol end to end.
     *
     * @param jobId Caller-allocated handle of the asynchronous job; must
     *              be unique among in-flight jobs.
     * @param plan  Plan minted by the engine half of an identically
     *              configured Store instance.
     *
     * @return Status indicating acceptance only; transfer failures are
     *         reported via GetFinished()/Wait().
     *
     * @note Completion discipline: exactly one (jobId, success) event
     *       surfaces through this Store's stream -- true iff every
     *       sub-batch concluded with a successful lower-tier Commit. The
     *       lower tier's own completions are consumed internally and never
     *       leak into this Store's stream.
     * @note Chunking: large plans are split internally into independent
     *       Alloc/Dump sub-batches; a sub-batch that fails to reserve
     *       (e.g. NoSpace) is dropped and counted while the remaining
     *       sub-batches proceed. The job completes only after every
     *       sub-batch has concluded in the lower tier.
     */
    virtual Status Offload(std::size_t jobId, AddressedToken plan) = 0;
};

}  // namespace UC::Store
