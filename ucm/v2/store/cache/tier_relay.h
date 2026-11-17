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
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include "cache_config.h"
#include "shm_layout.h"
#include "status/status.h"
#include "store.h"
#include "type/block.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The executor half of the Offloadable facet: drives offload plans
 *        into the wired lower tier and folds the lower tier's completion
 *        stream back into the outer jobs of this tier.
 *
 * Offload() splits each submitted plan into sub-batches of at most the
 * configured tier batch size and drives every sub-batch through the lower
 * tier's admission protocol: Alloc reserves, Dump copies the planned
 * blocks out of their pinned shared-pool slots, and the arrival of the
 * sub-batch's completion triggers its Commit -- true admits the copy,
 * false releases the reservation. A sub-batch that fails to reserve or to
 * submit is dropped and counted while the rest proceed; the outer job
 * concludes only after every sub-batch has concluded, its verdict true
 * iff none was lost, and that verdict surfaces exactly once through
 * GetFinished() or Wait(). The lower tier's own completions never leak:
 * the outward stream carries outer jobs only. A job stays in flight until
 * its last pending sub-batch concludes, so a pending job always has a
 * sub-batch to wait on.
 *
 * Driving is lazy and on the calling thread: every pump step happens
 * inside GetFinished() and Wait(), so conclusions land at the caller's
 * drain cadence -- no thread is spawned.
 *
 * @note Sole submitter: the pipe assembler wires the lower tier to this
 *       tier alone, so the sub-batch ids minted here are collision-free
 *       and this object is the lower tier's single completion consumer.
 *       A future prefetch direction shares this object's pump and id
 *       space for the same reason.
 * @note Threading contract: single-threaded like the facet it backs --
 *       control-plane methods are called from one thread or are
 *       externally serialized.
 * @note Destruction contract: the caller must Wait() for every in-flight
 *       job before destruction; behavior with in-flight jobs is
 *       undefined. No defensive cleanup exists: a reservation can be
 *       released only after its Dump has finished, and waiting that out
 *       would block the destructor.
 * @note Lifetime: the lower tier and the pool mapping must outlive this
 *       object -- the pipe's declaration order guarantees the former; the
 *       host guarantees the latter by declaring this component after the
 *       mapping holders.
 */
class TierRelay {
public:
    /**
     * @brief Creates a relay over the wired lower tier and the mapped
     *        shared pool of the offload sources.
     *
     * @param config The domain configuration (lower-tier wiring, block
     *               geometry, sub-batch size).
     * @param pool   The shared pool mapping the plans address; must stay
     *               mapped for the component's whole lifetime.
     *
     * @return Expected the relay on success; InvalidParam when no lower
     *         tier is wired or when the block size or the tier batch
     *         size is zero, with nothing left behind.
     */
    [[nodiscard]] static Expected<std::unique_ptr<TierRelay>> Create(const CacheConfig& config,
                                                                     const ShmMapping& pool);

    /**
     * @brief Submits one offload plan as an asynchronous job: splits it
     *        into sub-batches and drives each through the lower tier's
     *        admission protocol, backing the executor half of the
     *        Offloadable facet.
     *
     * A sub-batch that fails to reserve or to submit is dropped and
     * counted while the remaining sub-batches proceed; a job whose every
     * sub-batch was dropped is already concluded when this method
     * returns -- an empty plan likewise concludes immediately, its
     * verdict true.
     *
     * @param jobId Handle of the asynchronous job; must be unique among
     *              this relay's in-flight and unconsumed jobs.
     * @param plan  Plan token of the engine half of an identically
     *              configured instance; consumed verbatim.
     *
     * @return Status indicating acceptance only. InvalidParam when the
     *         plan is malformed (no receipt handle, or a slot count
     *         disagreeing with the keys) or the handle collides, with
     *         nothing started.
     */
    Status Offload(std::size_t jobId, AddressedToken plan);

    /**
     * @brief Returns the outer jobs concluded since the last call and
     *        consumes their verdicts.
     *
     * Concludes every lower-tier completion that has arrived so far
     * first, so a verdict lands as soon as its job's last sub-batch has.
     *
     * @return (jobId, success) pairs, where success tells whether every
     *         sub-batch of the job concluded with a successful
     *         lower-tier Commit. Each verdict is reported exactly once
     *         across GetFinished() and Wait().
     */
    std::vector<std::pair<std::size_t, bool>> GetFinished();

    /**
     * @brief Blocks until the given job has concluded, then consumes and
     *        reports its verdict.
     *
     * @param jobId Job to wait for; must identify an in-flight job of
     *              this relay or one already concluded but not yet
     *              consumed.
     *
     * @return Status::Ok() when every sub-batch of the job concluded
     *         with a successful lower-tier Commit; a General carrying
     *         the loss count otherwise; InvalidParam for an unknown or
     *         already-consumed id.
     */
    Status Wait(std::size_t jobId);

    /**
     * @brief Tells whether the id identifies a job of this relay: one
     *        still in flight, or one concluded but not yet consumed.
     *
     * A consumed verdict no longer counts, so its id is free for a new
     * submission.
     */
    bool Owns(std::size_t jobId) const;

private:
    /** One in-flight sub-batch: the outer job it serves and the keys of
     *  the reservation its conclusion commits. */
    struct SubBatch {
        std::size_t outerId;       /* Outer job the sub-batch belongs to. */
        std::vector<BlockId> keys; /* Keys of the sub-batch's reservation. */
    };

    /** One outer job's ledger. */
    struct OuterJob {
        std::size_t pending{0}; /* Sub-batches still in flight in the lower tier. */
        std::size_t total{0};   /* Sub-batches the plan split into. */
        std::size_t lost{0};    /* Sub-batches dropped or concluded lost. */
    };

    /** An outer job's concluded verdict, queued until consumed. */
    struct ConcludedJob {
        std::size_t id;    /* The submitted job's id. */
        bool success;      /* True iff no sub-batch was lost. */
        std::size_t lost;  /* Sub-batches dropped or concluded lost. */
        std::size_t total; /* Sub-batches the plan split into. */
    };

    TierRelay(Store* lower, void* base, std::size_t blockSize, std::size_t tierBatchSize) noexcept;

    /*== Driving ==*/

    /** Concludes one lower-tier completion: commits the sub-batch's
     *  reservation and folds its verdict into the outer job, concluding
     *  the job when its last pending sub-batch lands. */
    void Conclude(std::size_t subId, bool ok);

    /** Moves a fully concluded outer job into the unconsumed queue. */
    void ConcludeJob(std::size_t outerId);

    /** Takes the job's unconsumed verdict out of the queue; none when it
     *  is not queued. */
    std::optional<ConcludedJob> TakeConcluded(std::size_t jobId);

    /** First in-flight sub-batch of the job, 0 when none is in flight. */
    std::size_t FirstSubOf(std::size_t jobId) const;

    /** Concludes every completion the lower tier reports so far. */
    void DrainLower();

    /** Ok when the job lost no sub-batch; a General carrying the loss
     *  count otherwise. */
    Status VerdictOf(const ConcludedJob& job) const;

    /*== Wiring and geometry ==*/

    Store* lower_;          /* Non-owning; outlives this relay via the pipe's declaration order. */
    void* base_;            /* Mapped shared-pool base of the offload sources. */
    std::size_t blockSize_; /* Byte length of one pool slot. */
    std::size_t tierBatchSize_; /* Blocks per lower-tier transfer sub-batch. */

    /*== Job bookkeeping ==*/

    std::unordered_map<std::size_t, SubBatch> subs_;   /* In-flight sub-batches by their id. */
    std::unordered_map<std::size_t, OuterJob> outers_; /* In-flight outer jobs by their id. */
    std::deque<ConcludedJob> concluded_; /* Concluded jobs awaiting consumption, oldest first. */
    std::size_t nextSubJob_{1};          /* Id of the next sub-batch, minted from 1. */
};

}  // namespace UC::Store
