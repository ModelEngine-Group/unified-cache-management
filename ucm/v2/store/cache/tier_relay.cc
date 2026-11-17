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
#include "tier_relay.h"
#include <algorithm>
#include <utility>
#include "token_codec.h"

namespace UC::Store {

Expected<std::unique_ptr<TierRelay>> TierRelay::Create(const CacheConfig& config,
                                                       const ShmMapping& pool)
{
    /* Fail-closed wiring and geometry: a relay without a lower tier has
     * nothing to drive, and a zero block size or sub-batch size leaves
     * every plan degenerate. */
    if (config.lowerStore == nullptr) {
        return Status::InvalidParam("tier relay: no lower tier is wired");
    }
    if (config.blockSize == 0) {
        return Status::InvalidParam("tier relay: the block size is zero");
    }
    if (config.tierBatchSize == 0) {
        return Status::InvalidParam("tier relay: the tier batch size is zero");
    }
    return std::unique_ptr<TierRelay>{
        new TierRelay{config.lowerStore, pool.Base(), config.blockSize, config.tierBatchSize}
    };
}

TierRelay::TierRelay(Store* lower, void* base, std::size_t blockSize,
                     std::size_t tierBatchSize) noexcept
    : lower_{lower}, base_{base}, blockSize_{blockSize}, tierBatchSize_{tierBatchSize}
{
}

Status TierRelay::Offload(std::size_t jobId, AddressedToken plan)
{
    /* Shape checks first: a mismatch answers InvalidParam with nothing
     * started, the lower tier untouched. */
    const PlanPayload payload = DecodePlanPayload(plan.payload);
    if (payload.batchId == 0) {
        return Status::InvalidParam("offload job {} carries no receipt handle", jobId);
    }
    if (payload.slots.size() != plan.keys.size()) {
        return Status::InvalidParam("offload job {} addresses {} blocks over {} slots", jobId,
                                    plan.keys.size(), payload.slots.size());
    }
    if (Owns(jobId)) {
        return Status::InvalidParam("offload job {} is in flight or unconsumed", jobId);
    }

    /* The plan splits at the tier batch size; every sub-batch walks the
     * lower tier's admission protocol on its own, admitted with a
     * default context -- the blocks are already concluded on this tier,
     * so no request facts apply. */
    OuterJob& outer = outers_[jobId];
    for (std::size_t begin = 0; begin < plan.keys.size(); begin += tierBatchSize_) {
        const std::size_t count = std::min(tierBatchSize_, plan.keys.size() - begin);
        const std::vector<BlockId> chunk(plan.keys.begin() + begin,
                                         plan.keys.begin() + begin + count);
        outer.total++;

        /* A refused reservation is dropped and counted; nothing was
         * taken, so nothing needs releasing. */
        auto reserved = lower_->Alloc(chunk, RequestContext{});
        if (!reserved.HasValue()) {
            outer.lost++;
            continue;
        }

        /* One whole-slot segment per planned key: the sub-batch's source
         * set, pinned by the plan's engine half. */
        JobDescriptor transfer;
        transfer.blockIds = chunk;
        transfer.payloads.resize(chunk.size());
        for (std::size_t i = 0; i < count; ++i) {
            transfer.payloads[i].pointers.push_back(
                SlotAddress(base_, payload.slots[begin + i], blockSize_));
            transfer.payloads[i].sizes.push_back(blockSize_);
            transfer.payloads[i].offsets.push_back(0);
        }
        const std::size_t subId = nextSubJob_++;
        if (Status submitted =
                lower_->Dump(subId, std::move(transfer), std::move(reserved.Value()));
            submitted.Failure()) {
            /* Nothing is flying: the reservation is released right away,
             * and the sub-batch is lost like a refused one. */
            lower_->Commit(chunk, false);
            outer.lost++;
            continue;
        }
        subs_.emplace(subId, SubBatch{jobId, chunk});
        outer.pending++;
    }

    /* Every sub-batch dropped: the verdict is already final, so it queues
     * here instead of waiting for a completion that will never come. */
    if (outer.pending == 0) { ConcludeJob(jobId); }
    return Status::Ok();
}

std::vector<std::pair<std::size_t, bool>> TierRelay::GetFinished()
{
    DrainLower();
    std::vector<std::pair<std::size_t, bool>> finished;
    finished.reserve(concluded_.size());
    for (const ConcludedJob& job : concluded_) { finished.emplace_back(job.id, job.success); }
    concluded_.clear();
    return finished;
}

Status TierRelay::Wait(std::size_t jobId)
{
    /* A queued verdict needs no pumping at all. */
    if (auto concluded = TakeConcluded(jobId)) { return VerdictOf(*concluded); }
    if (outers_.find(jobId) == outers_.end()) {
        return Status::InvalidParam("offload job {} is not in flight here", jobId);
    }

    /* The pump: settle everything already finished, then block on one
     * pending sub-batch at a time through the lower tier's own wait --
     * each blocked wait concludes its sub-batch, so the job concludes
     * after at most as many rounds as it has sub-batches. */
    while (true) {
        DrainLower();
        if (auto concluded = TakeConcluded(jobId)) { return VerdictOf(*concluded); }
        const std::size_t subId = FirstSubOf(jobId);
        if (subId == 0) { break; }
        Conclude(subId, lower_->Wait(subId).Success());
    }
    /* Unreachable by construction; fail closed anyway. */
    return Status::General("offload job {} lost its in-flight sub-batches", jobId);
}

bool TierRelay::Owns(std::size_t jobId) const
{
    if (outers_.find(jobId) != outers_.end()) { return true; }
    for (const ConcludedJob& job : concluded_) {
        if (job.id == jobId) { return true; }
    }
    return false;
}

void TierRelay::Conclude(std::size_t subId, bool ok)
{
    const auto sub = subs_.find(subId);
    /* The lower tier's completions are this relay's alone, so every
     * drained event matches a live sub-batch; a stranger is ignored
     * rather than trusted with the ledgers. */
    if (sub == subs_.end()) { return; }

    const std::size_t outerId = sub->second.outerId;
    lower_->Commit(sub->second.keys, ok);
    subs_.erase(sub);

    OuterJob& outer = outers_[outerId];
    outer.pending--;
    if (!ok) { outer.lost++; }
    if (outer.pending == 0) { ConcludeJob(outerId); }
}

void TierRelay::ConcludeJob(std::size_t outerId)
{
    const OuterJob job = outers_[outerId];
    concluded_.push_back(ConcludedJob{outerId, job.lost == 0, job.lost, job.total});
    outers_.erase(outerId);
}

std::optional<TierRelay::ConcludedJob> TierRelay::TakeConcluded(std::size_t jobId)
{
    for (auto it = concluded_.begin(); it != concluded_.end(); ++it) {
        if (it->id == jobId) {
            const ConcludedJob job = *it;
            concluded_.erase(it);
            return job;
        }
    }
    return std::nullopt;
}

std::size_t TierRelay::FirstSubOf(std::size_t jobId) const
{
    for (const auto& [id, sub] : subs_) {
        if (sub.outerId == jobId) { return id; }
    }
    return 0;
}

void TierRelay::DrainLower()
{
    for (const auto& finished : lower_->GetFinished()) {
        Conclude(finished.first, finished.second);
    }
}

Status TierRelay::VerdictOf(const ConcludedJob& job) const
{
    if (job.success) { return Status::Ok(); }
    return Status::General("offload job {}: {} of {} sub-batches were lost in the lower tier",
                           job.id, job.lost, job.total);
}

}  // namespace UC::Store
