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
#include "data_plane.h"
#include <cstdint>
#include "memory/dma/dma_engine_registry.h"

namespace UC::Store {

Expected<std::unique_ptr<DataPlane>> DataPlane::Create(const CacheConfig& config,
                                                       const ShmMapping& pool)
{
    /* Engine selection: an empty name takes this build's default, a name
     * takes the registered kind it denotes; the registry resolves both in
     * one call. Selection failures and a failing engine Setup surface
     * here -- the host's Setup error, before any transfer exists. */
    std::unique_ptr<Transfer::DmaEngine> engine;
    auto created = Transfer::DmaEngineRegistry::Instance().Create(config.dmaEngine);
    if (!created.HasValue()) { return created.Error(); }
    engine.reset(created.Value());

    /* maxLength is the block size: the engine's own bound on a single
     * segment's size. Slot placement (offset + size <= blockSize) is
     * rejected at submit time; this bound is the engine-side half of the
     * same guarantee, independent of the caller-side checks. */
    Transfer::DmaEngine::InitParam param;
    param.deviceId = config.deviceId;
    param.maxLength = config.blockSize;
    param.hostBase = pool.Base();
    param.hostLength = pool.Size();
    param.nStream = config.nDmaStream;
    if (Status st = engine->Setup(param); st.Failure()) { return st; }

    return std::unique_ptr<DataPlane>{
        new DataPlane{config.blockSize, pool.Base(), std::move(engine)}
    };
}

DataPlane::DataPlane(std::size_t blockSize, void* base,
                     std::unique_ptr<Transfer::DmaEngine>&& engine) noexcept
    : blockSize_{blockSize}, base_{base}, engine_{std::move(engine)}
{
}

Status DataPlane::Submit(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token,
                         bool intoPool)
{
    /* Shape checks first: a mismatch answers InvalidParam with nothing started. */
    if (job.blockIds.empty()) { return Status::InvalidParam("job {} carries no blocks", jobId); }
    if (job.blockIds.size() != job.payloads.size()) {
        return Status::InvalidParam("job {} carries {} block ids and {} payloads", jobId,
                                    job.blockIds.size(), job.payloads.size());
    }
    const std::vector<std::uint32_t> slots = DecodeTokenPayload(token.payload);
    if (slots.size() != job.blockIds.size()) {
        return Status::InvalidParam("job {} transfers {} blocks over a token of {} slots", jobId,
                                    job.blockIds.size(), slots.size());
    }
    for (std::size_t i = 0; i < job.payloads.size(); ++i) {
        const BlockPayload& payload = job.payloads[i];
        if (payload.pointers.empty()) {
            return Status::InvalidParam("job {} block {} carries no segments", jobId, i);
        }
        if (payload.pointers.size() != payload.sizes.size() ||
            payload.pointers.size() != payload.offsets.size()) {
            return Status::InvalidParam(
                "job {} block {} carries {} pointers, {} sizes and {} offsets", jobId, i,
                payload.pointers.size(), payload.sizes.size(), payload.offsets.size());
        }
        /* A segment that starts past or runs past its slot bounds would
         * copy into the neighboring slot -- or past the pool, on the last
         * one; both shapes are rejected here, nothing started. The two
         * comparisons stay on the blockSize_ side so no offset + size
         * sum can overflow. */
        for (std::size_t j = 0; j < payload.sizes.size(); ++j) {
            if (payload.offsets[j] > blockSize_ ||
                payload.sizes[j] > blockSize_ - payload.offsets[j]) {
                return Status::InvalidParam(
                    "job {} block {} segment {} (offset {}, size {}) runs past the block length {}",
                    jobId, i, j, payload.offsets[j], payload.sizes[j], blockSize_);
            }
        }
    }

    /* One job flattens into one columnar batch: segment (j) of block (i)
     * pairs the caller's pointer with slot + offset in the pool, and
     * block (i) lands as group (i) of nSegmentPerGroup for the
     * stream-parallel engines. Segment validity -- non-null addresses,
     * non-zero, maxLength-bounded sizes -- is the engine's contract to
     * enforce. */
    Transfer::DmaEngine::DmaJob batch;
    std::size_t segments = 0;
    for (const BlockPayload& payload : job.payloads) { segments += payload.pointers.size(); }
    batch.sources.reserve(segments);
    batch.destinations.reserve(segments);
    batch.sizes.reserve(segments);
    batch.nSegmentPerGroup.reserve(job.payloads.size());
    batch.prerequisite = job.prerequisiteHandle;

    for (std::size_t i = 0; i < slots.size(); ++i) {
        const BlockPayload& payload = job.payloads[i];
        batch.nSegmentPerGroup.push_back(payload.pointers.size());
        std::byte* const slot = static_cast<std::byte*>(SlotAddress(base_, slots[i], blockSize_));
        for (std::size_t j = 0; j < payload.pointers.size(); ++j) {
            std::byte* const shared = slot + payload.offsets[j];
            if (intoPool) {
                batch.sources.push_back(payload.pointers[j]);
                batch.destinations.push_back(shared);
            } else {
                batch.sources.push_back(shared);
                batch.destinations.push_back(payload.pointers[j]);
            }
            batch.sizes.push_back(payload.sizes[j]);
        }
    }

    return intoPool ? engine_->SubmitD2HBatch(jobId, batch) : engine_->SubmitH2DBatch(jobId, batch);
}

Status DataPlane::Dump(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token)
{
    return Submit(jobId, job, token, true);
}

Status DataPlane::Load(std::size_t jobId, const JobDescriptor& job, const AddressedToken& token)
{
    return Submit(jobId, job, token, false);
}

std::vector<std::pair<std::size_t, bool>> DataPlane::GetFinished()
{
    auto finished = engine_->GetFinished();
    /* An engine that was set up successfully serves this call; the error
     * arm is unreachable by construction and has no Status return to
     * surface through -- an empty drain is the closest honest answer. */
    return finished.HasValue() ? std::move(finished.Value())
                               : std::vector<std::pair<std::size_t, bool>>{};
}

Status DataPlane::Wait(std::size_t jobId) { return engine_->Wait(jobId); }

}  // namespace UC::Store
