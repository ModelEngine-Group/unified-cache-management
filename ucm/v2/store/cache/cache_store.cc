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
#include "cache_store.h"
#include "cache_config.h"
#include "cache_store_primary.h"
#include "cache_store_replica.h"
#include "cache_store_scheduler.h"
#include "cache_store_worker.h"

namespace UC::Store {

Status CacheStore::Setup(const Dictionary& config)
{
    auto parsed = CacheConfig::From(config);
    if (!parsed.HasValue()) { return parsed.Error(); }
    const CacheConfig& cfg = parsed.Value();

    /* The scheduler never contests: a worker-led domain serves it, and the
     * join handshake decides membership -- not the lock. */
    if (cfg.llmRole == "scheduler") {
        auto scheduler = CacheStoreScheduler::Create(cfg);
        if (!scheduler.HasValue()) { return scheduler.Error(); }
        impl_ = std::move(scheduler.Value());
        return Status::Ok();
    }

    auto ticket = LeadershipTicket::Contest(cfg);
    if (!ticket.HasValue()) { return ticket.Error(); }

    /* The worker branches on the ticket itself: the winner leads the
     * domain, every loser attaches to it. */
    if (cfg.llmRole == "worker") {
        auto worker = CacheStoreWorker::Create(cfg, std::move(ticket.Value()));
        if (!worker.HasValue()) { return worker.Error(); }
        impl_ = std::move(worker.Value());
        return Status::Ok();
    }

    if (ticket.Value().Won()) {
        auto primary = CacheStorePrimary::Create(cfg, std::move(ticket.Value()));
        if (!primary.HasValue()) { return primary.Error(); }
        impl_ = std::move(primary.Value());
        return Status::Ok();
    }
    auto replica = CacheStoreReplica::Create(cfg);
    if (!replica.HasValue()) { return replica.Error(); }
    impl_ = std::move(replica.Value());
    return Status::Ok();
}

Expected<std::vector<std::uint8_t>> CacheStore::Lookup(const std::vector<BlockId>& blockIds)
{
    return impl_->Lookup(blockIds);
}

void CacheStore::Touch(const std::vector<BlockId>& blockIds, const RequestContext& context)
{
    impl_->Touch(blockIds, context);
}

Status CacheStore::Load(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return impl_->Load(jobId, std::move(job), std::move(token));
}

Status CacheStore::Dump(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return impl_->Dump(jobId, std::move(job), std::move(token));
}

std::vector<std::pair<std::size_t, bool>> CacheStore::GetFinished() { return impl_->GetFinished(); }

Status CacheStore::Wait(std::size_t jobId) { return impl_->Wait(jobId); }

Expected<AddressedToken> CacheStore::Pin(const std::vector<BlockId>& blockIds)
{
    return impl_->Pin(blockIds);
}

void CacheStore::Unpin(const std::vector<BlockId>& blockIds) { impl_->Unpin(blockIds); }

Expected<AddressedToken> CacheStore::Alloc(const std::vector<BlockId>& blockIds,
                                           const RequestContext& context)
{
    return impl_->Alloc(blockIds, context);
}

void CacheStore::Commit(const std::vector<BlockId>& blockIds, bool success)
{
    impl_->Commit(blockIds, success);
}

}  // namespace UC::Store
