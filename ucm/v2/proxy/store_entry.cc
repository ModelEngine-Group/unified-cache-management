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
#include "store_entry.h"
#include <algorithm>
#include <utility>

namespace UC::Proxy {

namespace {

Status ValidateTransfer(std::size_t jobId, const std::vector<BlockPayload>& payloads,
                        const AddressedToken& token, bool tokened,
                        const std::unordered_set<std::size_t>& inFlightJobs)
{
    if (inFlightJobs.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    if (payloads.size() != token.keys.size()) {
        return Status::InvalidParam("job '{}' pairs {} payloads with {} keys", jobId,
                                    payloads.size(), token.keys.size());
    }
    if (!tokened && !token.payload.empty()) {
        return Status::InvalidParam("job '{}' carries a token but the store is not token-addressed",
                                    jobId);
    }
    return Status::Ok();
}

}  // namespace

StoreEntry::StoreEntry(std::unique_ptr<LoadedStore> loaded) : loaded_(std::move(loaded))
{
    prefetchable_ = loaded_->Store()->AsPrefetchable();
    tokened_ = loaded_->Store()->AsTokenAddressed();
}

Status StoreEntry::OnNewRequest(const RequestContext& request)
{
    if (requests_.count(request.requestId) != 0) {
        return Status::InvalidParam("request '{}' is already live", request.requestId);
    }
    RequestState state;
    state.request = request;
    requests_.emplace(request.requestId, std::move(state));
    return Status::Ok();
}

void StoreEntry::OnRequestFinished(const RequestId& requestId)
{
    auto it = requests_.find(requestId);
    if (it == requests_.end()) { return; }
    /* Best-effort conclusion of windows the caller left open; the window
     * protocol expects them concluded and the in-flight jobs drained before
     * this call (see StoreScheduler). */
    RequestState& state = it->second;
    const AddressedToken& prefetchToken = state.prefetchToken;
    if (prefetchable_ && (!prefetchToken.keys.empty() || !prefetchToken.payload.empty())) {
        prefetchable_->CompletePrefetch(prefetchToken, false);
    }
    if (tokened_) {
        if (!state.pinnedBlocks.empty()) { tokened_->Unpin(state.pinnedBlocks); }
        if (!state.reservedBlocks.empty()) { tokened_->Commit(state.reservedBlocks, false); }
    }
    requests_.erase(it);
}

Expected<std::optional<std::vector<std::uint8_t>>> StoreEntry::Lookup(
    const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->Lookup(requestId, keys); }
    auto res = loaded_->Store()->Lookup(keys);
    if (!res) { return res.Error(); }
    return std::move(res).Value();
}

Expected<std::optional<std::int64_t>> StoreEntry::LookupOnPrefix(const RequestId& requestId,
                                                                 const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->LookupOnPrefix(requestId, keys); }
    auto res = loaded_->Store()->LookupOnPrefix(keys);
    if (!res) { return res.Error(); }
    return res.Value();
}

Expected<std::optional<std::int64_t>> StoreEntry::LookupOnReverse(const RequestId& requestId,
                                                                  const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->LookupOnReverse(requestId, keys); }
    auto res = loaded_->Store()->LookupOnReverse(keys);
    if (!res) { return res.Error(); }
    return res.Value();
}

void StoreEntry::Touch(const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    loaded_->Store()->Touch(keys, guard.Value()->request);
}

Expected<AddressedToken> StoreEntry::PreparePrefetch(const RequestId& requestId)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    const AddressedToken& prefetchToken = guard.Value()->prefetchToken;
    if (!prefetchToken.keys.empty() || !prefetchToken.payload.empty()) {
        return Status::InvalidParam("request '{}' has an unconcluded prefetch window", requestId);
    }
    if (!prefetchable_) { return AddressedToken{}; }
    auto res = prefetchable_->PreparePrefetch(requestId, guard.Value()->request);
    if (!res) { return res.Error(); }
    guard.Value()->prefetchToken = res.Value();
    return std::move(res).Value();
}

void StoreEntry::CompletePrefetch(const RequestId& requestId, bool success)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    if (prefetchable_) { prefetchable_->CompletePrefetch(guard.Value()->prefetchToken, success); }
    guard.Value()->prefetchToken = AddressedToken{};
}

Expected<AddressedToken> StoreEntry::PrepareGet(const RequestId& requestId,
                                                const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (!guard.Value()->pinnedBlocks.empty()) {
        return Status::InvalidParam("request '{}' has an unconcluded Get window", requestId);
    }
    if (!tokened_) {
        AddressedToken token;
        token.keys = keys;
        return token;
    }
    auto res = tokened_->Pin(keys);
    if (!res) { return res.Error(); }
    guard.Value()->pinnedBlocks = keys;
    return res;
}

void StoreEntry::CompleteGet(const RequestId& requestId)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    if (tokened_) { tokened_->Unpin(guard.Value()->pinnedBlocks); }
    guard.Value()->pinnedBlocks.clear();
}

Expected<AddressedToken> StoreEntry::PreparePut(const RequestId& requestId,
                                                const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (!guard.Value()->reservedBlocks.empty()) {
        return Status::InvalidParam("request '{}' has an unconcluded Put window", requestId);
    }
    if (!tokened_) {
        AddressedToken token;
        token.keys = keys;
        return token;
    }
    auto res = tokened_->Alloc(keys, guard.Value()->request);
    if (!res) { return res.Error(); }
    guard.Value()->reservedBlocks = keys;
    return res;
}

void StoreEntry::CompletePut(const RequestId& requestId, bool success)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    if (tokened_) { tokened_->Commit(guard.Value()->reservedBlocks, success); }
    guard.Value()->reservedBlocks.clear();
}

Status StoreEntry::SubmitPrefetch(std::size_t jobId, AddressedToken token)
{
    if (inFlightJobs_.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    if (prefetchable_ == nullptr) {
        inFlightJobs_.insert(jobId);
        completedJobs_.push_back(IoResult{jobId, true});
        return Status::Ok();
    }
    auto status = prefetchable_->Prefetch(jobId, std::move(token));
    if (status.Success()) { inFlightJobs_.insert(jobId); }
    return status;
}

Status StoreEntry::SubmitGet(std::size_t jobId, std::vector<BlockPayload> payloads,
                             AddressedToken token, const IoContext& ctx)
{
    auto check = ValidateTransfer(jobId, payloads, token, tokened_ != nullptr, inFlightJobs_);
    if (check.Failure()) { return check; }
    Store::JobDescriptor job;
    job.blockIds = std::move(token.keys);
    job.payloads = std::move(payloads);
    job.prerequisiteHandle = ctx.prerequisiteHandle;
    auto status = tokened_ != nullptr ? tokened_->Load(jobId, std::move(job), std::move(token))
                                      : loaded_->Store()->Load(jobId, std::move(job));
    if (status.Success()) { inFlightJobs_.insert(jobId); }
    return status;
}

Status StoreEntry::SubmitPut(std::size_t jobId, std::vector<BlockPayload> payloads,
                             AddressedToken token, const IoContext& ctx)
{
    auto check = ValidateTransfer(jobId, payloads, token, tokened_ != nullptr, inFlightJobs_);
    if (check.Failure()) { return check; }
    Store::JobDescriptor job;
    job.blockIds = std::move(token.keys);
    job.payloads = std::move(payloads);
    job.prerequisiteHandle = ctx.prerequisiteHandle;
    auto status = tokened_ != nullptr ? tokened_->Dump(jobId, std::move(job), std::move(token))
                                      : loaded_->Store()->Dump(jobId, std::move(job));
    if (status.Success()) { inFlightJobs_.insert(jobId); }
    return status;
}

std::vector<IoResult> StoreEntry::GetFinished()
{
    /* Completions -- synthesized or store-reported -- only exist for
     * in-flight jobs; an empty in-flight set means there is nothing to
     * drain. */
    if (inFlightJobs_.empty()) { return {}; }
    std::vector<IoResult> results;
    for (auto& result : completedJobs_) {
        inFlightJobs_.erase(result.jobId);
        results.push_back(std::move(result));
    }
    completedJobs_.clear();
    auto finished = loaded_->Store()->GetFinished();
    results.reserve(results.size() + finished.size());
    for (const auto& [jobId, success] : finished) {
        inFlightJobs_.erase(jobId);
        results.push_back(IoResult{jobId, success});
    }
    return results;
}

std::vector<IoResult> StoreEntry::Wait(const std::vector<std::size_t>& jobIds)
{
    std::vector<IoResult> results;
    std::vector<std::size_t> remaining;
    if (completedJobs_.empty()) {
        remaining = jobIds;
    } else {
        /* Pending ids are unique, so an index gives O(1) hits while the
         * jobIds loop keeps the results in waited order. */
        std::unordered_map<std::size_t, IoResult*> pending;
        pending.reserve(completedJobs_.size());
        for (auto& result : completedJobs_) { pending.emplace(result.jobId, &result); }
        for (const auto& jobId : jobIds) {
            auto it = pending.find(jobId);
            if (it == pending.end()) {
                remaining.push_back(jobId);
                continue;
            }
            inFlightJobs_.erase(jobId);
            results.push_back(std::move(*it->second));
            pending.erase(it);
        }
        completedJobs_.erase(std::remove_if(completedJobs_.begin(), completedJobs_.end(),
                                            [&pending](const IoResult& result) {
                                                return pending.count(result.jobId) == 0;
                                            }),
                             completedJobs_.end());
    }
    if (remaining.empty()) { return results; }
    auto finished = loaded_->Store()->Wait(remaining);
    results.reserve(results.size() + finished.size());
    for (const auto& [jobId, success] : finished) {
        inFlightJobs_.erase(jobId);
        results.push_back(IoResult{jobId, success});
    }
    return results;
}

Expected<StoreEntry::RequestState*> StoreEntry::RequireRequest(const RequestId& requestId)
{
    auto it = requests_.find(requestId);
    if (it == requests_.end()) {
        return Status::InvalidParam("request '{}' is not live", requestId);
    }
    return &it->second;
}

}  // namespace UC::Proxy
