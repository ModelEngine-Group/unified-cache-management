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
                        const AddressedToken& token,
                        const std::unordered_set<std::size_t>& inFlightJobs)
{
    if (inFlightJobs.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    if (payloads.size() != token.keys.size()) {
        return Status::InvalidParam("job '{}' pairs {} payloads with {} keys", jobId,
                                    payloads.size(), token.keys.size());
    }
    return Status::Ok();
}

}  // namespace

StoreEntry::StoreEntry(std::unique_ptr<LoadedStore> loaded) : loaded_(std::move(loaded))
{
    prefetchable_ = loaded_->Store()->AsPrefetchable();
    offloadable_ = loaded_->Store()->AsOffloadable();
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
    /* Every open Get/Put window is concluded individually; the store's
     * pin and reservation references are per window. */
    for (const auto& pinned : state.getPins) { loaded_->Store()->Unpin(pinned); }
    for (const auto& reserved : state.putReservations) {
        loaded_->Store()->Commit(reserved, false);
    }
    requests_.erase(it);
}

Expected<std::pair<std::vector<std::uint8_t>, bool>> StoreEntry::Lookup(
    const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->Lookup(requestId, keys); }
    auto res = loaded_->Store()->Lookup(keys);
    if (!res) { return res.Error(); }
    return std::make_pair(std::move(res).Value(), false);
}

Expected<std::pair<std::int64_t, bool>> StoreEntry::LookupOnPrefix(const RequestId& requestId,
                                                                   const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->LookupOnPrefix(requestId, keys); }
    auto res = loaded_->Store()->LookupOnPrefix(keys);
    if (!res) { return res.Error(); }
    return std::make_pair(res.Value(), false);
}

Expected<std::pair<std::int64_t, bool>> StoreEntry::LookupOnReverse(
    const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    if (prefetchable_) { return prefetchable_->LookupOnReverse(requestId, keys); }
    auto res = loaded_->Store()->LookupOnReverse(keys);
    if (!res) { return res.Error(); }
    return std::make_pair(res.Value(), false);
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
    auto res = loaded_->Store()->Pin(keys);
    if (!res) { return res.Error(); }
    guard.Value()->getPins.push_back(keys);
    return res;
}

void StoreEntry::CompleteGet(const RequestId& requestId, const AddressedToken& token)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    /* Conclude exactly the window the token addresses: the one recorded
     * window whose keys match, leaving any others open. A token matching
     * no open window -- already concluded, foreign, or minted by another
     * request -- never reaches the store. */
    auto& windows = guard.Value()->getPins;
    auto it = std::find(windows.begin(), windows.end(), token.keys);
    if (it == windows.end()) { return; }
    loaded_->Store()->Unpin(*it);
    windows.erase(it);
}

Expected<AddressedToken> StoreEntry::PreparePut(const RequestId& requestId,
                                                const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    auto res = loaded_->Store()->Alloc(keys, guard.Value()->request);
    if (!res) { return res.Error(); }
    guard.Value()->putReservations.push_back(keys);
    return res;
}

void StoreEntry::CompletePut(const RequestId& requestId, const AddressedToken& token, bool success)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    /* See CompleteGet: one matching window concluded, others left open. */
    auto& windows = guard.Value()->putReservations;
    auto it = std::find(windows.begin(), windows.end(), token.keys);
    if (it == windows.end()) { return; }
    loaded_->Store()->Commit(*it, success);
    windows.erase(it);
}

std::vector<AddressedToken> StoreEntry::PrepareOffload()
{
    std::vector<AddressedToken> plans;
    if (offloadable_ == nullptr) { return plans; }
    AddressedToken plan = offloadable_->PrepareOffload();
    /* An idle store mints an empty token (no keys): nothing to drain. */
    if (!plan.keys.empty()) { plans.push_back(std::move(plan)); }
    return plans;
}

void StoreEntry::CompleteOffload(const AddressedToken& plan)
{
    if (offloadable_ != nullptr) { offloadable_->CompleteOffload(plan); }
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
    auto check = ValidateTransfer(jobId, payloads, token, inFlightJobs_);
    if (check.Failure()) { return check; }
    Store::JobDescriptor job;
    job.blockIds = std::move(token.keys);
    job.payloads = std::move(payloads);
    job.prerequisiteHandle = ctx.prerequisiteHandle;
    auto status = loaded_->Store()->Load(jobId, std::move(job), std::move(token));
    if (status.Success()) { inFlightJobs_.insert(jobId); }
    return status;
}

Status StoreEntry::SubmitPut(std::size_t jobId, std::vector<BlockPayload> payloads,
                             AddressedToken token, const IoContext& ctx)
{
    auto check = ValidateTransfer(jobId, payloads, token, inFlightJobs_);
    if (check.Failure()) { return check; }
    Store::JobDescriptor job;
    job.blockIds = std::move(token.keys);
    job.payloads = std::move(payloads);
    job.prerequisiteHandle = ctx.prerequisiteHandle;
    auto status = loaded_->Store()->Dump(jobId, std::move(job), std::move(token));
    if (status.Success()) { inFlightJobs_.insert(jobId); }
    return status;
}

Status StoreEntry::SubmitOffload(std::size_t jobId, AddressedToken plan)
{
    if (inFlightJobs_.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    /* A store that never could have minted the plan cannot execute it:
     * a synthesized success here would falsely license the plan's
     * pin release. */
    if (offloadable_ == nullptr) {
        return Status::InvalidParam("store cannot execute offload plans");
    }
    auto status = offloadable_->Offload(jobId, std::move(plan));
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

Status StoreEntry::Wait(std::size_t jobId)
{
    /* Ids unknown here or already consumed are rejected: only in-flight
     * jobs reach the store, so the store never sees a Wait for a foreign
     * id. */
    if (inFlightJobs_.count(jobId) == 0) {
        return Status::InvalidParam("job '{}' is not in flight", jobId);
    }
    inFlightJobs_.erase(jobId);
    /* Synthetic completions (submits the store never saw) are served from
     * the pending queue without a store round trip. */
    auto it = std::find_if(completedJobs_.begin(), completedJobs_.end(),
                           [jobId](const IoResult& result) { return result.jobId == jobId; });
    if (it != completedJobs_.end()) {
        const bool success = it->success;
        completedJobs_.erase(it);
        return success ? Status::Ok() : Status::General("job '{}' failed", jobId);
    }
    return loaded_->Store()->Wait(jobId);
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
