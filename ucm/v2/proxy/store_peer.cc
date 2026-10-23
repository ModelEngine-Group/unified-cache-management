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
#include "store_peer.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace UC::Proxy {

namespace {

/* The scheduler side stamps its routing decision into the payload of every
 * token it hands out, because the worker side cannot see the request table:
 * the payload carries [4B magic][8B branch][original payload]. The magic
 * marks the payload as peer-framed, so a token that never passed through
 * this peer -- whose payload could otherwise decode as a valid branch --
 * is rejected instead of being silently misrouted and truncated. */
constexpr std::array<std::byte, 4> PEER_TOKEN_MAGIC = {
    static_cast<std::byte>('U'), static_cast<std::byte>('C'), static_cast<std::byte>('P'),
    static_cast<std::byte>('1')};
constexpr std::size_t BRANCH_PREFIX_SIZE = sizeof(std::uint64_t);
constexpr std::size_t PEER_FRAME_SIZE = PEER_TOKEN_MAGIC.size() + BRANCH_PREFIX_SIZE;

void AppendFrame(std::array<std::byte, PEER_FRAME_SIZE>& frame, std::size_t branch)
{
    std::copy(PEER_TOKEN_MAGIC.begin(), PEER_TOKEN_MAGIC.end(), frame.begin());
    const auto encoded = static_cast<std::uint64_t>(branch);
    std::memcpy(frame.data() + PEER_TOKEN_MAGIC.size(), &encoded, sizeof(encoded));
}

Expected<AddressedToken> WrapToken(std::size_t branch, Expected<AddressedToken> token)
{
    if (!token) { return token; }
    std::vector<std::byte>& payload = token.Value().payload;
    std::array<std::byte, PEER_FRAME_SIZE> frame;
    AppendFrame(frame, branch);
    payload.reserve(PEER_FRAME_SIZE + payload.size());
    payload.insert(payload.begin(), frame.begin(), frame.end());
    return token;
}

Expected<std::size_t> StripTokenFrame(AddressedToken& token, std::size_t branchCount)
{
    const auto& payload = token.payload;
    const bool framed =
        payload.size() >= PEER_TOKEN_MAGIC.size() &&
        std::equal(PEER_TOKEN_MAGIC.begin(), PEER_TOKEN_MAGIC.end(), payload.begin());
    if (!framed) { return Status::InvalidParam("addressed token was not minted by this peer"); }
    if (payload.size() < PEER_FRAME_SIZE) {
        return Status::InvalidParam("addressed token carries no branch prefix");
    }
    std::uint64_t encoded = 0;
    std::memcpy(&encoded, payload.data() + PEER_TOKEN_MAGIC.size(), sizeof(encoded));
    const std::size_t branch = static_cast<std::size_t>(encoded);
    if (branch >= branchCount) {
        return Status::InvalidParam(
            "addressed token routes to branch {} but the peer has {} branches", branch,
            branchCount);
    }
    token.payload.erase(token.payload.begin(), token.payload.begin() + PEER_FRAME_SIZE);
    return branch;
}

}  // namespace

StorePeer::StorePeer(std::vector<std::unique_ptr<StoreEntry>> branches,
                     std::unique_ptr<RoutingStrategy> routing)
    : branches_(std::move(branches)), routing_(std::move(routing))
{
    inFlightPerBranch_.resize(branches_.size(), 0);
}

Status StorePeer::OnNewRequest(const RequestContext& request)
{
    if (requests_.count(request.requestId) != 0) {
        return Status::InvalidParam("request '{}' is already live", request.requestId);
    }
    const std::size_t branch = routing_->Select(request);
    Status res = branches_[branch]->OnNewRequest(request);
    if (res.Failure()) { return res; }
    requests_.emplace(request.requestId, RequestState{branch});
    return Status::Ok();
}

void StorePeer::OnRequestFinished(const RequestId& requestId)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    branches_[guard.Value()->branch]->OnRequestFinished(requestId);
    requests_.erase(requestId);
}

Expected<std::optional<std::vector<std::uint8_t>>> StorePeer::Lookup(
    const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    return branches_[guard.Value()->branch]->Lookup(requestId, keys);
}

Expected<std::optional<std::int64_t>> StorePeer::LookupOnPrefix(const RequestId& requestId,
                                                                const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    return branches_[guard.Value()->branch]->LookupOnPrefix(requestId, keys);
}

Expected<std::optional<std::int64_t>> StorePeer::LookupOnReverse(const RequestId& requestId,
                                                                 const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    return branches_[guard.Value()->branch]->LookupOnReverse(requestId, keys);
}

void StorePeer::Touch(const RequestId& requestId, const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    branches_[guard.Value()->branch]->Touch(requestId, keys);
}

Expected<AddressedToken> StorePeer::PreparePrefetch(const RequestId& requestId)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    const std::size_t branch = guard.Value()->branch;
    return WrapToken(branch, branches_[branch]->PreparePrefetch(requestId));
}

void StorePeer::CompletePrefetch(const RequestId& requestId, bool success)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    branches_[guard.Value()->branch]->CompletePrefetch(requestId, success);
}

Expected<AddressedToken> StorePeer::PrepareGet(const RequestId& requestId,
                                               const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    const std::size_t branch = guard.Value()->branch;
    return WrapToken(branch, branches_[branch]->PrepareGet(requestId, keys));
}

void StorePeer::CompleteGet(const RequestId& requestId)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    branches_[guard.Value()->branch]->CompleteGet(requestId);
}

Expected<AddressedToken> StorePeer::PreparePut(const RequestId& requestId,
                                               const std::vector<BlockId>& keys)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return guard.Error(); }
    const std::size_t branch = guard.Value()->branch;
    return WrapToken(branch, branches_[branch]->PreparePut(requestId, keys));
}

void StorePeer::CompletePut(const RequestId& requestId, bool success)
{
    auto guard = RequireRequest(requestId);
    if (!guard) { return; }
    branches_[guard.Value()->branch]->CompletePut(requestId, success);
}

Status StorePeer::SubmitPrefetch(std::size_t jobId, AddressedToken token)
{
    if (inFlightJobs_.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    auto branch = StripTokenFrame(token, branches_.size());
    if (!branch) { return branch.Error(); }
    auto status = branches_[branch.Value()]->SubmitPrefetch(jobId, std::move(token));
    if (status.Success()) {
        inFlightJobs_.emplace(jobId, branch.Value());
        ++inFlightPerBranch_[branch.Value()];
    }
    return status;
}

Status StorePeer::SubmitGet(std::size_t jobId, std::vector<BlockPayload> payloads,
                            AddressedToken token, const IoContext& ctx)
{
    if (inFlightJobs_.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    auto branch = StripTokenFrame(token, branches_.size());
    if (!branch) { return branch.Error(); }
    auto status =
        branches_[branch.Value()]->SubmitGet(jobId, std::move(payloads), std::move(token), ctx);
    if (status.Success()) {
        inFlightJobs_.emplace(jobId, branch.Value());
        ++inFlightPerBranch_[branch.Value()];
    }
    return status;
}

Status StorePeer::SubmitPut(std::size_t jobId, std::vector<BlockPayload> payloads,
                            AddressedToken token, const IoContext& ctx)
{
    if (inFlightJobs_.count(jobId) != 0) {
        return Status::InvalidParam("job '{}' is already in flight", jobId);
    }
    auto branch = StripTokenFrame(token, branches_.size());
    if (!branch) { return branch.Error(); }
    auto status =
        branches_[branch.Value()]->SubmitPut(jobId, std::move(payloads), std::move(token), ctx);
    if (status.Success()) {
        inFlightJobs_.emplace(jobId, branch.Value());
        ++inFlightPerBranch_[branch.Value()];
    }
    return status;
}

std::vector<IoResult> StorePeer::GetFinished()
{
    std::vector<IoResult> results;
    /* A branch without unconsumed in-flight jobs cannot report completions:
     * every job enters a branch through this peer and each completion is
     * consumed exactly once, so idle branches are skipped. */
    for (std::size_t branch = 0; branch < branches_.size(); branch++) {
        if (inFlightPerBranch_[branch] == 0) { continue; }
        for (auto& result : branches_[branch]->GetFinished()) {
            inFlightJobs_.erase(result.jobId);
            --inFlightPerBranch_[branch];
            results.push_back(std::move(result));
        }
    }
    return results;
}

std::vector<IoResult> StorePeer::Wait(const std::vector<std::size_t>& jobIds)
{
    std::vector<IoResult> results;
    std::unordered_map<std::size_t, std::vector<std::size_t>> groups;
    for (const auto& jobId : jobIds) {
        auto it = inFlightJobs_.find(jobId);
        if (it != inFlightJobs_.end()) { groups[it->second].push_back(jobId); }
    }
    for (std::size_t branch = 0; branch < branches_.size(); branch++) {
        auto it = groups.find(branch);
        if (it == groups.end()) { continue; }
        for (auto& result : branches_[branch]->Wait(it->second)) {
            inFlightJobs_.erase(result.jobId);
            --inFlightPerBranch_[branch];
            results.push_back(std::move(result));
        }
    }
    return results;
}

Expected<StorePeer::RequestState*> StorePeer::RequireRequest(const RequestId& requestId)
{
    auto it = requests_.find(requestId);
    if (it == requests_.end()) {
        return Status::InvalidParam("request '{}' is not live", requestId);
    }
    return &it->second;
}

}  // namespace UC::Proxy
