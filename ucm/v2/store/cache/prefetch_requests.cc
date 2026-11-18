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
#include "prefetch_requests.h"
#include <cstddef>
#include <utility>
#include "control_ops.h"
#include "token_codec.h"

namespace UC::Store {
namespace {

/** The request frame of one engine call: the op, the blocks, and the
 *  TTL fact; op-specific fields are filled by the caller. */
ControlRequest MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                           const RequestContext& context)
{
    ControlRequest request;
    request.op = op;
    request.blockIds = blockIds;
    if (context.ttl) {
        request.hasTtl = true;
        request.ttlMs = context.ttl->count();
    }
    return request;
}

}  // namespace

PrefetchRequests::PrefetchRequests(ControlEndpoint& engine, Store* lower)
    : engine_(engine), lower_(lower)
{
}

Expected<std::pair<std::vector<std::uint8_t>, bool>> PrefetchRequests::Lookup(
    const RequestId& requestId, const std::vector<BlockId>& blockIds)
{
    auto exchange = Exchange(MakeRequest(ControlOp::Lookup, blockIds, RequestContext{}));
    if (!exchange.HasValue()) { return exchange.Error(); }
    auto hits = LookupOutcomeFrom(std::move(exchange.Value()));
    if (!hits.HasValue()) { return hits.Error(); }

    bool prefetchable = false;
    std::vector<BlockId> absent;
    for (std::size_t i = 0; i < blockIds.size(); i++) {
        if (hits.Value()[i] == 0) { absent.push_back(blockIds[i]); }
    }
    if (!absent.empty() && lower_ != nullptr) {
        auto probed = lower_->Lookup(absent);
        if (!probed.HasValue()) { return probed.Error(); }
        for (std::size_t i = 0; i < probed.Value().size(); i++) {
            if (probed.Value()[i] != 0) { prefetchable = true; }
        }
    }

    RequestRecord& record = requests_[requestId];
    record.probed.insert(blockIds.begin(), blockIds.end());
    return std::make_pair(std::move(hits.Value()), prefetchable);
}

Expected<AddressedToken> PrefetchRequests::PreparePrefetch(const RequestId& requestId,
                                                           const std::vector<BlockId>& keys,
                                                           const RequestContext& context)
{
    auto record = requests_.find(requestId);
    if (record == requests_.end()) {
        return Status::InvalidParam("request '{}' has no lookup state", requestId);
    }
    for (const BlockId& key : keys) {
        if (record->second.probed.count(key) == 0) {
            return Status::InvalidParam("request '{}' has a key outside its probed set", requestId);
        }
    }
    if (record->second.window != 0) {
        return Status::InvalidParam("request '{}' has an open prefetch window", requestId);
    }

    auto exchange = Exchange(MakeRequest(ControlOp::PreparePrefetch, keys, context));
    if (!exchange.HasValue()) { return exchange.Error(); }
    auto plan = PlanOutcomeFrom(std::move(exchange.Value()));
    if (!plan.HasValue()) { return plan; }
    if (plan.Value().keys.empty()) { return plan; }
    record->second.window = DecodePlanPayload(plan.Value().payload).batchId;
    return plan;
}

void PrefetchRequests::CompletePrefetch(const AddressedToken& plan, bool success)
{
    const std::uint64_t batchId = DecodePlanPayload(plan.payload).batchId;
    if (batchId == 0) { return; }
    for (auto& entry : requests_) {
        if (entry.second.window != batchId) { continue; }
        entry.second.window = 0;
        ConcludeWindow(batchId, success);
        return;
    }
}

void PrefetchRequests::ReleaseRequest(const RequestId& requestId)
{
    auto record = requests_.find(requestId);
    if (record == requests_.end()) { return; }
    if (record->second.window != 0) { ConcludeWindow(record->second.window, false); }
    requests_.erase(record);
}

Expected<ControlResponse> PrefetchRequests::Exchange(ControlRequest request)
{
    auto response = engine_.Execute(std::move(request));
    if (!response.HasValue()) { return response.Error(); }
    return response;
}

void PrefetchRequests::ConcludeWindow(std::uint64_t batchId, bool success)
{
    ControlRequest request = MakeRequest(ControlOp::CompletePrefetch, {}, RequestContext{});
    request.planId = batchId;
    request.success = success;
    (void)Exchange(std::move(request));
}

}  // namespace UC::Store
