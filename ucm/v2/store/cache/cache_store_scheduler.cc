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
#include "cache_store_scheduler.h"
#include "bootstrap.h"
#include "control_ops.h"
#include "policy/replacement_policy_registry.h"

namespace UC::Store {
namespace {

constexpr const char* kDataPlaneMessage = "cache data-plane jobs are not supported";

}  // namespace

Expected<std::unique_ptr<CacheStoreScheduler>> CacheStoreScheduler::Create(
    const CacheConfig& config)
{
    auto join = JoinDomain(config);
    if (!join.HasValue()) { return join.Error(); }
    /* A join connection's whole life is the handshake; the mapping of this
     * role is empty by design. Both drop with the local here. */

    auto policy = ReplacementPolicyRegistry::Instance().Create(config.replacementPolicy);
    if (!policy.HasValue()) { return policy.Error(); }
    std::unique_ptr<ReplacementPolicy> replacementPolicy{policy.Value()};

    std::unique_ptr<CacheStoreScheduler> scheduler{new CacheStoreScheduler{config}};
    if (Status st = scheduler->buffer_.Setup(config.blockTotalNumber, std::move(replacementPolicy));
        st.Failure()) {
        return st;
    }
    return scheduler;
}

CacheStoreScheduler::CacheStoreScheduler(const CacheConfig& config) : config_{config} {}

ControlRequest CacheStoreScheduler::MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                                const RequestContext& context) const
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

Expected<std::vector<std::uint8_t>> CacheStoreScheduler::Lookup(
    const std::vector<BlockId>& blockIds)
{
    return LookupOutcomeFrom(
        ExecuteControl(buffer_, MakeRequest(ControlOp::Lookup, blockIds, RequestContext{})));
}

void CacheStoreScheduler::Touch(const std::vector<BlockId>& blockIds, const RequestContext& context)
{
    (void)ExecuteControl(buffer_, MakeRequest(ControlOp::Touch, blockIds, context));
}

Expected<AddressedToken> CacheStoreScheduler::Pin(const std::vector<BlockId>& blockIds)
{
    return TokenOutcomeFrom(
        ExecuteControl(buffer_, MakeRequest(ControlOp::Pin, blockIds, RequestContext{})), blockIds);
}

void CacheStoreScheduler::Unpin(const std::vector<BlockId>& blockIds)
{
    (void)ExecuteControl(buffer_, MakeRequest(ControlOp::Unpin, blockIds, RequestContext{}));
}

Expected<AddressedToken> CacheStoreScheduler::Alloc(const std::vector<BlockId>& blockIds,
                                                    const RequestContext& context)
{
    return TokenOutcomeFrom(
        ExecuteControl(buffer_, MakeRequest(ControlOp::Alloc, blockIds, context)), blockIds);
}

void CacheStoreScheduler::Commit(const std::vector<BlockId>& blockIds, bool success)
{
    ControlRequest request = MakeRequest(ControlOp::Commit, blockIds, RequestContext{});
    request.success = success;
    (void)ExecuteControl(buffer_, std::move(request));
}

Status CacheStoreScheduler::Load(std::size_t /*jobId*/, JobDescriptor /*job*/,
                                 AddressedToken /*token*/)
{
    return Status::Unsupported(kDataPlaneMessage);
}

Status CacheStoreScheduler::Dump(std::size_t /*jobId*/, JobDescriptor /*job*/,
                                 AddressedToken /*token*/)
{
    return Status::Unsupported(kDataPlaneMessage);
}

std::vector<std::pair<std::size_t, bool>> CacheStoreScheduler::GetFinished() { return {}; }

Status CacheStoreScheduler::Wait(std::size_t jobId)
{
    return Status::InvalidParam("job {} is unknown", jobId);
}

}  // namespace UC::Store
