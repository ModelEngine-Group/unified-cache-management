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
#include "cache_store_replica.h"
#include <unistd.h>
#include "control_ops.h"

namespace UC::Store {

Expected<std::unique_ptr<CacheStoreReplica>> CacheStoreReplica::Create(const CacheConfig& config)
{
    auto join = JoinDomain(config);
    if (!join.HasValue()) { return join.Error(); }
    std::unique_ptr<CacheStoreReplica> replica{
        new CacheStoreReplica{config, std::move(join.Value())}
    };
    /* The data plane comes up over the attached pool; its failure unwinds
     * the replica with the local. */
    auto plane = DataPlane::Create(config, replica->mapping_);
    if (!plane.HasValue()) { return plane.Error(); }
    replica->dataPlane_ = std::move(plane.Value());
    return replica;
}

CacheStoreReplica::CacheStoreReplica(const CacheConfig& config, JoinResult&& join)
    : config_{config}, connection_{std::move(join.connection)}, mapping_{std::move(join.mapping)}
{
}

ControlRequest CacheStoreReplica::MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                              const RequestContext& context)
{
    ControlRequest request;
    request.op = op;
    request.seq = ++seq_;
    request.blockIds = blockIds;
    if (context.ttl) {
        request.hasTtl = true;
        request.ttlMs = context.ttl->count();
    }
    return request;
}

Expected<ControlResponse> CacheStoreReplica::RoundTrip(const ControlRequest& request)
{
    auto encoded = EncodeRequest(request);
    if (!encoded.HasValue()) { return encoded.Error(); }
    if (Status st = connection_.SendFrame(encoded.Value()); st.Failure()) { return st; }

    std::vector<std::byte> payload;
    std::vector<int> fds;
    if (Status st = connection_.ReceiveFrame(payload, fds); st.Failure()) { return st; }
    if (!fds.empty()) {
        for (const int fd : fds) { ::close(fd); }
        return Status::InvalidParam("control reply carried {} descriptors, expected none",
                                    fds.size());
    }

    auto decoded = DecodeResponse(payload);
    if (!decoded.HasValue()) { return decoded.Error(); }
    if (decoded.Value().seq != request.seq) {
        return Status::InvalidParam("control reply has seq {}, expected {}", decoded.Value().seq,
                                    request.seq);
    }
    return decoded;
}

void CacheStoreReplica::Discard(ControlRequest request) { (void)RoundTrip(request); }

Expected<std::vector<std::uint8_t>> CacheStoreReplica::Lookup(const std::vector<BlockId>& blockIds)
{
    auto response = RoundTrip(MakeRequest(ControlOp::Lookup, blockIds, RequestContext{}));
    if (!response.HasValue()) { return response.Error(); }
    return LookupOutcomeFrom(std::move(response.Value()));
}

void CacheStoreReplica::Touch(const std::vector<BlockId>& blockIds, const RequestContext& context)
{
    Discard(MakeRequest(ControlOp::Touch, blockIds, context));
}

Expected<AddressedToken> CacheStoreReplica::Pin(const std::vector<BlockId>& blockIds)
{
    auto response = RoundTrip(MakeRequest(ControlOp::Pin, blockIds, RequestContext{}));
    if (!response.HasValue()) { return response.Error(); }
    return TokenOutcomeFrom(std::move(response.Value()), blockIds);
}

void CacheStoreReplica::Unpin(const std::vector<BlockId>& blockIds)
{
    Discard(MakeRequest(ControlOp::Unpin, blockIds, RequestContext{}));
}

Expected<AddressedToken> CacheStoreReplica::Alloc(const std::vector<BlockId>& blockIds,
                                                  const RequestContext& context)
{
    auto response = RoundTrip(MakeRequest(ControlOp::Alloc, blockIds, context));
    if (!response.HasValue()) { return response.Error(); }
    return TokenOutcomeFrom(std::move(response.Value()), blockIds);
}

void CacheStoreReplica::Commit(const std::vector<BlockId>& blockIds, bool success)
{
    ControlRequest request = MakeRequest(ControlOp::Commit, blockIds, RequestContext{});
    request.success = success;
    Discard(std::move(request));
}

Status CacheStoreReplica::Load(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Load(jobId, job, token);
}

Status CacheStoreReplica::Dump(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Dump(jobId, job, token);
}

std::vector<std::pair<std::size_t, bool>> CacheStoreReplica::GetFinished()
{
    return dataPlane_->GetFinished();
}

Status CacheStoreReplica::Wait(std::size_t jobId) { return dataPlane_->Wait(jobId); }

}  // namespace UC::Store
