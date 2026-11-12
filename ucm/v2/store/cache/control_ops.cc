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
#include "control_ops.h"
#include <chrono>
#include <utility>
#include "shm_layout.h"

namespace UC::Store {
namespace {

/** Rebuilds the RequestContext of a request: the TTL fact, losslessly. */
RequestContext ContextOf(const ControlRequest& request)
{
    RequestContext context;
    if (request.hasTtl) { context.ttl = std::chrono::milliseconds{request.ttlMs}; }
    return context;
}

/** Recodes a failed engine call into the response's error fields. */
void FailWith(ControlResponse& response, const Status& status)
{
    response.ok = false;
    response.statusCode = status.Underlying();
    response.message = status.ToString();
}

/** The response's Status, recoded from the wire's error fields. */
Status StatusFromResponse(const ControlResponse& response)
{
    return Status::Make(static_cast<Status::Error>(response.statusCode), "{}", response.message);
}

}  // namespace

ControlResponse ExecuteControl(CacheBuffer& buffer, const ControlRequest& request)
{
    ControlResponse response;
    response.seq = request.seq;
    switch (request.op) {
        case ControlOp::Lookup: {
            response.hits = buffer.Lookup(request.blockIds);
            response.ok = true;
            return response;
        }
        case ControlOp::Touch: {
            buffer.Touch(request.blockIds, ContextOf(request));
            response.ok = true;
            return response;
        }
        case ControlOp::Pin:
        case ControlOp::Alloc: {
            const auto slots = request.op == ControlOp::Pin
                                   ? buffer.Pin(request.blockIds)
                                   : buffer.Reserve(request.blockIds, ContextOf(request));
            if (!slots.HasValue()) {
                FailWith(response, slots.Error());
                return response;
            }
            response.slots.reserve(slots.Value().size());
            for (const std::size_t slot : slots.Value()) {
                response.slots.push_back(static_cast<std::uint32_t>(slot));
            }
            response.tokenPayload = EncodeTokenPayload(response.slots);
            response.ok = true;
            return response;
        }
        case ControlOp::Unpin: {
            buffer.Unpin(request.blockIds);
            response.ok = true;
            return response;
        }
        case ControlOp::Commit: {
            buffer.Commit(request.blockIds, request.success);
            response.ok = true;
            return response;
        }
        default: {
            FailWith(response, Status::InvalidParam("control op {} is not a data-plane operation",
                                                    static_cast<unsigned>(request.op)));
            return response;
        }
    }
}

Expected<std::vector<std::uint8_t>> LookupOutcomeFrom(ControlResponse&& response)
{
    if (!response.ok) { return StatusFromResponse(response); }
    return std::move(response.hits);
}

Expected<AddressedToken> TokenOutcomeFrom(ControlResponse&& response,
                                          const std::vector<BlockId>& blockIds)
{
    if (!response.ok) { return StatusFromResponse(response); }
    if (DecodeTokenPayload(response.tokenPayload) != response.slots) {
        return Status::General("token payload disagrees with the reported slots");
    }
    AddressedToken token;
    token.keys = blockIds;
    token.payload = std::move(response.tokenPayload);
    return token;
}

}  // namespace UC::Store
