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
#pragma once

#include <cstdint>
#include <vector>
#include "cache_buffer.h"
#include "control_codec.h"
#include "status/status.h"
#include "type/block.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief Executes one control-plane request against the CacheBuffer it is
 *        given.
 *
 * Every control-plane request executes through this one function, so all
 * callers share a single op-to-engine mapping and the engine keeps exactly
 * one serialized caller.
 *
 * The TTL fact travels losslessly: hasTtl and ttlMs are rebuilt into the
 * RequestContext exactly once here. No clock is read on this path -- the
 * engine reads time itself.
 *
 * @return ControlResponse echoing the request's seq; ok with the op's result
 *         body (Lookup answers with hits, Pin/Alloc with slots plus the
 *         minted token payload, the void ops with none), or the engine's
 *         error Status recoded into statusCode and message.
 */
[[nodiscard]] ControlResponse ExecuteControl(CacheBuffer& buffer, const ControlRequest& request);

/**
 * @brief Recovers the Lookup outcome of a control response.
 *
 * @return Expected the hit vector on success -- moved out of the response;
 *         the recoded error Status of a failed response.
 */
[[nodiscard]] Expected<std::vector<std::uint8_t>> LookupOutcomeFrom(ControlResponse&& response);

/**
 * @brief Recovers the Pin/Alloc outcome of a control response: the
 *        read/write-end AddressedToken, keys echoing the requested blocks.
 *
 * @param response  The response of the matching Pin/Alloc request; its token
 *                  payload is moved into the returned token.
 * @param blockIds  The request's blocks, echoed into token.keys.
 *
 * @return Expected the token on success -- payload as carried, verified
 *         against the reported slots to catch codec-integration drift, then
 *         moved out of the response; the recoded error Status of a failed
 *         response, or a General error when payload and slots disagree.
 */
[[nodiscard]] Expected<AddressedToken> TokenOutcomeFrom(ControlResponse&& response,
                                                        const std::vector<BlockId>& blockIds);

}  // namespace UC::Store
