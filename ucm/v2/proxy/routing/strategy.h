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

#include <cstddef>
#include "status/status.h"
#include "type/dictionary.h"
#include "type/request.h"

namespace UC::Proxy {

/**
 * @brief Pluggable routing policy of a Peer: selects the branch a request's
 *        whole windows are routed to, from the request facts of
 *        RequestContext.
 *
 * Implementations register themselves in the compile-time
 * RoutingStrategyRegistry under a configuration name
 * (UC_PROXY_REGISTER_ROUTING_STRATEGY), the peer's routing node selects one by
 * that name, and the assembly hands the strategy its dedicated parameter
 * dictionary (the routing node's "config" sub-dictionary) — the assembly
 * parses only the strategy name, never the parameters.
 *
 * Calls are serialized by the caller (the scheduler process);
 * implementations need not be thread-safe.
 */
class RoutingStrategy {
public:
    virtual ~RoutingStrategy() = default;

    /**
     * @brief Validate the parameters and bind the branch count.
     *
     * @param config      The strategy's parameter dictionary (the routing
     *                    node's "config" sub-dictionary; opaque to the
     *                    assembly).
     * @param branchCount The peer's branch count; strategies that encode
     *                    branch references validate against it.
     *
     * @return Ok, or InvalidParam on malformed parameters (e.g. a missing
     *         parameter or an out-of-range branch reference).
     */
    virtual Status Setup(const Dictionary& config, std::size_t branchCount) = 0;
    /**
     * @brief Select the branch the request's windows route to.
     *
     * Setup has validated the parameters, so the return value is always a
     * legal branch index — a strategy carries its own fallback semantics
     * (e.g. a missing fact degrades to the fallback segment).
     */
    virtual std::size_t Select(const RequestContext& request) = 0;
};

}  // namespace UC::Proxy
