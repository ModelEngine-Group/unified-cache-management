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
#include <memory>
#include <utility>
#include <vector>
#include "store.h"

namespace UC::Store {

/**
 * @brief Factory-facing facade of the multiprocess cache store: parses the
 *        configuration, runs the leadership contest, and installs this
 *        process's role implementation -- the scheduler's local decision
 *        maker, the worker's leader-or-leaf data plane, or, for "both", the
 *        primary for the contest winner and a replica for every loser.
 *        Every Store method forwards to the installed implementation.
 *
 * The facade stays default-constructible (the plugin contract) and stateless
 * apart from that implementation. The Prefetchable facet is not implemented:
 * AsPrefetchable answers nullptr, so the proxy-side capability dispatch
 * serves the basic interface and synthesizes prefetch completions.
 *
 * @note Lifecycle: the upper layer guarantees exactly one Setup before
 *       any call and no calls during destruction; the facade deliberately
 *       performs no state checking of that contract (violating it is
 *       undefined behavior, by design).
 */
class CacheStore final : public Store {
public:
    CacheStore() = default;
    ~CacheStore() override = default;
    CacheStore(const CacheStore&) = delete;
    CacheStore& operator=(const CacheStore&) = delete;

    Status Setup(const Dictionary& config) override;

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) override;

    void Touch(const std::vector<BlockId>& blockIds, const RequestContext& context) override;

    Status Load(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    std::vector<std::pair<std::size_t, bool>> GetFinished() override;

    Status Wait(std::size_t jobId) override;

    /*== Addressing & admission ==*/

    Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) override;

    void Unpin(const std::vector<BlockId>& blockIds) override;

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                   const RequestContext& context) override;

    void Commit(const std::vector<BlockId>& blockIds, bool success) override;

private:
    /** The process's role implementation, installed by Setup. */
    std::unique_ptr<Store> impl_;
};

}  // namespace UC::Store
