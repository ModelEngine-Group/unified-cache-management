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

#include <memory>
#include <utility>
#include <vector>
#include "cache_buffer.h"
#include "cache_config.h"
#include "control_codec.h"
#include "control_ops.h"
#include "offloadable.h"
#include "prefetch_requests.h"
#include "prefetchable.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The cache store of a scheduler-role process: the domain's single
 *        admission decision maker.
 *
 * The domain's workers hold no engine of their own; this store serves the
 * whole admission protocol locally, from its own CacheBuffer driven over the
 * configured replacement policy. That buffer is bookkeeping only -- block
 * bytes live in the domain's shared pool, which the workers map -- so the
 * store's outputs are slot decisions and the tokens minted from them;
 * delivering tokens to the workers is the deployment's own channel, outside
 * this store.
 *
 * The domain is joined, not led: the scheduler never contests leadership.
 * The handshake admits at most one scheduler per domain and validates the
 * joiner's digest against the leader's; neither the connection nor any
 * shared memory outlives Create -- the data plane runs entirely in the
 * workers, and this store holds neither.
 *
 * The Offloadable facet serves the engine half: plans mint and receipt
 * through the local engine, while the executor half is refused -- no
 * data plane runs here, so the offload direction never drives the
 * lower tier. The facet is served only when the configuration wires a
 * lower tier: without an executor the minted plans would never
 * conclude and their pins would never release.
 *
 * The Prefetchable facet admits the tier below's data into the buffer:
 * the request glue probes the wired lower tier's presence, its only
 * call below -- a metadata lookup that needs no data plane -- while
 * windows mint and conclude through the local engine and the executor
 * half is refused. The facet is served only when the configuration
 * wires a lower tier: without one there is nothing to admit from.
 *
 * @note Threading contract: one calling thread or external serialization;
 *       the engine keeps exactly one serialized caller.
 */
class CacheStoreScheduler final : public Store, public Offloadable, public Prefetchable {
public:
    /**
     * @brief Joins the caller's cache domain and builds the engine.
     *
     * Steps: join the domain (the handshake validating digest and
     * uniqueness), instantiate the configured replacement policy, set up
     * the engine over it.
     *
     * @return Expected the scheduler on success; the failing step's error
     *         Status otherwise, with nothing left behind.
     */
    [[nodiscard]] static Expected<std::unique_ptr<CacheStoreScheduler>> Create(
        const CacheConfig& config);

    /** @brief The instance is configured at construction; this call is out
     *         of protocol and changes nothing. */
    Status Setup(const Dictionary& /*config*/) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) override;

    void Touch(const std::vector<BlockId>& blockIds, const RequestContext& context) override;

    Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) override;

    void Unpin(const std::vector<BlockId>& blockIds) override;

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                   const RequestContext& context) override;

    void Commit(const std::vector<BlockId>& blockIds, bool success) override;

    Status Load(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    std::vector<std::pair<std::size_t, bool>> GetFinished() override;

    Status Wait(std::size_t jobId) override;

    /*== The Offloadable facet ==*/

    AddressedToken PrepareOffload() override;

    void CompleteOffload(const AddressedToken& plan) override;

    Status Offload(std::size_t jobId, AddressedToken plan) override;

    Offloadable* AsOffloadable() override { return config_.lowerStore != nullptr ? this : nullptr; }

    /*== The Prefetchable facet ==*/

    Expected<std::pair<std::vector<std::uint8_t>, bool>> Lookup(
        const RequestId& requestId, const std::vector<BlockId>& blockIds) override;

    Expected<AddressedToken> PreparePrefetch(const std::vector<BlockId>& keys,
                                             const RequestContext& context) override;

    void CompletePrefetch(const AddressedToken& plan, bool success) override;

    Status Prefetch(std::size_t jobId, AddressedToken plan) override;

    void ReleaseRequest(const RequestId& requestId) override;

    Prefetchable* AsPrefetchable() override
    {
        return config_.lowerStore != nullptr ? this : nullptr;
    }

private:
    CacheStoreScheduler(const CacheConfig& config);

    /** Builds the request of one call; the engine is local, so no sequence
     *  numbers are needed. */
    [[nodiscard]] ControlRequest MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                             const RequestContext& context) const;

    /** @brief The prefetch glue's carrying channel into the local engine. */
    class LocalEndpoint final : public ControlEndpoint {
    public:
        explicit LocalEndpoint(CacheStoreScheduler& owner) : owner_{owner} {}

        Expected<ControlResponse> Execute(ControlRequest request) override
        {
            return ExecuteControl(owner_.buffer_, request);
        }

    private:
        CacheStoreScheduler& owner_;
    };

    CacheConfig config_;
    CacheBuffer buffer_;
    /** The prefetch engine half's carrying channel; declared before the
     *  glue that references it. */
    LocalEndpoint endpoint_{*this};
    /** The prefetch facet's request glue over the endpoint and the
     *  wired lower tier. */
    PrefetchRequests prefetch_{endpoint_, config_.lowerStore};
};

}  // namespace UC::Store
