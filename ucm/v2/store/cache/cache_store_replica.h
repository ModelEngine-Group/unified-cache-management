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
#include <vector>
#include "bootstrap.h"
#include "cache_config.h"
#include "control_codec.h"
#include "data_plane.h"
#include "shm_layout.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The cache store of a "both"-role process that lost the leadership
 *        contest: every control-plane call is a synchronous round trip to
 *        the primary over the domain's control connection, executed on the
 *        caller's thread.
 *
 * The replica holds the shared pool's mapping for the data plane, which the
 * control plane leaves unused: transfers run on the caller's thread through
 * a DmaEngine over the attached pool, and completions are reported exactly
 * once through GetFinished()/Wait(). A dead peer surfaces
 * as the transport error of the next control call: errors are returned,
 * never translated into self-termination. Void operations
 * (Touch/Unpin/Commit) have no return channel -- their transport errors are
 * dropped, and the dead connection reports itself at the next
 * value-returning call.
 *
 * @note Threading contract: one calling thread or external serialization;
 *       the sequence numbers and the connection are single-consumer state.
 */
class CacheStoreReplica final : public Store {
public:
    /**
     * @brief Joins the caller's cache domain and builds the replica.
     *
     * @return Expected the replica on success; the join's error Status
     *         otherwise (rejection or transport failure).
     */
    [[nodiscard]] static Expected<std::unique_ptr<CacheStoreReplica>> Create(
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

    Prefetchable* AsPrefetchable() override { return nullptr; }

private:
    CacheStoreReplica(const CacheConfig& config, JoinResult&& join);

    /** Builds the wire request of one call, consuming the next sequence number. */
    [[nodiscard]] ControlRequest MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                             const RequestContext& context);

    /**
     * @brief One synchronous exchange: encode, send, receive, decode, verify
     *        the sequence echo.
     *
     * @return Expected the response on success; the transport or codec error
     *         otherwise. Descriptors riding along a reply are closed and
     *         rejected -- the control plane never passes any.
     */
    [[nodiscard]] Expected<ControlResponse> RoundTrip(const ControlRequest& request);

    /** @brief Void-op round trip: transport errors are dropped by design. */
    void Discard(ControlRequest request);

    CacheConfig config_;
    Transfer::UdsConnection connection_;
    /** Shared pool of the domain, held for the data plane. */
    ShmMapping mapping_;
    /* The transfers: declared after the mapping so in-flight jobs drain
     * while the mapping is still valid. */
    std::unique_ptr<DataPlane> dataPlane_;
    std::uint32_t seq_{0};
};

}  // namespace UC::Store
