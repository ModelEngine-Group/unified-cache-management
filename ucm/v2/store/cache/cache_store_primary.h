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

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "bootstrap.h"
#include "cache_buffer.h"
#include "cache_config.h"
#include "control_codec.h"
#include "data_plane.h"
#include "ipc/uds.h"
#include "policy/replacement_policy.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The cache store of a "both"-role process that won the leadership
 *        contest: it owns the domain's single CacheBuffer and serves the
 *        whole domain's control plane on one pump thread.
 *
 * The pump is the only serializer of the engine: it multiplexes the domain
 * listener (replicas joining), every replica control connection, and its
 * own local calls over a level-triggered epoll, executing each request
 * through ExecuteControl in arrival order. The processing order is
 * therefore the domain's total order of metadata decisions.
 *
 * Local calls marshal into the pump through a lock-free single-slot
 * rendezvous: the primary's caller issues one control call and blocks for
 * its response before issuing the next, so at most one local request is
 * ever in flight and the slot never contends. A release-store of the
 * pending flag publishes the request; the pump's acquire-load sees it,
 * moves the request out, and fulfills the promise.
 *
 * The data plane runs on the caller's thread through a DmaEngine over the
 * led pool; the pump never touches it, so the Store contract's serialized
 * caller is the engine's only thread.
 *
 * I/O model: blocking frame I/O -- every connection is a strict
 * request-response exchange between trusted peers; a peer dying mid-frame
 * is an external failure the pump does not defend against. A malformed
 * request (undecodable, or carrying descriptors) drops the connection
 * without an error reply.
 *
 * @note Non-movable: the pump thread captures this. The factory places the
 *       instance behind a unique_ptr, where it stays for its lifetime.
 */
class CacheStorePrimary final : public Store {
public:
    /**
     * @brief Brings the domain up and starts the pump.
     *
     * Steps: bootstrap the domain as leader (stale-artifact GC, shared
     * pool, listener), set up the engine over the configured replacement
     * policy, create the epoll and its two eventfds (local-call queue,
     * stop signal), register the listener, and start the pump thread last.
     *
     * @return Expected the primary on success; the failing step's error
     *         Status otherwise, with nothing left behind.
     */
    [[nodiscard]] static Expected<std::unique_ptr<CacheStorePrimary>> Create(
        const CacheConfig& config, LeadershipTicket ticket);

    /** @brief Stops the pump (stop eventfd + join) and releases the domain. */
    ~CacheStorePrimary() override;

    CacheStorePrimary(CacheStorePrimary&&) = delete;
    CacheStorePrimary& operator=(CacheStorePrimary&&) = delete;
    CacheStorePrimary(const CacheStorePrimary&) = delete;
    CacheStorePrimary& operator=(const CacheStorePrimary&) = delete;

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
    CacheStorePrimary(const CacheConfig& config, BootstrapLeader&& leader,
                      std::unique_ptr<ReplacementPolicy> replacementPolicy);

    /** @brief Engine setup, epoll/eventfd creation, registration, pump start. */
    Status Init() noexcept;

    /** @brief The pump loop: dispatch epoll events until the stop signal. */
    void RunPump();

    /** @brief Accepts one joining replica and registers its connection. */
    void AcceptPeer();

    /** @brief Serves one request of a connected replica; drops it on failure. */
    void ServePeer(int fd);

    /** @brief Removes a replica: deregisters and closes its connection. */
    void DropPeer(int fd) noexcept;

    /** @brief Serves the pending local request, fulfilling its promise. */
    void ServeLocal();

    /**
     * @brief Marshals one local call into the pump and blocks for the
     *        response.
     */
    ControlResponse CallPump(ControlRequest request);

    /** @brief Builds the local request of one call (no sequence: no wire). */
    [[nodiscard]] ControlRequest MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                             const RequestContext& context) const;

    /** @brief Registers one fd as readable, level-triggered. */
    Status Watch(int fd) noexcept;

    /** @brief Closes the pump's own descriptors; idempotent. */
    void CloseFds() noexcept;

    CacheConfig config_;
    /** The engine. Pump-thread-only state. */
    CacheBuffer buffer_;
    /** The domain: leadership ticket, shared pool, listener; liveness anchor. */
    BootstrapLeader leader_;
    std::unique_ptr<ReplacementPolicy> replacementPolicy_;
    /* The transfers: declared after the leader so in-flight jobs drain
     * while the led pool mapping is still valid. */
    std::unique_ptr<DataPlane> dataPlane_;
    int epollFd_{-1};
    /** eventfd: wakes the pump for a pending local call. */
    int queueFd_{-1};
    /** eventfd: wakes the pump for shutdown. */
    int stopFd_{-1};
    /** Replica connections by descriptor; pump-thread-only state. */
    std::unordered_map<int, Transfer::UdsConnection> peers_;
    std::thread pump_;

    /* Single-slot rendezvous state: caller writes, pump reads. */
    ControlRequest localRequest_;
    std::promise<ControlResponse> localDone_;
    std::atomic_bool localPending_{false};
};

}  // namespace UC::Store
