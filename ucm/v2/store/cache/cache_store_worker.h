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
#include <memory>
#include <optional>
#include <thread>
#include <vector>
#include "bootstrap.h"
#include "cache_config.h"
#include "data_plane.h"
#include "shm_layout.h"
#include "store.h"
#include "type/block.h"
#include "type/request.h"
#include "type/token.h"

namespace UC::Store {

/**
 * @brief The cache store of a worker-role process: the data plane over the
 *        domain's shared pool.
 *
 * Metadata is not served here -- admission decisions belong to the processes
 * that hold the engine, and their tokens reach workers over the deployment's
 * own broadcast channel. Value-returning metadata calls answer Unsupported;
 * the void ones (Touch/Unpin/Commit) are silent no-ops, the interface having
 * no error channel for them.
 *
 * Two installments of one behavior. The leadership winner owns the domain:
 * it bootstraps the shared pool (stale-artifact GC, creation, fault-in) and
 * serves joining peers on a small accept loop -- a join ends with its
 * handshake, the pool fd having been handed over, so no connection is kept.
 * Every loser holds the attached pool mapping. Both installments run
 * identical transfers.
 *
 * A transfer consumes the token's slot addresses over the pool: each payload
 * pointer copies between the caller's memory and its slot, honoring the
 * pointer's offset within the block. Transfers execute asynchronously on the
 * data plane's DmaEngine -- Dump/Load return with the job in flight -- and
 * completions are reported exactly once through GetFinished()/Wait();
 * the Store contract's single-consumer rule serializes them.
 *
 * I/O model of the accept loop: blocking frame I/O between trusted peers; a
 * joiner dying mid-handshake is an external failure the loop does not defend
 * against, and a transport failure on one join leaves it in service.
 *
 * @note Non-movable: the accept thread captures this. The factory places
 *       the instance behind a unique_ptr, where it stays for its lifetime.
 */
class CacheStoreWorker final : public Store {
public:
    /**
     * @brief Installs the worker: the winner leads the domain, every loser
     *        attaches to it.
     *
     * @param config  The worker's domain configuration.
     * @param ticket  The leadership contest's outcome for this process.
     *
     * @return Expected the worker on success; the failing step's error
     *         Status otherwise, with nothing left behind.
     */
    [[nodiscard]] static Expected<std::unique_ptr<CacheStoreWorker>> Create(
        const CacheConfig& config, LeadershipTicket ticket);

    /** @brief Stops the accept loop (stop eventfd + join) and releases the domain. */
    ~CacheStoreWorker() override;

    CacheStoreWorker(CacheStoreWorker&&) = delete;
    CacheStoreWorker& operator=(CacheStoreWorker&&) = delete;
    CacheStoreWorker(const CacheStoreWorker&) = delete;
    CacheStoreWorker& operator=(const CacheStoreWorker&) = delete;

    /** @brief The instance is configured at construction; this call is out
     *         of protocol and changes nothing. */
    Status Setup(const Dictionary& /*config*/) override { return Status::Ok(); }

    Expected<std::vector<std::uint8_t>> Lookup(const std::vector<BlockId>& blockIds) override;

    void Touch(const std::vector<BlockId>& /*blockIds*/, const RequestContext& /*context*/) override
    {
    }

    Expected<AddressedToken> Pin(const std::vector<BlockId>& blockIds) override;

    void Unpin(const std::vector<BlockId>& /*blockIds*/) override {}

    Expected<AddressedToken> Alloc(const std::vector<BlockId>& blockIds,
                                   const RequestContext& context) override;

    void Commit(const std::vector<BlockId>& /*blockIds*/, bool /*success*/) override {}

    Status Load(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    Status Dump(std::size_t jobId, JobDescriptor job, AddressedToken token) override;

    std::vector<std::pair<std::size_t, bool>> GetFinished() override;

    Status Wait(std::size_t jobId) override;

    Prefetchable* AsPrefetchable() override { return nullptr; }

private:
    CacheStoreWorker(CacheConfig config);

    /** @brief Starts the leader's accept loop over the won domain. */
    Status StartLeading(BootstrapLeader&& leader) noexcept;

    /** @brief The accept loop: serves joins until the stop signal; owns its
     *         epoll descriptor and closes it on every exit. */
    void RunAcceptLoop(int epollFd);

    /** @brief The pool this worker transfers over. */
    [[nodiscard]] const ShmMapping& Pool() const;

    CacheConfig config_;

    /* Branch state: exactly one of the two is engaged -- the leader's pool
     * lives inside the domain it leads, a leaf holds its own attachment. */
    std::optional<BootstrapLeader> leader_;
    ShmMapping mapping_;

    /* The transfers: declared after the mapping holders so in-flight jobs
     * drain while the pool mapping is still valid. */
    std::unique_ptr<DataPlane> dataPlane_;

    /** eventfd: wakes the accept loop for shutdown; leader branch only. */
    int stopFd_{-1};
    std::thread acceptor_;
};

}  // namespace UC::Store
