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
#include <string>
#include <utility>
#include <vector>
#include "cache_config.h"
#include "ipc/uds.h"
#include "shm_layout.h"

namespace UC::Store {

/**
 * @brief Outcome of the leader election: a ticket that won carries the held
 *        flock until the process exits, a ticket that lost carries nothing.
 *        Move-only; the scheduler role never contests and always gets a
 *        lost ticket without touching any file.
 */
class LeadershipTicket {
public:
    LeadershipTicket() noexcept = default;
    ~LeadershipTicket();
    LeadershipTicket(LeadershipTicket&& other) noexcept;
    LeadershipTicket& operator=(LeadershipTicket&& other) noexcept;
    LeadershipTicket(const LeadershipTicket&) = delete;
    LeadershipTicket& operator=(const LeadershipTicket&) = delete;

    /** @brief Tells whether this process holds the leader lock. */
    [[nodiscard]] bool Won() const noexcept { return fd_ >= 0; }

    /**
     * @brief Runs the leader election for the caller's role.
     *
     * Roles worker and both open the domain's lock file (creating it if
     * needed) and try to take an exclusive non-blocking flock: the winner
     * keeps the descriptor -- and with it the lock -- for the process
     * lifetime, losers release immediately and are told so by a lost
     * ticket. The scheduler role never contests, never creates the file,
     * and gets a lost ticket.
     *
     * @return Expected the ticket on success -- won or lost; InvalidParam
     *         for an unusable unique id or an unknown role; OsApiError
     *         when open or flock fails for a reason other than the lock
     *         being taken.
     */
    [[nodiscard]] static Expected<LeadershipTicket> Contest(const CacheConfig& config);

private:
    explicit LeadershipTicket(int fd) noexcept : fd_{fd} {}

    int fd_{-1};
};

/** One accepted peer of the leader, its role and control connection. */
struct PeerSession {
    std::string role;
    Transfer::UdsConnection connection;
};

/**
 * @brief Result of one accepted-connection attempt of the leader: either the
 *        peer joined (its session is handed to the caller, who owns the
 *        connection from here on), or the peer was rejected (the leader
 *        already answered and closed; it stays in service for other peers).
 *        Transport-level failures surface as the Expected's error instead,
 *        leaving the leader in service.
 */
struct HandshakeOutcome {
    bool joined{false};
    PeerSession session;
    Status rejection{Status::Ok()};
};

/**
 * @brief The bootstrap leader of one cache domain: the process that won the
 *        flock. It reclaims stale artifacts of dead domains, creates and
 *        faults in the shared pool, and listens on the domain socket for
 *        joining peers.
 *
 * Handshake admission rules: the joining digest (unique id, block count,
 * block size, replacement policy, huge-page flag) must equal the leader's,
 * and the roles must fit the domain shape -- a worker-led domain admits
 * workers and at most one scheduler, a both-led domain admits only both
 * peers. A rejected peer gets a negative response and the leader keeps
 * serving; transport failures on one connection do not stop the service.
 */
class BootstrapLeader {
public:
    BootstrapLeader() noexcept = default;
    ~BootstrapLeader() = default;
    BootstrapLeader(BootstrapLeader&& other) noexcept = default;
    BootstrapLeader& operator=(BootstrapLeader&& other) noexcept = default;
    BootstrapLeader(const BootstrapLeader&) = delete;
    BootstrapLeader& operator=(const BootstrapLeader&) = delete;

    /**
     * @brief Brings the domain up: reclaim stale artifacts of other dead
     *        domains, drop a provably stale own socket (once the flock is
     *        held, a live predecessor would still hold it), create and fault
     *        in the shared pool, then listen. Nothing listens before the
     *        pool is fully resident. The winning ticket is kept for the
     *        leader's lifetime: while it lives, the flock stays held and
     *        the domain's artifacts are exempt from every GC scan.
     *
     * @return Expected the leader on success; InvalidParam for an unusable
     *         configuration or a lost ticket; OsApiError when a reclaim or
     *         syscall step fails. Every failure is fatal for the domain and
     *         belongs to the caller's restart policy.
     */
    [[nodiscard]] static Expected<BootstrapLeader> Create(const CacheConfig& config,
                                                          LeadershipTicket ticket);

    /**
     * @brief Blocks until one peer connects and finishes its handshake.
     *
     * @return Expected a HandshakeOutcome for every accepted connection --
     *         joined or rejected; an error Status only for transport-level
     *         failures (accept, receive, reply send), which leave the leader
     *         in service.
     */
    [[nodiscard]] Expected<HandshakeOutcome> AcceptOne();

    /**
     * @brief Convenience accept loop: serves peers one after another and
     *         keeps accepted sessions open until the listener dies. It
     *         returns only when the listener stops working, with the
     *         terminal error, or Ok after an orderly close.
     */
    Status Serve();

    /** @brief Mapping of the leader's shared pool. */
    [[nodiscard]] const ShmMapping& Mapping() const noexcept { return region_.Mapping(); }

    /** @brief Descriptor of the listening socket, for callers that multiplex
     *         it in their own event loop. */
    [[nodiscard]] int ListenerFd() const noexcept { return listener_.Fd(); }

private:
    BootstrapLeader(CacheConfig config, LeadershipTicket&& ticket, MemfdRegion&& region,
                    Transfer::UdsListener&& listener);

    /*
     * The ticket is the liveness anchor of the domain: while it lives, the
     * flock stays held and no other domain's leader can reclaim this
     * domain's artifacts. It must therefore outlive the listener and the
     * pool, hence its leading position among the members.
     */
    LeadershipTicket ticket_;
    CacheConfig config_;
    MemfdRegion region_;
    Transfer::UdsListener listener_;
    bool hasScheduler_{false};
    std::vector<PeerSession> sessions_;
};

/** Result of joining a domain: the control connection, and the shared pool's
 * mapping for data-plane roles (empty for the scheduler, which holds no
 * shared memory by design). */
struct JoinResult {
    ShmMapping mapping;
    Transfer::UdsConnection connection;
};

/**
 * @brief Joins the caller's cache domain, retrying the connection forever
 *        with exponential backoff (1ms doubling up to 100ms): the leader
 *        may simply not be up yet. Only the connect step retries; once
 *        connected, every outcome -- rejection, malformed reply, peer death
 *        mid-handshake -- is terminal and returned as the error Status.
 *
 * @return Expected the join result on success; the reconstructed rejection
 *         Status when the leader refused; a transport error otherwise.
 */
[[nodiscard]] Expected<JoinResult> JoinDomain(const CacheConfig& config);

/**
 * @brief Single-shot join: one connect, one handshake, no retry. Bounded
 *        test compositions build on this primitive; callers that can wait
 *         for the leader use JoinDomain.
 */
[[nodiscard]] Expected<JoinResult> AttemptJoinOnce(const CacheConfig& config);

}  // namespace UC::Store
