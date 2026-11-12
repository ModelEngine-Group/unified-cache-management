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
#include "cache_store_worker.h"
#include <cerrno>
#include <cstdint>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace UC::Store {
namespace {

constexpr const char* kMetadataMessage =
    "cache metadata operations are not served by worker processes";

/** Writes one wakeup unit into an eventfd; the only write of a call cycle. */
void WriteEventfd(int fd) noexcept
{
    const std::uint64_t one = 1;
    while (::write(fd, &one, sizeof(one)) < 0 && errno == EINTR) {}
}

/** Closes and invalidates a descriptor member. */
void CloseFd(int& fd) noexcept
{
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

/** Registers one read interest; the fd's ownership stays with its holder. */
Status WatchFd(int epollFd, int fd) noexcept
{
    ::epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = fd;
    if (::epoll_ctl(epollFd, EPOLL_CTL_ADD, fd, &event) != 0) {
        return Status::OsApiError("registering fd {} with epoll failed: {}", fd,
                                  std::strerror(errno));
    }
    return Status::Ok();
}

}  // namespace

Expected<std::unique_ptr<CacheStoreWorker>> CacheStoreWorker::Create(const CacheConfig& config,
                                                                     LeadershipTicket ticket)
{
    std::unique_ptr<CacheStoreWorker> worker{new CacheStoreWorker{config}};
    if (ticket.Won()) {
        auto leader = BootstrapLeader::Create(config, std::move(ticket));
        if (!leader.HasValue()) { return leader.Error(); }
        if (Status st = worker->StartLeading(std::move(leader.Value())); st.Failure()) {
            /* Every failure path of StartLeading closes its own descriptors
             * and starts no thread; the leader unwinds with the local, so
             * nothing outlives this return. */
            return st;
        }
    } else {
        auto join = JoinDomain(config);
        if (!join.HasValue()) { return join.Error(); }
        worker->mapping_ = std::move(join.Value().mapping);
        /* A join connection's whole life is the handshake: dropping it with
         * the join result is the protocol, not an accident. */
    }
    /* The data plane comes up last, over whichever pool the branch holds;
     * its failure unwinds the worker with the local. */
    auto plane = DataPlane::Create(config, worker->Pool());
    if (!plane.HasValue()) { return plane.Error(); }
    worker->dataPlane_ = std::move(plane.Value());
    return worker;
}

CacheStoreWorker::CacheStoreWorker(CacheConfig config) : config_{std::move(config)} {}

CacheStoreWorker::~CacheStoreWorker()
{
    if (stopFd_ >= 0) { WriteEventfd(stopFd_); }
    if (acceptor_.joinable()) { acceptor_.join(); }
    CloseFd(stopFd_);
}

Status CacheStoreWorker::StartLeading(BootstrapLeader&& leader) noexcept
{
    stopFd_ = ::eventfd(0, EFD_CLOEXEC);
    if (stopFd_ < 0) {
        return Status::OsApiError("eventfd of the accept-loop stop signal failed: {}",
                                  std::strerror(errno));
    }
    const int epollFd = ::epoll_create1(EPOLL_CLOEXEC);
    if (epollFd < 0) {
        const int err = errno;
        CloseFd(stopFd_);
        return Status::OsApiError("epoll_create1 failed: {}", std::strerror(err));
    }
    if (Status st = WatchFd(epollFd, leader.ListenerFd()); st.Failure()) {
        CloseFd(stopFd_);
        ::close(epollFd);
        return st;
    }
    if (Status st = WatchFd(epollFd, stopFd_); st.Failure()) {
        CloseFd(stopFd_);
        ::close(epollFd);
        return st;
    }
    leader_.emplace(std::move(leader));
    /* The loop owns the epoll descriptor from here on and closes it on every
     * exit; this method's failure surface ends before the thread exists. */
    acceptor_ = std::thread{&CacheStoreWorker::RunAcceptLoop, this, epollFd};
    return Status::Ok();
}

void CacheStoreWorker::RunAcceptLoop(int epollFd)
{
    ::epoll_event events[8];
    for (;;) {
        const int ready = ::epoll_wait(epollFd, events, 8, -1);
        if (ready < 0) {
            if (errno == EINTR) { continue; }
            break;  // an epoll failure is unrecoverable in-process
        }
        for (int i = 0; i < ready; ++i) {
            if (events[i].data.fd == stopFd_) {
                ::close(epollFd);
                return;
            }
            /* One served join. The call has answered the peer whatever the
             * outcome -- admitted, rejected, or lost to a transport failure
             * -- and no session is stored: destroying the outcome closes an
             * admitted connection right here, a join being its connection's
             * whole life. */
            (void)leader_->AcceptOne();
        }
    }
    ::close(epollFd);
}

const ShmMapping& CacheStoreWorker::Pool() const
{
    /* Safe against the running accept loop: it touches only the leader's
     * admission bookkeeping, never the pool mapping. */
    return leader_ ? leader_->Mapping() : mapping_;
}

Expected<std::vector<std::uint8_t>> CacheStoreWorker::Lookup(
    const std::vector<BlockId>& /*blockIds*/)
{
    return Status::Unsupported(kMetadataMessage);
}

Expected<AddressedToken> CacheStoreWorker::Pin(const std::vector<BlockId>& /*blockIds*/)
{
    return Status::Unsupported(kMetadataMessage);
}

Expected<AddressedToken> CacheStoreWorker::Alloc(const std::vector<BlockId>& /*blockIds*/,
                                                 const RequestContext& /*context*/)
{
    return Status::Unsupported(kMetadataMessage);
}

Status CacheStoreWorker::Load(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Load(jobId, job, token);
}

Status CacheStoreWorker::Dump(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Dump(jobId, job, token);
}

std::vector<std::pair<std::size_t, bool>> CacheStoreWorker::GetFinished()
{
    return dataPlane_->GetFinished();
}

Status CacheStoreWorker::Wait(std::size_t jobId) { return dataPlane_->Wait(jobId); }

}  // namespace UC::Store
