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
#include "cache_store_primary.h"
#include <cerrno>
#include <cstring>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include "control_ops.h"
#include "policy/replacement_policy_registry.h"

namespace UC::Store {
namespace {

/** Writes one wakeup unit into an eventfd; the only write of a call cycle. */
void WriteEventfd(int fd) noexcept
{
    const std::uint64_t one = 1;
    while (::write(fd, &one, sizeof(one)) < 0 && errno == EINTR) {}
}

/** Drains an eventfd: the level-triggered interest goes away with the count. */
void DrainEventfd(int fd) noexcept
{
    std::uint64_t count = 0;
    while (::read(fd, &count, sizeof(count)) < 0 && errno == EINTR) {}
}

}  // namespace

Expected<std::unique_ptr<CacheStorePrimary>> CacheStorePrimary::Create(const CacheConfig& config,
                                                                       LeadershipTicket ticket)
{
    /* Configuration errors surface before any domain side effect. */
    auto policy = ReplacementPolicyRegistry::Instance().Create(config.replacementPolicy);
    if (!policy.HasValue()) { return policy.Error(); }

    auto leader = BootstrapLeader::Create(config, std::move(ticket));
    if (!leader.HasValue()) { return leader.Error(); }

    std::unique_ptr<CacheStorePrimary> primary{
        new CacheStorePrimary{config, std::move(leader.Value()),
                              std::unique_ptr<ReplacementPolicy>{policy.Value()}}
    };
    if (Status st = primary->Init(); st.Failure()) { return st; }
    /* The data plane comes up over the led pool, after the pump: the pump
     * serves metadata only and never touches the engine. */
    auto plane = DataPlane::Create(config, primary->leader_.Mapping());
    if (!plane.HasValue()) { return plane.Error(); }
    primary->dataPlane_ = std::move(plane.Value());
    return primary;
}

CacheStorePrimary::CacheStorePrimary(const CacheConfig& config, BootstrapLeader&& leader,
                                     std::unique_ptr<ReplacementPolicy> replacementPolicy)
    : config_{config}, leader_{std::move(leader)}, replacementPolicy_{std::move(replacementPolicy)}
{
}

CacheStorePrimary::~CacheStorePrimary()
{
    if (stopFd_ >= 0) { WriteEventfd(stopFd_); }
    if (pump_.joinable()) { pump_.join(); }
    CloseFds();
}

Status CacheStorePrimary::Init() noexcept
{
    if (Status st = buffer_.Setup(config_.blockTotalNumber, std::move(replacementPolicy_));
        st.Failure()) {
        return st;
    }

    epollFd_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (epollFd_ < 0) {
        return Status::OsApiError("epoll_create1 failed: {}", std::strerror(errno));
    }
    queueFd_ = ::eventfd(0, EFD_CLOEXEC);
    if (queueFd_ < 0) {
        const int err = errno;
        CloseFds();
        return Status::OsApiError("eventfd of the local call queue failed: {}", std::strerror(err));
    }
    stopFd_ = ::eventfd(0, EFD_CLOEXEC);
    if (stopFd_ < 0) {
        const int err = errno;
        CloseFds();
        return Status::OsApiError("eventfd of the pump stop signal failed: {}", std::strerror(err));
    }

    if (Status st = Watch(leader_.ListenerFd()); st.Failure()) {
        CloseFds();
        return st;
    }
    if (Status st = Watch(queueFd_); st.Failure()) {
        CloseFds();
        return st;
    }
    if (Status st = Watch(stopFd_); st.Failure()) {
        CloseFds();
        return st;
    }

    pump_ = std::thread{&CacheStorePrimary::RunPump, this};
    return Status::Ok();
}

Status CacheStorePrimary::Watch(int fd) noexcept
{
    ::epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = fd;
    if (::epoll_ctl(epollFd_, EPOLL_CTL_ADD, fd, &event) != 0) {
        return Status::OsApiError("registering fd {} with epoll failed: {}", fd,
                                  std::strerror(errno));
    }
    return Status::Ok();
}

void CacheStorePrimary::CloseFds() noexcept
{
    for (int* fd : {&epollFd_, &queueFd_, &stopFd_}) {
        if (*fd >= 0) {
            ::close(*fd);
            *fd = -1;
        }
    }
}

void CacheStorePrimary::RunPump()
{
    ::epoll_event events[16];
    for (;;) {
        const int ready = ::epoll_wait(epollFd_, events, 16, -1);
        if (ready < 0) {
            if (errno == EINTR) { continue; }
            return;  // an epoll failure is unrecoverable in-process
        }
        for (int i = 0; i < ready; ++i) {
            const int fd = events[i].data.fd;
            if (fd == stopFd_) { return; }
            if (fd == queueFd_) {
                ServeLocal();
                continue;
            }
            if (fd == leader_.ListenerFd()) {
                AcceptPeer();
                continue;
            }
            ServePeer(fd);
        }
    }
}

void CacheStorePrimary::AcceptPeer()
{
    auto outcome = leader_.AcceptOne();
    if (!outcome.HasValue()) { return; }      // transport failure: stay in service
    if (!outcome.Value().joined) { return; }  // rejected peer: already answered

    Transfer::UdsConnection& connection = outcome.Value().session.connection;
    const int fd = connection.Fd();
    if (Watch(fd).Failure()) { return; }  // connection dtor closes it

    peers_.emplace(fd, std::move(outcome.Value().session.connection));
}

void CacheStorePrimary::ServePeer(int fd)
{
    auto it = peers_.find(fd);
    if (it == peers_.end()) { return; }  // stale event of an already-dropped peer

    std::vector<std::byte> payload;
    std::vector<int> fds;
    if (it->second.ReceiveFrame(payload, fds).Failure()) {
        DropPeer(fd);
        return;
    }
    if (!fds.empty()) {
        /* Control requests never carry descriptors; a peer that sends any is
         * out of protocol and loses its connection. */
        for (const int received : fds) { ::close(received); }
        DropPeer(fd);
        return;
    }

    auto request = DecodeRequest(payload);
    if (!request.HasValue()) {
        /* Undecodable: the sequence cannot be trusted for an error reply. */
        DropPeer(fd);
        return;
    }

    auto encoded = EncodeResponse(ExecuteControl(buffer_, request.Value()));
    if (!encoded.HasValue()) {
        DropPeer(fd);
        return;
    }
    if (it->second.SendFrame(encoded.Value()).Failure()) { DropPeer(fd); }
}

void CacheStorePrimary::DropPeer(int fd) noexcept
{
    (void)::epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, nullptr);
    peers_.erase(fd);
}

void CacheStorePrimary::ServeLocal()
{
    DrainEventfd(queueFd_);
    if (!localPending_.load(std::memory_order_acquire)) { return; }
    /* Move the request out before un-pending: the caller cannot offer the
     * next one until this response is fulfilled, so the slot stays race-free. */
    ControlRequest request = std::move(localRequest_);
    localPending_.store(false, std::memory_order_release);
    localDone_.set_value(ExecuteControl(buffer_, request));
}

ControlResponse CacheStorePrimary::CallPump(ControlRequest request)
{
    localRequest_ = std::move(request);
    localDone_ = std::promise<ControlResponse>{};
    auto fulfilled = localDone_.get_future();
    localPending_.store(true, std::memory_order_release);
    WriteEventfd(queueFd_);
    return fulfilled.get();
}

ControlRequest CacheStorePrimary::MakeRequest(ControlOp op, const std::vector<BlockId>& blockIds,
                                              const RequestContext& context) const
{
    ControlRequest request;
    request.op = op;
    request.blockIds = blockIds;
    if (context.ttl) {
        request.hasTtl = true;
        request.ttlMs = context.ttl->count();
    }
    return request;
}

Expected<std::vector<std::uint8_t>> CacheStorePrimary::Lookup(const std::vector<BlockId>& blockIds)
{
    return LookupOutcomeFrom(CallPump(MakeRequest(ControlOp::Lookup, blockIds, RequestContext{})));
}

void CacheStorePrimary::Touch(const std::vector<BlockId>& blockIds, const RequestContext& context)
{
    (void)CallPump(MakeRequest(ControlOp::Touch, blockIds, context));
}

Expected<AddressedToken> CacheStorePrimary::Pin(const std::vector<BlockId>& blockIds)
{
    return TokenOutcomeFrom(CallPump(MakeRequest(ControlOp::Pin, blockIds, RequestContext{})),
                            blockIds);
}

void CacheStorePrimary::Unpin(const std::vector<BlockId>& blockIds)
{
    (void)CallPump(MakeRequest(ControlOp::Unpin, blockIds, RequestContext{}));
}

Expected<AddressedToken> CacheStorePrimary::Alloc(const std::vector<BlockId>& blockIds,
                                                  const RequestContext& context)
{
    return TokenOutcomeFrom(CallPump(MakeRequest(ControlOp::Alloc, blockIds, context)), blockIds);
}

void CacheStorePrimary::Commit(const std::vector<BlockId>& blockIds, bool success)
{
    ControlRequest request = MakeRequest(ControlOp::Commit, blockIds, RequestContext{});
    request.success = success;
    (void)CallPump(std::move(request));
}

Status CacheStorePrimary::Load(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Load(jobId, job, token);
}

Status CacheStorePrimary::Dump(std::size_t jobId, JobDescriptor job, AddressedToken token)
{
    return dataPlane_->Dump(jobId, job, token);
}

std::vector<std::pair<std::size_t, bool>> CacheStorePrimary::GetFinished()
{
    return dataPlane_->GetFinished();
}

Status CacheStorePrimary::Wait(std::size_t jobId) { return dataPlane_->Wait(jobId); }

}  // namespace UC::Store
