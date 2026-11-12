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
#include "bootstrap.h"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "control_codec.h"

namespace UC::Store {
namespace {

/**
 * @brief Directory holding the rendezvous artifacts of every cache domain.
 *
 * A compile-time constant by design: the lock and socket paths of a domain
 * are derived from it, so a per-process value that drifts would silently put
 * two members of one domain in different rendezvous points -- they would
 * never meet and never learn why. The handshake digest cannot catch this
 * either, because the connection it protects already depends on the paths
 * agreeing.
 */
constexpr const char* kArtifactDir = "/dev/shm";

/**
 * @brief Grace period of the stale-artifact scan, in milliseconds. A lock
 *        file that nobody holds must be this old before it is reclaimed,
 *        covering the window between creating the file and acquiring its
 *        flock.
 */
constexpr std::int64_t kGcGracePeriodMs = 5000;

/**
 * @brief Length cap of a domain's unique id. The rendezvous paths must stay
 *        under the kernel's socket-path limit, and the longest path is
 *        "/dev/shm/ucm-" + id + ".sock".
 */
constexpr std::size_t kMaxUniqueIdLength = 88;

}  // namespace

/*
 * Rendezvous path scheme of a cache domain. Deliberately absent from any
 * header: every production caller lives in this translation unit, and
 * white-box tests redeclare the pair to build and clean up artifacts.
 */

/** @brief Path of a domain's leader lock: <dir>/ucm-<uniqueId>.lock. */
[[nodiscard]] std::string LockPathOf(const std::string& uniqueId)
{
    return std::string{kArtifactDir} + "/ucm-" + uniqueId + ".lock";
}

/** @brief Path of a domain's control socket: <dir>/ucm-<uniqueId>.sock. */
[[nodiscard]] std::string SocketPathOf(const std::string& uniqueId)
{
    return std::string{kArtifactDir} + "/ucm-" + uniqueId + ".sock";
}

namespace {

std::int64_t NowMs()
{
    ::timespec now{};
    ::clock_gettime(CLOCK_REALTIME, &now);
    return static_cast<std::int64_t>(now.tv_sec) * 1000 + now.tv_nsec / 1000000;
}

Status ValidateUniqueId(const std::string& uniqueId)
{
    if (uniqueId.empty()) { return Status::InvalidParam("unique id is empty"); }
    if (uniqueId.find('/') != std::string::npos) {
        return Status::InvalidParam("unique id '{}' contains a path separator", uniqueId);
    }
    if (uniqueId.size() > kMaxUniqueIdLength) {
        return Status::InvalidParam("unique id of {} bytes exceeds the {} byte cap",
                                    uniqueId.size(), kMaxUniqueIdLength);
    }
    return Status::Ok();
}

HelloRequest MakeHello(const CacheConfig& config)
{
    HelloRequest hello;
    hello.role = config.llmRole;
    hello.uniqueId = config.uniqueId;
    hello.blockTotalNumber = config.blockTotalNumber;
    hello.blockSizeBytes = config.blockSize;
    hello.replacementPolicy = config.replacementPolicy;
    hello.forceHugePage = config.forceHugePage;
    return hello;
}

/**
 * @brief Admission rules for one joining hello: the digest (unique id, block
 *        count, block size, replacement policy, huge-page flag) must equal
 *        the leader's configuration, and the joiner's role must fit the
 *        domain shape -- a worker-led domain admits workers and at most one
 *        scheduler, a both-led domain admits only both peers.
 */
Status ValidateHello(const CacheConfig& config, const HelloRequest& hello, bool hasScheduler)
{
    if (hello.uniqueId != config.uniqueId || hello.blockTotalNumber != config.blockTotalNumber ||
        hello.blockSizeBytes != config.blockSize ||
        hello.replacementPolicy != config.replacementPolicy ||
        hello.forceHugePage != config.forceHugePage) {
        return Status::InvalidParam("handshake digest mismatch: domain configuration differs");
    }
    if (config.llmRole == "worker") {
        if (hello.role != "worker" && hello.role != "scheduler") {
            return Status::InvalidParam("worker-led domain rejects joiner role '{}'", hello.role);
        }
        if (hello.role == "scheduler" && hasScheduler) {
            return Status::DuplicateKey("worker-led domain already has its scheduler");
        }
        return Status::Ok();
    }
    if (hello.role != "both") {
        return Status::InvalidParam("both-led domain rejects joiner role '{}'", hello.role);
    }
    return Status::Ok();
}

/**
 * @brief Reclaims the artifacts of dead domains: every foreign
 *        "ucm-<id>.lock" whose flock can be taken and whose mtime is older
 *        than the grace period takes its ".sock" and ".lock" with it. A held
 *        flock means a live leader; a fresh mtime covers the window between
 *        file creation and flock acquisition. The socket goes first: a crash
 *        between the two unlinks leaves the lock file behind, which the next
 *        scan rediscovers, while the reverse order could strand a socket no
 *        scan ever looks at. Individual artifacts that cannot be opened or
 *        locked are skipped -- one unopenable file must not block every
 *        domain's bootstrap.
 */
Status CollectStaleArtifacts(const std::string& selfUniqueId)
{
    DIR* dir = ::opendir(kArtifactDir);
    if (dir == nullptr) {
        return Status::OsApiError("opendir of '{}' failed: {}", kArtifactDir, std::strerror(errno));
    }

    while (const dirent* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.size() <= 9 || name.compare(0, 4, "ucm-") != 0 ||
            name.compare(name.size() - 5, 5, ".lock") != 0) {
            continue;
        }
        const std::string domainId = name.substr(4, name.size() - 9);
        if (domainId.empty() || domainId == selfUniqueId) { continue; }

        const std::string lockPath = std::string{kArtifactDir} + "/" + name;
        const int fd = ::open(lockPath.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) { continue; }

        bool stale = false;
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
            struct ::stat st{};
            if (::fstat(fd, &st) == 0) {
                const std::int64_t modifiedMs =
                    static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000 +
                    st.st_mtim.tv_nsec / 1000000;
                stale = NowMs() - modifiedMs >= kGcGracePeriodMs;
            }
        }
        if (stale) {
            const std::string socketPath = SocketPathOf(domainId);
            (void)::unlink(socketPath.c_str());
            (void)::unlink(lockPath.c_str());
        }
        ::close(fd);
    }
    ::closedir(dir);
    return Status::Ok();
}

/**
 * @brief Drives one full handshake over an established connection. Runs
 *        strictly on the joiner's own configuration: the digest it sends and
 *        the pool size it expects. Terminal on any failure -- retrying
 *        belongs to the connect step only.
 */
Expected<JoinResult> HandshakeOver(Transfer::UdsConnection&& connection, const CacheConfig& config)
{
    auto encoded = EncodeHelloRequest(MakeHello(config));
    if (!encoded.HasValue()) { return encoded.Error(); }
    if (Status st = connection.SendFrame(encoded.Value()); st.Failure()) { return st; }

    std::vector<std::byte> payload;
    std::vector<int> fds;
    if (Status st = connection.ReceiveFrame(payload, fds); st.Failure()) { return st; }

    auto decoded = DecodeResponse(payload);
    if (!decoded.HasValue()) { return decoded.Error(); }
    const ControlResponse& response = decoded.Value();
    if (response.seq != 0) {
        return Status::InvalidParam("handshake reply has seq {}, expected 0", response.seq);
    }
    if (!response.ok) {
        if (response.statusCode == 0) {
            return Status::General("handshake rejected without a status code: {}",
                                   response.message);
        }
        return Status::Make(static_cast<Status::Error>(response.statusCode), response.message);
    }

    JoinResult result;
    if (config.llmRole == "scheduler") {
        if (!fds.empty()) {
            return Status::InvalidParam(
                "scheduler handshake received {} descriptors, expected none", fds.size());
        }
        result.connection = std::move(connection);
        return result;
    }
    if (config.llmRole != "worker" && config.llmRole != "both") {
        return Status::InvalidParam("llm_role '{}' cannot join a domain", config.llmRole);
    }
    if (config.blockTotalNumber > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam("block count {} exceeds the 32-bit capacity field",
                                    config.blockTotalNumber);
    }
    if (fds.size() != 1) {
        return Status::InvalidParam("handshake received {} descriptors, expected one", fds.size());
    }

    auto total =
        ShmTotalSize(static_cast<std::uint32_t>(config.blockTotalNumber), config.blockSize);
    if (!total.HasValue()) { return total.Error(); }
    auto mapping = ShmMapping::Attach(fds[0], total.Value());
    if (!mapping.HasValue()) {
        ::close(fds[0]);
        return mapping.Error();
    }
    ::close(fds[0]);

    result.mapping = std::move(mapping.Value());
    result.connection = std::move(connection);
    return result;
}

}  // namespace

LeadershipTicket::~LeadershipTicket()
{
    if (fd_ >= 0) { ::close(fd_); }
}

LeadershipTicket::LeadershipTicket(LeadershipTicket&& other) noexcept
    : fd_{std::exchange(other.fd_, -1)}
{
}

LeadershipTicket& LeadershipTicket::operator=(LeadershipTicket&& other) noexcept
{
    if (this != &other) {
        if (fd_ >= 0) { ::close(fd_); }
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

Expected<LeadershipTicket> LeadershipTicket::Contest(const CacheConfig& config)
{
    if (Status st = ValidateUniqueId(config.uniqueId); st.Failure()) { return st; }
    if (config.llmRole == "scheduler") { return LeadershipTicket{}; }
    if (config.llmRole != "worker" && config.llmRole != "both") {
        return Status::InvalidParam("llm_role '{}' cannot contest leadership", config.llmRole);
    }

    const std::string path = LockPathOf(config.uniqueId);
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        return Status::OsApiError("open of leader lock '{}' failed: {}", path,
                                  std::strerror(errno));
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) == 0) { return LeadershipTicket{fd}; }

    const int err = errno;
    ::close(fd);
    if (err == EWOULDBLOCK || err == EAGAIN) { return LeadershipTicket{}; }
    return Status::OsApiError("flock on leader lock '{}' failed: {}", path, std::strerror(err));
}

BootstrapLeader::BootstrapLeader(CacheConfig config, LeadershipTicket&& ticket,
                                 MemfdRegion&& region, Transfer::UdsListener&& listener)
    : ticket_{std::move(ticket)},
      config_{std::move(config)},
      region_{std::move(region)},
      listener_{std::move(listener)}
{
}

Expected<BootstrapLeader> BootstrapLeader::Create(const CacheConfig& config,
                                                  LeadershipTicket ticket)
{
    if (Status st = ValidateUniqueId(config.uniqueId); st.Failure()) { return st; }
    if (!ticket.Won()) {
        return Status::InvalidParam("bootstrapping a domain requires a winning leadership ticket");
    }
    if (config.llmRole != "worker" && config.llmRole != "both") {
        return Status::InvalidParam("llm_role '{}' cannot lead a domain", config.llmRole);
    }
    if (config.blockTotalNumber > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidParam("block count {} exceeds the 32-bit capacity field",
                                    config.blockTotalNumber);
    }
    if (config.blockSize == 0) { return Status::InvalidParam("block size is zero"); }

    if (Status st = CollectStaleArtifacts(config.uniqueId); st.Failure()) { return st; }

    /*
     * Holding the flock proves no predecessor of our own domain is alive, so
     * a leftover socket of ours is stale by definition and must go before
     * the bind, which refuses an occupied path.
     */
    const std::string socketPath = SocketPathOf(config.uniqueId);
    if (::unlink(socketPath.c_str()) != 0 && errno != ENOENT) {
        return Status::OsApiError("unlink of stale socket '{}' failed: {}", socketPath,
                                  std::strerror(errno));
    }

    auto region =
        MemfdRegion::Create(static_cast<std::uint32_t>(config.blockTotalNumber), config.blockSize,
                            config.forceHugePage, "ucm-" + config.uniqueId);
    if (!region.HasValue()) { return region.Error(); }

    Transfer::UdsListener listener;
    if (Status st = listener.Listen(socketPath); st.Failure()) { return st; }

    return BootstrapLeader{config, std::move(ticket), std::move(region.Value()),
                           std::move(listener)};
}

Expected<HandshakeOutcome> BootstrapLeader::AcceptOne()
{
    auto accepted = listener_.Accept();
    if (!accepted.HasValue()) { return accepted.Error(); }
    Transfer::UdsConnection& connection = accepted.Value();

    std::vector<std::byte> payload;
    std::vector<int> fds;
    if (Status st = connection.ReceiveFrame(payload, fds); st.Failure()) { return st; }

    Status rejection = Status::Ok();
    auto decoded = DecodeHelloRequest(payload);
    if (decoded.HasValue()) {
        rejection = ValidateHello(config_, decoded.Value(), hasScheduler_);
    } else {
        rejection = decoded.Error();
    }

    ControlResponse reply;
    reply.seq = 0;
    reply.ok = rejection.Success();
    if (rejection.Failure()) {
        reply.statusCode = rejection.Underlying();
        reply.message = rejection.ToString();
    }
    auto encoded = EncodeResponse(reply);
    if (!encoded.HasValue()) { return encoded.Error(); }

    if (rejection.Success()) {
        const bool wantsMemfd = decoded.Value().role != "scheduler";
        const Status sent = wantsMemfd
                                ? connection.SendFrameWithFds(encoded.Value(), {region_.Memfd()})
                                : connection.SendFrame(encoded.Value());
        if (sent.Failure()) { return sent; }

        HandshakeOutcome outcome;
        outcome.joined = true;
        outcome.session.role = decoded.Value().role;
        outcome.session.connection = std::move(connection);
        if (decoded.Value().role == "scheduler") { hasScheduler_ = true; }
        return outcome;
    }

    if (Status st = connection.SendFrame(encoded.Value()); st.Failure()) { return st; }
    HandshakeOutcome outcome;
    outcome.rejection = rejection;
    return outcome;
}

Status BootstrapLeader::Serve()
{
    while (listener_.IsValid()) {
        auto outcome = AcceptOne();
        if (!outcome.HasValue()) {
            if (!listener_.IsValid()) { return outcome.Error(); }
            continue;
        }
        if (outcome.Value().joined) { sessions_.push_back(std::move(outcome.Value().session)); }
    }
    return Status::Ok();
}

Expected<JoinResult> AttemptJoinOnce(const CacheConfig& config)
{
    if (Status st = ValidateUniqueId(config.uniqueId); st.Failure()) { return st; }
    auto connection = Transfer::UdsConnection::Connect(SocketPathOf(config.uniqueId));
    if (!connection.HasValue()) { return connection.Error(); }
    return HandshakeOver(std::move(connection.Value()), config);
}

Expected<JoinResult> JoinDomain(const CacheConfig& config)
{
    if (Status st = ValidateUniqueId(config.uniqueId); st.Failure()) { return st; }
    const std::string path = SocketPathOf(config.uniqueId);

    constexpr std::int64_t kInitialBackoffMs = 1;
    constexpr std::int64_t kMaxBackoffMs = 100;
    std::int64_t backoffMs = kInitialBackoffMs;
    for (;;) {
        auto connection = Transfer::UdsConnection::Connect(path);
        if (connection.HasValue()) { return HandshakeOver(std::move(connection.Value()), config); }
        ::timespec pause{};
        pause.tv_sec = static_cast<time_t>(backoffMs / 1000);
        pause.tv_nsec = static_cast<long>((backoffMs % 1000) * 1000000);
        ::nanosleep(&pause, nullptr);
        backoffMs = std::min(backoffMs * 2, kMaxBackoffMs);
    }
}

}  // namespace UC::Store
