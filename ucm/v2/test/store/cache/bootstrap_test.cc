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
#include "cache/bootstrap.h"
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "cache/shm_layout.h"
#include "status/status.h"

namespace UC::Store {

/*
 * Rendezvous path helpers are defined in bootstrap.cc with external linkage
 * but declared in no header. The signatures below must stay identical to the
 * definitions: C++ name mangling turns any drift into a link error, so the
 * pair cannot silently fall out of sync.
 */
[[nodiscard]] std::string LockPathOf(const std::string& uniqueId);
[[nodiscard]] std::string SocketPathOf(const std::string& uniqueId);

namespace {

constexpr std::uint64_t kSlotMagic = 0xDEADBEEFCAFEF00DULL;
constexpr std::size_t kBlockSize = std::size_t{1} << 20;
constexpr std::size_t kCapacity = 2;

/** What a forked child tells its parent. kind: 3 step passed, 4 step failed,
 * 1 peer joined, 2 peer rejected, 0 fatal; code is a status code or 0. */
struct ChildReport {
    std::int32_t kind{0};
    std::int32_t code{0};
};

bool WriteAll(int fd, const void* data, std::size_t len)
{
    const auto* cursor = static_cast<const char*>(data);
    while (len > 0) {
        const ssize_t n = ::write(fd, cursor, len);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        cursor += n;
        len -= static_cast<std::size_t>(n);
    }
    return true;
}

bool ReadAll(int fd, void* data, std::size_t len)
{
    auto* cursor = static_cast<char*>(data);
    while (len > 0) {
        const ssize_t n = ::read(fd, cursor, len);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (n == 0) { return false; }
        cursor += n;
        len -= static_cast<std::size_t>(n);
    }
    return true;
}

bool WriteReport(int fd, const ChildReport& report)
{
    return WriteAll(fd, &report, sizeof(report));
}

bool ReadReport(int fd, ChildReport& report) { return ReadAll(fd, &report, sizeof(report)); }

/** Bounds a forked child: a step that blocks forever exits with code 99
 * instead of hanging the suite. Children never run gtest assertions. */
void InstallWatchdog()
{
    struct ::sigaction action{};
    action.sa_handler = [](int) { ::_exit(99); };
    ::sigemptyset(&action.sa_mask);
    ::sigaction(SIGALRM, &action, nullptr);
    ::alarm(10);
}

void TouchFile(const std::string& path)
{
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::write(fd, "x", 1), 1);
    ::close(fd);
}

void Backdate(const std::string& path, std::int64_t ageNs)
{
    const std::int64_t past = static_cast<std::int64_t>(::time(nullptr)) * 1000000000LL - ageNs;
    const struct ::timespec when{static_cast<time_t>(past / 1000000000),
                                 static_cast<long>(past % 1000000000)};
    const struct ::timespec times[2]{when, when};
    ASSERT_EQ(::utimensat(AT_FDCWD, path.c_str(), times, 0), 0);
}

bool Exists(const std::string& path)
{
    struct ::stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

int StatusCode(const Status& status) { return status.Underlying(); }

class BootstrapTest : public ::testing::Test {
protected:
    void TearDown() override
    {
        for (const std::string& id : usedIds_) {
            ::unlink(LockPathOf(id).c_str());
            ::unlink(SocketPathOf(id).c_str());
        }
    }

    /** Unique per test process and per call, so parallel runs never collide. */
    std::string MakeUniqueId(const char* label)
    {
        static std::uint32_t counter = 0;
        std::string id = "gtest-";
        id += label;
        id += "-";
        id += std::to_string(::getpid());
        id += "-";
        id += std::to_string(counter++);
        usedIds_.push_back(id);
        return id;
    }

    CacheConfig MakeConfig(const std::string& role, const std::string& uniqueId,
                           std::size_t capacity = kCapacity, std::size_t blockSize = kBlockSize)
    {
        CacheConfig cfg;
        cfg.uniqueId = uniqueId;
        cfg.llmRole = role;
        cfg.nWorker = 1;
        cfg.deviceId = 0;
        cfg.blockSize = blockSize;
        cfg.capacityGb = 1;
        cfg.blockTotalNumber = capacity;
        cfg.replacementPolicy = "clock";
        cfg.forceHugePage = false;
        return cfg;
    }

private:
    std::vector<std::string> usedIds_;
};

TEST_F(BootstrapTest, ElectionYieldsExactlyOneWinner)
{
    const std::string id = MakeUniqueId("elect");
    const CacheConfig cfg = MakeConfig("worker", id);

    int reportPipe[2];
    int releasePipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    ASSERT_EQ(::pipe(releasePipe), 0);

    std::vector<pid_t> children;
    for (int i = 0; i < 2; ++i) {
        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            InstallWatchdog();
            ::close(reportPipe[0]);
            ::close(releasePipe[1]);
            auto ticket = LeadershipTicket::Contest(cfg);
            const bool won = ticket.HasValue() && ticket.Value().Won();
            const char byte = won ? 1 : 0;
            (void)WriteAll(reportPipe[1], &byte, 1);
            char release = 0;
            (void)ReadAll(releasePipe[0], &release, 1);
            ::_exit(0);
        }
        children.push_back(pid);
    }
    ::close(reportPipe[1]);
    ::close(releasePipe[0]);

    int wins = 0;
    for (int i = 0; i < 2; ++i) {
        char byte = 0;
        ASSERT_TRUE(ReadAll(reportPipe[0], &byte, 1));
        wins += byte;
    }
    EXPECT_EQ(wins, 1);

    ASSERT_TRUE(WriteAll(releasePipe[1], "rr", 2));
    ::close(releasePipe[1]);
    ::close(reportPipe[0]);
    for (pid_t pid : children) {
        int status = 0;
        ASSERT_EQ(::waitpid(pid, &status, 0), pid);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
}

TEST_F(BootstrapTest, SchedulerNeverContestsLeadership)
{
    const std::string id = MakeUniqueId("sched");
    const CacheConfig cfg = MakeConfig("scheduler", id);

    auto ticket = LeadershipTicket::Contest(cfg);
    ASSERT_TRUE(ticket.HasValue()) << ticket.Error().ToString();
    EXPECT_FALSE(ticket.Value().Won());
    EXPECT_FALSE(Exists(LockPathOf(id)));
}

TEST_F(BootstrapTest, LeaderAndJoinerSeeEachOthersWrites)
{
    const std::string id = MakeUniqueId("hs");
    const CacheConfig cfg = MakeConfig("worker", id);

    int reportPipe[2];
    int releasePipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    ASSERT_EQ(::pipe(releasePipe), 0);

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        ::close(releasePipe[1]);
        auto ticket = LeadershipTicket::Contest(cfg);
        if (!ticket.HasValue() || !ticket.Value().Won()) {
            (void)WriteReport(reportPipe[1], {0, 0});
            ::_exit(1);
        }
        auto leader = BootstrapLeader::Create(cfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        *static_cast<std::uint64_t*>(SlotAddress(leader.Value().Mapping().Base(), 0, kBlockSize)) =
            kSlotMagic;
        (void)WriteReport(reportPipe[1], {3, 0});

        auto outcome = leader.Value().AcceptOne();
        if (!outcome.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, outcome.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1],
                          {outcome.Value().joined ? 1 : 2,
                           outcome.Value().joined ? 0 : StatusCode(outcome.Value().rejection)});
        if (!outcome.Value().joined) { ::_exit(0); }

        char release = 0;
        (void)ReadAll(releasePipe[0], &release, 1);
        const auto* slot1 = static_cast<const std::uint8_t*>(
            SlotAddress(leader.Value().Mapping().Base(), 1, kBlockSize));
        const bool match = slot1[0] == 0x5C && slot1[kBlockSize - 1] == 0x5C;
        (void)WriteReport(reportPipe[1], {match ? 3 : 4, 0});
        ::_exit(0);
    }
    ::close(reportPipe[1]);
    ::close(releasePipe[0]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3) << "leader bootstrap failed";

    auto joined = JoinDomain(cfg);
    ASSERT_TRUE(joined.HasValue()) << joined.Error().ToString();
    EXPECT_NE(joined.Value().mapping.Base(), nullptr);
    EXPECT_EQ(joined.Value().mapping.Size(), kBlockSize * kCapacity);
    EXPECT_TRUE(joined.Value().connection.IsValid());

    // Leader-to-joiner visibility: the magic the leader wrote before the
    // handshake is already in the mapping the joiner received.
    EXPECT_EQ(
        *static_cast<std::uint64_t*>(SlotAddress(joined.Value().mapping.Base(), 0, kBlockSize)),
        kSlotMagic);

    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 1);

    // Joiner-to-leader visibility: the leader sees what the joiner writes.
    std::memset(SlotAddress(joined.Value().mapping.Base(), 1, kBlockSize), 0x5C, kBlockSize);
    ASSERT_TRUE(WriteAll(releasePipe[1], "g", 1));

    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 3);

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(reportPipe[0]);
    ::close(releasePipe[1]);
}

TEST_F(BootstrapTest, DigestMismatchIsRejectedOnBothSides)
{
    const std::string id = MakeUniqueId("digest");
    const CacheConfig leaderCfg = MakeConfig("worker", id);
    const CacheConfig joinerCfg = MakeConfig("worker", id, kCapacity, kBlockSize * 2);

    int reportPipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        auto ticket = LeadershipTicket::Contest(leaderCfg);
        auto leader = BootstrapLeader::Create(leaderCfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1], {3, 0});
        auto outcome = leader.Value().AcceptOne();
        if (!outcome.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, outcome.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1],
                          {outcome.Value().joined ? 1 : 2,
                           outcome.Value().joined ? 0 : StatusCode(outcome.Value().rejection)});
        ::_exit(0);
    }
    ::close(reportPipe[1]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3);

    auto joined = AttemptJoinOnce(joinerCfg);
    ASSERT_FALSE(joined.HasValue());
    EXPECT_EQ(joined.Error(), Status::InvalidParam());

    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 2);
    EXPECT_EQ(report.code, Status::InvalidParam().Underlying());

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(reportPipe[0]);
}

TEST_F(BootstrapTest, SchedulerJoinsWithoutMemfdAndSecondSchedulerIsRejected)
{
    const std::string id = MakeUniqueId("ones");
    const CacheConfig leaderCfg = MakeConfig("worker", id);
    const CacheConfig schedulerCfg = MakeConfig("scheduler", id);

    int reportPipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        auto ticket = LeadershipTicket::Contest(leaderCfg);
        auto leader = BootstrapLeader::Create(leaderCfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1], {3, 0});
        for (int i = 0; i < 2; ++i) {
            auto outcome = leader.Value().AcceptOne();
            if (!outcome.HasValue()) {
                (void)WriteReport(reportPipe[1], {0, outcome.Error().Underlying()});
                ::_exit(1);
            }
            (void)WriteReport(reportPipe[1],
                              {outcome.Value().joined ? 1 : 2,
                               outcome.Value().joined ? 0 : StatusCode(outcome.Value().rejection)});
        }
        ::_exit(0);
    }
    ::close(reportPipe[1]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3);

    auto first = AttemptJoinOnce(schedulerCfg);
    ASSERT_TRUE(first.HasValue()) << first.Error().ToString();
    // The scheduler holds no shared memory by design.
    EXPECT_EQ(first.Value().mapping.Base(), nullptr);
    EXPECT_EQ(first.Value().mapping.Size(), std::size_t{0});
    EXPECT_TRUE(first.Value().connection.IsValid());

    auto second = AttemptJoinOnce(schedulerCfg);
    ASSERT_FALSE(second.HasValue());
    EXPECT_EQ(second.Error(), Status::DuplicateKey());

    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 1);
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 2);
    EXPECT_EQ(report.code, Status::DuplicateKey().Underlying());

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(reportPipe[0]);
}

TEST_F(BootstrapTest, BothLedDomainRejectsWorkerAndSchedulerRoles)
{
    const std::string id = MakeUniqueId("nw");
    const CacheConfig leaderCfg = MakeConfig("both", id);
    const CacheConfig workerCfg = MakeConfig("worker", id);
    const CacheConfig schedulerCfg = MakeConfig("scheduler", id);

    int reportPipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        auto ticket = LeadershipTicket::Contest(leaderCfg);
        auto leader = BootstrapLeader::Create(leaderCfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1], {3, 0});
        for (int i = 0; i < 2; ++i) {
            auto outcome = leader.Value().AcceptOne();
            if (!outcome.HasValue()) {
                (void)WriteReport(reportPipe[1], {0, outcome.Error().Underlying()});
                ::_exit(1);
            }
            (void)WriteReport(reportPipe[1],
                              {outcome.Value().joined ? 1 : 2,
                               outcome.Value().joined ? 0 : StatusCode(outcome.Value().rejection)});
        }
        ::_exit(0);
    }
    ::close(reportPipe[1]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3);

    auto asWorker = AttemptJoinOnce(workerCfg);
    ASSERT_FALSE(asWorker.HasValue());
    EXPECT_EQ(asWorker.Error(), Status::InvalidParam());

    auto asScheduler = AttemptJoinOnce(schedulerCfg);
    ASSERT_FALSE(asScheduler.HasValue());
    EXPECT_EQ(asScheduler.Error(), Status::InvalidParam());

    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(ReadReport(reportPipe[0], report));
        EXPECT_EQ(report.kind, 2);
        EXPECT_EQ(report.code, Status::InvalidParam().Underlying());
    }

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(reportPipe[0]);
}

TEST_F(BootstrapTest, StaleArtifactsAreReclaimedSelectively)
{
    const std::string mine = MakeUniqueId("gcmine");
    const std::string dead = MakeUniqueId("gcdead");
    const std::string live = MakeUniqueId("gclive");
    const std::string fresh = MakeUniqueId("gcfresh");

    // Dead domain: nobody holds the lock and it is older than the grace.
    TouchFile(LockPathOf(dead));
    TouchFile(SocketPathOf(dead));
    Backdate(LockPathOf(dead), 10LL * 1000000000LL);

    // Fresh domain: nobody holds the lock but it is younger than the grace.
    TouchFile(LockPathOf(fresh));
    TouchFile(SocketPathOf(fresh));

    // Live domain: a child holds the flock, however old the file looks.
    TouchFile(LockPathOf(live));
    TouchFile(SocketPathOf(live));
    int livePipe[2];
    int liveRelease[2];
    ASSERT_EQ(::pipe(livePipe), 0);
    ASSERT_EQ(::pipe(liveRelease), 0);
    const pid_t holder = ::fork();
    ASSERT_GE(holder, 0);
    if (holder == 0) {
        InstallWatchdog();
        ::close(livePipe[0]);
        ::close(liveRelease[1]);
        const int fd = ::open(LockPathOf(live).c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0 || ::flock(fd, LOCK_EX) != 0) { ::_exit(1); }
        (void)WriteReport(livePipe[1], {3, 0});
        char release = 0;
        (void)ReadAll(liveRelease[0], &release, 1);
        ::close(fd);
        ::_exit(0);
    }
    ::close(livePipe[1]);
    ::close(liveRelease[0]);
    ChildReport holderReport;
    ASSERT_TRUE(ReadReport(livePipe[0], holderReport));
    ASSERT_EQ(holderReport.kind, 3);
    Backdate(LockPathOf(live), 10LL * 1000000000LL);

    // A stale socket of the leader's own domain must not block the bind.
    TouchFile(SocketPathOf(mine));

    const CacheConfig cfg = MakeConfig("worker", mine);
    auto ticket = LeadershipTicket::Contest(cfg);
    ASSERT_TRUE(ticket.HasValue()) << ticket.Error().ToString();
    ASSERT_TRUE(ticket.Value().Won());
    auto leader = BootstrapLeader::Create(cfg, std::move(ticket.Value()));
    ASSERT_TRUE(leader.HasValue()) << leader.Error().ToString();

    EXPECT_FALSE(Exists(LockPathOf(dead)));
    EXPECT_FALSE(Exists(SocketPathOf(dead)));
    EXPECT_TRUE(Exists(LockPathOf(live)));
    EXPECT_TRUE(Exists(SocketPathOf(live)));
    EXPECT_TRUE(Exists(LockPathOf(fresh)));
    EXPECT_TRUE(Exists(SocketPathOf(fresh)));
    EXPECT_TRUE(Exists(LockPathOf(mine)));
    EXPECT_TRUE(Exists(SocketPathOf(mine)));

    ASSERT_TRUE(WriteAll(liveRelease[1], "g", 1));
    ::close(liveRelease[1]);
    ::close(livePipe[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(holder, &status, 0), holder);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

TEST_F(BootstrapTest, JoinDomainRetriesUntilTheLeaderAppears)
{
    const std::string id = MakeUniqueId("retry");
    const CacheConfig cfg = MakeConfig("worker", id);

    int reportPipe[2];
    int releasePipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    ASSERT_EQ(::pipe(releasePipe), 0);

    // The leader appears late; the joiner must retry its way through it.
    const pid_t leaderPid = ::fork();
    ASSERT_GE(leaderPid, 0);
    if (leaderPid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        ::close(releasePipe[1]);
        ::timespec delay{0, 150 * 1000000};
        ::nanosleep(&delay, nullptr);
        auto ticket = LeadershipTicket::Contest(cfg);
        auto leader = BootstrapLeader::Create(cfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        auto outcome = leader.Value().AcceptOne();
        if (!outcome.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, outcome.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1],
                          {outcome.Value().joined ? 1 : 2,
                           outcome.Value().joined ? 0 : StatusCode(outcome.Value().rejection)});
        char release = 0;
        (void)ReadAll(releasePipe[0], &release, 1);
        ::_exit(0);
    }

    int joinReportPipe[2];
    ASSERT_EQ(::pipe(joinReportPipe), 0);
    const pid_t joinerPid = ::fork();
    ASSERT_GE(joinerPid, 0);
    if (joinerPid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        ::close(reportPipe[1]);
        ::close(releasePipe[0]);
        ::close(releasePipe[1]);
        ::close(joinReportPipe[0]);
        auto joined = JoinDomain(cfg);
        if (!joined.HasValue()) {
            (void)WriteReport(joinReportPipe[1], {0, joined.Error().Underlying()});
            ::_exit(1);
        }
        const bool mapped = joined.Value().mapping.Base() != nullptr;
        (void)WriteReport(joinReportPipe[1], {mapped ? 3 : 4, 0});
        ::_exit(0);
    }
    ::close(reportPipe[1]);
    ::close(releasePipe[0]);
    ::close(joinReportPipe[1]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(joinReportPipe[0], report));
    EXPECT_EQ(report.kind, 3);
    int status = 0;
    ASSERT_EQ(::waitpid(joinerPid, &status, 0), joinerPid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(joinReportPipe[0]);

    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    EXPECT_EQ(report.kind, 1);
    ASSERT_TRUE(WriteAll(releasePipe[1], "g", 1));
    ASSERT_EQ(::waitpid(leaderPid, &status, 0), leaderPid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ::close(reportPipe[0]);
    ::close(releasePipe[1]);
}

TEST_F(BootstrapTest, ServeLoopKeepsAcceptingPeers)
{
    const std::string id = MakeUniqueId("serve");
    const CacheConfig cfg = MakeConfig("worker", id);

    int reportPipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        auto ticket = LeadershipTicket::Contest(cfg);
        auto leader = BootstrapLeader::Create(cfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1], {3, 0});
        // Blocks for the process lifetime; peers stay connected until then.
        (void)leader.Value().Serve();
        ::_exit(0);
    }
    ::close(reportPipe[1]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3);

    auto first = JoinDomain(cfg);
    ASSERT_TRUE(first.HasValue()) << first.Error().ToString();
    auto second = JoinDomain(cfg);
    ASSERT_TRUE(second.HasValue()) << second.Error().ToString();
    EXPECT_NE(first.Value().mapping.Base(), nullptr);
    EXPECT_NE(second.Value().mapping.Base(), nullptr);

    ASSERT_EQ(::kill(pid, SIGTERM), 0);
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM);
    ::close(reportPipe[0]);
}

TEST_F(BootstrapTest, LeaderKeepsTheFlockForItsLifetime)
{
    const std::string id = MakeUniqueId("flockkeep");
    const CacheConfig cfg = MakeConfig("both", id);

    int reportPipe[2];
    int releasePipe[2];
    ASSERT_EQ(::pipe(reportPipe), 0);
    ASSERT_EQ(::pipe(releasePipe), 0);

    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        InstallWatchdog();
        ::close(reportPipe[0]);
        ::close(releasePipe[1]);
        auto ticket = LeadershipTicket::Contest(cfg);
        if (!ticket.HasValue() || !ticket.Value().Won()) {
            (void)WriteReport(reportPipe[1], {0, 0});
            ::_exit(1);
        }
        auto leader = BootstrapLeader::Create(cfg, std::move(ticket.Value()));
        if (!leader.HasValue()) {
            (void)WriteReport(reportPipe[1], {0, leader.Error().Underlying()});
            ::_exit(1);
        }
        (void)WriteReport(reportPipe[1], {3, 0});
        char release = 0;
        (void)ReadAll(releasePipe[0], &release, 1);
        ::_exit(0);
    }
    ::close(reportPipe[1]);
    ::close(releasePipe[0]);

    ChildReport report;
    ASSERT_TRUE(ReadReport(reportPipe[0], report));
    ASSERT_EQ(report.kind, 3);

    /*
     * The leader keeps the winning ticket -- and with it the flock -- for
     * its lifetime: a contestant of the same domain must lose while the
     * leader lives. Regression: Create used to drop the ticket on return,
     * releasing the flock, so a long-lived domain could be garbage-collected
     * by the next domain's leader after the grace period.
     */
    auto ticket = LeadershipTicket::Contest(cfg);
    ASSERT_TRUE(ticket.HasValue()) << ticket.Error().ToString();
    EXPECT_FALSE(ticket.Value().Won());

    ASSERT_TRUE(WriteAll(releasePipe[1], "r", 1));
    ::close(releasePipe[1]);
    ::close(reportPipe[0]);
    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

}  // namespace
}  // namespace UC::Store
