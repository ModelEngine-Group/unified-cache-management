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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <gtest/gtest.h>
#include <initializer_list>
#include <optional>
#include <string>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>
#include "cache/cache_store.h"
#include "cache/shm_layout.h"
#include "status/status.h"
#include "type/block.h"
#include "type/dictionary.h"

namespace UC::Store {

/*
 * Rendezvous path helpers are defined in bootstrap.cc with external linkage
 * but declared in no header; the signatures must stay identical (mismatch
 * turns into a link error, never silent drift).
 */
[[nodiscard]] std::string LockPathOf(const std::string& uniqueId);
[[nodiscard]] std::string SocketPathOf(const std::string& uniqueId);

namespace {

/*
 * Block geometries of the test domains. The configuration computes the slot
 * count as (capacityGb << 30) / blockSize with capacityGb pinned to its
 * minimum of 1, so a small pool demands a large block; every domain below
 * is one gigabyte in total.
 */
constexpr std::size_t kSlots4BlockSize = 268435456;  // exactly 4 slots
constexpr std::size_t kSlots5BlockSize = 214748364;  // exactly 5 slots
constexpr std::size_t kSlots6BlockSize = 178956970;  // exactly 6 slots
constexpr std::size_t kSmallBlockSize = 4194304;     // exactly 256 slots, for data transfers

/** Fixed protocol cap: one batch of ids in a command. */
constexpr std::size_t kMaxBatch = 16;

/** Fixed-size command the parent sends to a store child. */
struct Cmd {
    std::int32_t op{0};
    std::int32_t n{0};
    /** Request TTL fact in milliseconds; 0 = no fact. */
    std::int32_t ttlMs{0};
    /** Commit's success flag; 0 = false. */
    std::int32_t success{0};
    /** Data-plane job id, and the token's slot numbers in block order. */
    std::uint32_t jobId{0};
    std::uint32_t slots[kMaxBatch]{};
    BlockId ids[kMaxBatch]{};
};

/** Fixed-size result a store child answers with. */
struct Result {
    /** Status::Underlying() of the call; 0 = Ok. */
    std::int32_t code{0};
    /** Hit count for Lookup, slot count for Pin/Alloc. */
    std::int32_t n{0};
    std::int32_t hits[kMaxBatch]{};
    std::uint32_t slots[kMaxBatch]{};
};

static_assert(std::is_trivially_copyable_v<Cmd> && std::is_trivially_copyable_v<Result>);
static_assert(std::has_unique_object_representations_v<Result>);

/** Command opcodes of the driving protocol. */
enum : std::int32_t {
    kOpLookup = 1,
    kOpTouch,
    kOpPin,
    kOpUnpin,
    kOpAlloc,
    kOpCommit,
    kOpSleepMs,
    kOpDump,
    kOpLoad,
    kOpWait,
    kOpGetFinished,
    kOpVerify,
};

/** What a forked child tells its parent: 3 = ready, 4 = self-check failed,
 *  0 = fatal; code is a status code or 0. */
struct ChildReport {
    std::int32_t kind{0};
    std::int32_t code{0};
};

/** One forked store child as seen by the parent. */
struct Child {
    pid_t pid{0};
    int cmdW{-1};    /* parent writes Cmds */
    int resultR{-1}; /* parent reads Results */
    int reportR{-1}; /* parent reads ChildReports */
};

BlockId MakeBlock(std::uint64_t tag)
{
    BlockId id;
    std::memcpy(id.bytes.data(), "blk", 3);
    std::memcpy(id.bytes.data() + 8, &tag, sizeof(tag));
    return id;
}

Cmd MakeCmd(std::int32_t op, std::initializer_list<std::uint64_t> tags, std::int32_t ttlMs = 0,
            std::int32_t success = 0)
{
    Cmd cmd;
    cmd.op = op;
    cmd.n = static_cast<std::int32_t>(tags.size());
    std::size_t i = 0;
    for (const std::uint64_t tag : tags) { cmd.ids[i++] = MakeBlock(tag); }
    cmd.ttlMs = ttlMs;
    cmd.success = success;
    return cmd;
}

/** A sleep command: the child pauses for n milliseconds, then answers. */
Cmd MakeSleep(std::int32_t milliseconds)
{
    Cmd cmd;
    cmd.op = kOpSleepMs;
    cmd.n = milliseconds;
    return cmd;
}

/** A transfer command: job id, blocks, and the token's slot numbers in the
 *  minter's order. */
Cmd MakeTransferCmd(std::int32_t op, std::uint32_t jobId, std::initializer_list<std::uint64_t> tags,
                    std::initializer_list<std::uint32_t> slots)
{
    Cmd cmd;
    cmd.op = op;
    cmd.jobId = jobId;
    cmd.n = static_cast<std::int32_t>(tags.size());
    std::size_t i = 0;
    for (const std::uint64_t tag : tags) { cmd.ids[i++] = MakeBlock(tag); }
    i = 0;
    for (const std::uint32_t slot : slots) { cmd.slots[i++] = slot; }
    return cmd;
}

/** A job-lifecycle command: Wait or GetFinished, on one job id. */
Cmd MakeJobCmd(std::int32_t op, std::uint32_t jobId)
{
    Cmd cmd;
    cmd.op = op;
    cmd.jobId = jobId;
    return cmd;
}

/** Fills len bytes with the block's deterministic pattern: a byte stream
 *  out of an LCG seeded from the first eight bytes of the block's id. */
void FillPattern(const BlockId& id, void* out, std::size_t len)
{
    std::uint64_t state = 0;
    std::memcpy(&state, id.bytes.data(), sizeof(state));
    auto* cursor = static_cast<std::byte*>(out);
    for (std::size_t i = 0; i < len; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        std::memcpy(cursor + i, &state, 1);
    }
}

/** Tells whether the buffer still carries the block's pattern. */
bool MatchesPattern(const BlockId& id, const void* in, std::size_t len)
{
    std::vector<std::byte> expected(len);
    FillPattern(id, expected.data(), len);
    return std::memcmp(expected.data(), in, len) == 0;
}

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
 *  instead of hanging the suite. Children never run gtest assertions. */
void InstallWatchdog()
{
    struct ::sigaction action{};
    action.sa_handler = [](int) { ::_exit(99); };
    ::sigemptyset(&action.sa_mask);
    ::sigaction(SIGALRM, &action, nullptr);
    ::alarm(30);
}

/** One command executed against the child's CacheStore. The transfer buffer
 *  persists across commands -- the store's transfers are asynchronous, so
 *  pattern verification belongs to kOpVerify, after the completion. */
Result RunOne(CacheStore& store, const Cmd& cmd, std::size_t blockSize,
              std::vector<std::byte>& buffer)
{
    Result result{};
    std::size_t count = 0;
    if (cmd.n > 0) {
        count = static_cast<std::size_t>(cmd.n);
        if (count > kMaxBatch) { count = kMaxBatch; }
    }
    std::vector<BlockId> ids(cmd.ids, cmd.ids + count);

    RequestContext context;
    if (cmd.ttlMs != 0) { context.ttl = std::chrono::milliseconds{cmd.ttlMs}; }

    switch (cmd.op) {
        case kOpLookup: {
            auto outcome = store.Lookup(ids);
            if (!outcome.HasValue()) {
                result.code = outcome.Error().Underlying();
                return result;
            }
            result.n = cmd.n;
            for (std::size_t i = 0; i < count && i < outcome.Value().size(); i++) {
                result.hits[i] = outcome.Value()[i] != 0 ? 1 : 0;
            }
            return result;
        }
        case kOpTouch: {
            store.Touch(ids, context);
            return result;
        }
        case kOpUnpin: {
            store.Unpin(ids);
            return result;
        }
        case kOpCommit: {
            store.Commit(ids, cmd.success != 0);
            return result;
        }
        case kOpPin:
        case kOpAlloc: {
            auto outcome = cmd.op == kOpPin ? store.Pin(ids) : store.Alloc(ids, context);
            if (!outcome.HasValue()) {
                result.code = outcome.Error().Underlying();
                return result;
            }
            const auto slots = DecodeTokenPayload(outcome.Value().payload);
            result.n = cmd.n;
            for (std::size_t i = 0; i < slots.size() && i < kMaxBatch; i++) {
                result.slots[i] = slots[i];
            }
            return result;
        }
        case kOpDump:
        case kOpLoad: {
            /* Whole-slot, single-pointer payloads; the token is rebuilt from
             * the command's slot numbers, standing in for the deployment's
             * token broadcast. The buffer persists across commands: Dump
             * fills the patterns, Load poisons them, and neither inspects
             * the result -- the transfer is asynchronous, so verification
             * belongs to kOpVerify after the completion. */
            const std::size_t bytes = count * blockSize;
            if (buffer.size() < bytes) { buffer.resize(bytes); }
            if (cmd.op == kOpDump) {
                for (std::size_t i = 0; i < count; i++) {
                    FillPattern(ids[i], buffer.data() + i * blockSize, blockSize);
                }
            } else if (bytes > 0) {
                std::memset(buffer.data(), 0xA5, bytes);
            }
            JobDescriptor job;
            job.blockIds = ids;
            job.payloads.resize(count);
            for (std::size_t i = 0; i < count; i++) {
                BlockPayload& payload = job.payloads[i];
                payload.pointers.push_back(buffer.data() + i * blockSize);
                payload.sizes.push_back(blockSize);
                payload.offsets.push_back(0);
            }
            AddressedToken token;
            token.keys = ids;
            token.payload =
                EncodeTokenPayload(std::vector<std::uint32_t>(cmd.slots, cmd.slots + count));
            const Status st = cmd.op == kOpDump
                                  ? store.Dump(cmd.jobId, std::move(job), std::move(token))
                                  : store.Load(cmd.jobId, std::move(job), std::move(token));
            result.code = st.Underlying();
            if (st.Failure()) { return result; }
            result.n = cmd.n;
            return result;
        }
        case kOpVerify: {
            /* Pattern check of the persistent buffer's blocks: run after
             * the completion of the transfer that filled it. */
            result.n = cmd.n;
            for (std::size_t i = 0; i < count && (i + 1) * blockSize <= buffer.size(); i++) {
                result.hits[i] =
                    MatchesPattern(ids[i], buffer.data() + i * blockSize, blockSize) ? 1 : 0;
            }
            return result;
        }
        case kOpWait: {
            result.code = store.Wait(cmd.jobId).Underlying();
            return result;
        }
        case kOpGetFinished: {
            const std::vector<std::pair<std::size_t, bool>> finished = store.GetFinished();
            result.n = static_cast<std::int32_t>(finished.size());
            for (std::size_t i = 0; i < finished.size() && i < kMaxBatch; i++) {
                result.slots[i] = static_cast<std::uint32_t>(finished[i].first);
                result.hits[i] = finished[i].second ? 1 : 0;
            }
            return result;
        }
        case kOpSleepMs: {
            ::timespec pause{};
            pause.tv_sec = cmd.n / 1000;
            pause.tv_nsec = static_cast<long>(cmd.n % 1000) * 1000000L;
            ::nanosleep(&pause, nullptr);
            return result;
        }
        default: {
            result.code = Status::InvalidParam().Underlying();
            return result;
        }
    }
}

/**
 * The body of one forked store child: set the store up from the dictionary,
 * report readiness, then serve the parent's commands until the command pipe
 * closes. The role and block geometry are read from the dictionary -- the
 * same source the store itself was configured from. Returning lets the
 * CacheStore destructor run before the caller terminates the process.
 */
void ServeStoreChild(const Dictionary& dict, int cmdFd, int resultFd, int reportFd)
{
    ::alarm(30);
    std::string role;
    std::int64_t blockSize = 0;
    if (!dict.Get("llm_role", role) || !dict.GetInt64("block_size", blockSize) || blockSize <= 0) {
        (void)WriteReport(reportFd, {0, 0});
        ::_exit(1);
    }
    CacheStore store;
    const Status setup = store.Setup(dict);
    if (setup.Failure()) {
        (void)WriteReport(reportFd, {0, setup.Underlying()});
        ::_exit(1);
    }
    /* Out-of-role stubs, checked while the store is live. */
    if (role == "worker") {
        /* A worker serves no metadata: the value-returning calls refuse. */
        const std::vector<BlockId> probe{MakeBlock(1)};
        const auto lookup = store.Lookup(probe);
        const auto pin = store.Pin(probe);
        const auto alloc = store.Alloc(probe, RequestContext{});
        if (lookup.HasValue() || pin.HasValue() || alloc.HasValue() ||
            lookup.Error().Underlying() != Status::Unsupported().Underlying() ||
            pin.Error().Underlying() != Status::Unsupported().Underlying() ||
            alloc.Error().Underlying() != Status::Unsupported().Underlying()) {
            (void)WriteReport(reportFd, {4, 0});
            ::_exit(1);
        }
    } else if (role == "scheduler") {
        /* The scheduler runs no data plane: transfers refuse, the completion
         * log stays empty, unknown jobs are invalid. */
        const Status wait = store.Wait(1);
        const Status load = store.Load(1, JobDescriptor{}, AddressedToken{});
        const Status dump = store.Dump(1, JobDescriptor{}, AddressedToken{});
        if (wait.Underlying() != Status::InvalidParam().Underlying() ||
            load.Underlying() != Status::Unsupported().Underlying() ||
            dump.Underlying() != Status::Unsupported().Underlying() ||
            !store.GetFinished().empty()) {
            (void)WriteReport(reportFd, {4, 0});
            ::_exit(1);
        }
    }
    (void)WriteReport(reportFd, {3, 0});

    std::vector<std::byte> transferBuffer;
    for (;;) {
        ::alarm(30);
        Cmd cmd;
        if (!ReadAll(cmdFd, &cmd, sizeof(cmd))) { break; }
        const Result result =
            RunOne(store, cmd, static_cast<std::size_t>(blockSize), transferBuffer);
        if (!WriteAll(resultFd, &result, sizeof(result))) { break; }
    }
}

/**
 * The determinism script of test A: admission, conclusion, visibility,
 * pinning, eviction, re-admission and the two error paths, with no TTL fact
 * anywhere -- TTL couples block retention to real time, so a TTL-free
 * script is strictly deterministic and two runs must agree bit for bit.
 */
std::vector<Cmd> BuildDeterminismScript()
{
    std::vector<Cmd> script;
    script.push_back(MakeCmd(kOpLookup, {1})); /* miss */
    script.push_back(MakeCmd(kOpAlloc, {1, 2, 3}));
    script.push_back(MakeCmd(kOpCommit, {1, 2, 3}, 0, 1));
    script.push_back(MakeCmd(kOpLookup, {1, 2, 3})); /* all hit */
    script.push_back(MakeCmd(kOpAlloc, {4, 5, 6}));
    script.push_back(MakeCmd(kOpCommit, {4, 5, 6}, 0, 0)); /* release */
    script.push_back(MakeCmd(kOpLookup, {4, 5, 6}));       /* all miss */
    script.push_back(MakeCmd(kOpAlloc, {4, 5, 6}));        /* re-admission */
    script.push_back(MakeCmd(kOpCommit, {4, 5, 6}, 0, 1));
    script.push_back(MakeCmd(kOpLookup, {1, 2, 3, 4, 5, 6})); /* all hit */
    script.push_back(MakeCmd(kOpPin, {1, 2}));
    script.push_back(MakeCmd(kOpAlloc, {7})); /* evicts 3 */
    script.push_back(MakeCmd(kOpCommit, {7}, 0, 1));
    script.push_back(MakeCmd(kOpLookup, {1, 2, 3, 7})); /* 1,1,0,1 */
    script.push_back(MakeCmd(kOpTouch, {1, 2}));
    script.push_back(MakeCmd(kOpUnpin, {1, 2}));
    script.push_back(MakeCmd(kOpAlloc, {8, 9})); /* evicts 1, 2 */
    script.push_back(MakeCmd(kOpCommit, {8, 9}, 0, 1));
    script.push_back(MakeCmd(kOpLookup, {1, 2, 3, 4, 5, 6, 7, 8, 9})); /* 0,0,0,1,1,1,1,1,1 */
    script.push_back(MakeCmd(kOpAlloc, {1}));                          /* re-admission */
    script.push_back(MakeCmd(kOpCommit, {1}, 0, 1));
    script.push_back(MakeCmd(kOpLookup, {1}));                         /* hit again */
    script.push_back(MakeCmd(kOpAlloc, {10, 11, 12, 13, 14, 15, 16})); /* NoSpace */
    script.push_back(MakeCmd(kOpLookup, {4}));                         /* miss: 4 fell above */
    script.push_back(MakeCmd(kOpLookup, {5}));                         /* hit: batch rolled back */
    script.push_back(MakeCmd(kOpPin, {99}));                           /* NotFound */
    script.push_back(MakeCmd(kOpUnpin, {99}));                         /* no-op */
    return script;
}

class MultiprocessTest : public ::testing::Test {
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

    Dictionary MakeDict(const std::string& uniqueId, const std::string& role, std::size_t blockSize,
                        std::int64_t workerNumber = 2, const std::string& dmaEngine = {})
    {
        Dictionary dict;
        dict.Set("unique_id", uniqueId);
        dict.Set("llm_role", role);
        dict.SetInt64("worker_number", workerNumber);
        dict.SetInt64("device_id", 0);
        dict.SetInt64("block_size", static_cast<std::int64_t>(blockSize));
        dict.SetInt64("cache_buffer_capacity_gb", 1);
        dict.Set("cache_replacement_policy", std::string{"fifo"});
        dict.Set("cache_force_huge_page", false);
        if (!dmaEngine.empty()) { dict.Set("cache_dma_engine", dmaEngine); }
        return dict;
    }

    /**
     * Forks one store child. The winner determinism is the fork order: fork
     * the leadership candidate first and wait for its readiness before
     * forking anyone else -- it holds the domain's leadership by then, so
     * every later worker joins as a leaf and every later both-child joins as
     * a replica.
     */
    void ForkStoreChild(Child& child, const Dictionary& dict)
    {
        int cmdPipe[2];
        int resultPipe[2];
        int reportPipe[2];
        ASSERT_EQ(::pipe(cmdPipe), 0);
        ASSERT_EQ(::pipe(resultPipe), 0);
        ASSERT_EQ(::pipe(reportPipe), 0);

        const pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            InstallWatchdog();
            ::close(cmdPipe[1]);
            ::close(resultPipe[0]);
            ::close(reportPipe[0]);
            ServeStoreChild(dict, cmdPipe[0], resultPipe[1], reportPipe[1]);
            ::_exit(0);
        }
        ::close(cmdPipe[0]);
        ::close(resultPipe[1]);
        ::close(reportPipe[1]);
        child = Child{pid, cmdPipe[1], resultPipe[0], reportPipe[0]};
    }

    void ExpectReady(const Child& child)
    {
        ChildReport report;
        ASSERT_TRUE(ReadReport(child.reportR, report));
        ASSERT_EQ(report.kind, 3) << "child did not come up (kind " << report.code << ")";
    }

    /** Sends one command and waits for its result. */
    Result Drive(const Child& child, const Cmd& cmd)
    {
        EXPECT_TRUE(WriteAll(child.cmdW, &cmd, sizeof(cmd)));
        Result result{};
        if (!ReadAll(child.resultR, &result, sizeof(result))) {
            ADD_FAILURE() << "child " << child.pid << " stopped answering";
        }
        return result;
    }

    /** Polls one child's completion log until the given job reports: the
     *  store's transfers are asynchronous, so a submit returns with the
     *  job in flight and the completion lands on the engine's schedule.
     *  Returns nullopt after a generous deadline. */
    std::optional<std::pair<std::uint32_t, bool>> PollFinished(const Child& child,
                                                               std::uint32_t jobId)
    {
        for (int attempt = 0; attempt < 5000; ++attempt) {
            const Result result = Drive(child, MakeJobCmd(kOpGetFinished, 0));
            const std::size_t reported =
                result.n > 0 ? static_cast<std::size_t>(result.n) : std::size_t{0};
            for (std::size_t i = 0; i < reported && i < kMaxBatch; i++) {
                if (result.slots[i] == jobId) {
                    return std::make_pair(result.slots[i], result.hits[i] != 0);
                }
            }
            ::usleep(1000);
        }
        return std::nullopt;
    }

    /** Orderly shutdown: closing the command pipe ends the child's loop,
     *  letting the store destructor release the domain. */
    void ReleaseChild(const Child& child)
    {
        ::close(child.cmdW);
        int status = 0;
        ASSERT_EQ(::waitpid(child.pid, &status, 0), child.pid);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        ::close(child.resultR);
        ::close(child.reportR);
    }

    void KillChild(const Child& child)
    {
        ::close(child.cmdW);
        ASSERT_EQ(::kill(child.pid, SIGKILL), 0);
        int status = 0;
        ASSERT_EQ(::waitpid(child.pid, &status, 0), child.pid);
        EXPECT_TRUE(WIFSIGNALED(status));
        ::close(child.resultR);
        ::close(child.reportR);
    }

    /** Reads the report of a child whose Setup failed: kind 0, carrying
     *  the rejected Status. */
    void ExpectSetupFailure(const Child& child)
    {
        ChildReport report;
        ASSERT_TRUE(ReadReport(child.reportR, report));
        ASSERT_EQ(report.kind, 0);
        EXPECT_EQ(report.code, Status::InvalidParam().Underlying());
    }

    /** Collects a child that failed its setup: it exits 1 on its own. */
    void ReleaseFailedChild(const Child& child)
    {
        ::close(child.cmdW);
        int status = 0;
        ASSERT_EQ(::waitpid(child.pid, &status, 0), child.pid);
        EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 1);
        ::close(child.resultR);
        ::close(child.reportR);
    }

private:
    std::vector<std::string> usedIds_;
};

/*== The facade's process-local edges ==*/

TEST_F(MultiprocessTest, SetupFailuresPropagate)
{
    Dictionary empty = MakeDict(MakeUniqueId("emptyid"), "both", kSlots4BlockSize);
    empty.Set("unique_id", std::string{});
    CacheStore store;
    const Status st = store.Setup(empty);
    ASSERT_TRUE(st.Failure());
    EXPECT_EQ(st.Underlying(), Status::InvalidParam().Underlying());

    Dictionary badPolicy = MakeDict(MakeUniqueId("badpolicy"), "both", kSlots4BlockSize);
    badPolicy.Set("cache_replacement_policy", std::string{"nope"});
    CacheStore other;
    const Status policy = other.Setup(badPolicy);
    ASSERT_TRUE(policy.Failure());
    EXPECT_EQ(policy.Underlying(), Status::NotFound().Underlying());
}

/*== A: cross-run, cross-path bit-for-bit agreement ==*/

TEST_F(MultiprocessTest, ReplicaDrivenAndLocallyDrivenRunsAgreeBitForBit)
{
    const std::vector<Cmd> script = BuildDeterminismScript();

    /* Run 1: a fresh domain, primary plus one replica, the whole script
     * driven through the replica's forwarded path. */
    const std::string drivenId = MakeUniqueId("bitrep");
    Child primary;
    Child replica;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(drivenId, "both", kSlots6BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(replica, MakeDict(drivenId, "both", kSlots6BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(replica));

    std::vector<Result> forwarded;
    forwarded.reserve(script.size());
    for (const Cmd& cmd : script) { forwarded.push_back(Drive(replica, cmd)); }

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(replica));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));

    /* Run 2: another fresh domain, a single primary, the same script driven
     * over the local path. */
    const std::string localId = MakeUniqueId("bitloc");
    Child solo;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(solo, MakeDict(localId, "both", kSlots6BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(solo));

    std::vector<Result> local;
    local.reserve(script.size());
    for (const Cmd& cmd : script) { local.push_back(Drive(solo, cmd)); }

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(solo));

    /* The core property: identical results on both paths. */
    ASSERT_EQ(forwarded.size(), local.size());
    for (std::size_t i = 0; i < script.size(); i++) {
        EXPECT_EQ(std::memcmp(&forwarded[i], &local[i], sizeof(Result)), 0)
            << "step " << i << " diverged between the forwarded and local paths";
    }

    /* Spot checks so the equality above cannot be two equally wrong runs:
     * the assertions below document the fifo-traced script behavior. */
    EXPECT_EQ(forwarded[0].code, 0);
    EXPECT_EQ(forwarded[0].hits[0], 0); /* miss */
    EXPECT_EQ(forwarded[3].hits[0] + forwarded[3].hits[1] + forwarded[3].hits[2], 3);
    EXPECT_EQ(forwarded[6].hits[0] + forwarded[6].hits[1] + forwarded[6].hits[2], 0);
    EXPECT_EQ(forwarded[9].n, 6);
    EXPECT_EQ(forwarded[13].hits[0], 1);
    EXPECT_EQ(forwarded[13].hits[1], 1);
    EXPECT_EQ(forwarded[13].hits[2], 0); /* evicted */
    EXPECT_EQ(forwarded[13].hits[3], 1);
    EXPECT_EQ(forwarded[22].code, Status::NoSpace().Underlying()); /* oversize batch */
    EXPECT_EQ(forwarded[23].hits[0], 0); /* block 4 fell to the step-19 re-admission */
    EXPECT_EQ(forwarded[24].hits[0], 1); /* rollback held */
    EXPECT_EQ(forwarded[25].code, Status::NotFound().Underlying()); /* pin of a miss */
}

/*== B: cross-replica visibility ==*/

TEST_F(MultiprocessTest, ReplicasSeeEachOthersAdmissions)
{
    const std::string id = MakeUniqueId("visibility");
    Child primary;
    Child alice;
    Child bob;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots5BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSlots5BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(bob, MakeDict(id, "both", kSlots5BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(bob));

    /* Alice fills the pool and commits; Bob sees everything. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {1, 2, 3, 4, 5})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1, 2, 3, 4, 5}, 0, 1)).code, 0);
    const Result seen = Drive(bob, MakeCmd(kOpLookup, {1, 2, 3, 4, 5}));
    EXPECT_EQ(seen.code, 0);
    EXPECT_EQ(seen.hits[0] + seen.hits[1] + seen.hits[2] + seen.hits[3] + seen.hits[4], 5);

    /* Alice's re-admission evicts block 1; Bob now sees exactly 4 of 5. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {6})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {6}, 0, 1)).code, 0);
    const Result evicted = Drive(bob, MakeCmd(kOpLookup, {1, 2, 3, 4, 5}));
    EXPECT_EQ(evicted.hits[0], 0);
    EXPECT_EQ(evicted.hits[1] + evicted.hits[2] + evicted.hits[3] + evicted.hits[4], 4);

    /* A failed conclusion stays invisible; its victim stays evicted. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {7})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {7}, 0, 0)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {7})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {7})).hits[0], 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {2})).hits[0], 0);
    const Result kept = Drive(bob, MakeCmd(kOpLookup, {3, 4, 5, 6}));
    EXPECT_EQ(kept.hits[0] + kept.hits[1] + kept.hits[2] + kept.hits[3], 4);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(bob));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== C: TTL expiry and revival are visible across replicas ==*/

TEST_F(MultiprocessTest, TtlExpiryAndRevivalAreVisibleAcrossReplicas)
{
    const std::string id = MakeUniqueId("ttl");
    Child primary;
    Child alice;
    Child bob;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(bob, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(bob));

    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {1}, 200)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 1);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 1);

    /* Past the TTL both replicas miss: only the primary reads the clock,
     * so both observe the same lazy expiry. */
    Drive(alice, MakeSleep(500));
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 0);

    /* Bob's TTL-free re-admission revives the block for the whole domain. */
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {1})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {1}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 1);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 1);

    /* Revived without a TTL fact, the block never expires again. */
    Drive(bob, MakeSleep(500));
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 1);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(bob));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== D: pins protect blocks across replicas ==*/

TEST_F(MultiprocessTest, PinsProtectBlocksAcrossReplicas)
{
    const std::string id = MakeUniqueId("pins");
    Child primary;
    Child alice;
    Child bob;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(bob, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(bob));

    /* Alice fills the pool and pins half of it. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {1, 2, 3, 4})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1, 2, 3, 4}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpPin, {1, 2})).code, 0);

    /* Bob's admissions squeeze past the pins: 3 and 4 fall, 1 and 2 hold. */
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {5})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {5}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpPin, {5})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {6})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {6}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpPin, {6})).code, 0);

    /* Every slot is pinned now: Bob's next admission fails with NoSpace,
     * forwarded and surfaced as an error Status. */
    const Result squeezed = Drive(bob, MakeCmd(kOpAlloc, {7}));
    EXPECT_EQ(squeezed.code, Status::NoSpace().Underlying());

    const Result held = Drive(bob, MakeCmd(kOpLookup, {1, 2}));
    EXPECT_EQ(held.hits[0], 1);
    EXPECT_EQ(held.hits[1], 1);

    /* Alice releases her pins; Bob's admission proceeds and takes the
     * freed ground. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpUnpin, {1, 2})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {7})).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {7}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {2})).hits[0], 1);

    /* Bob's own pins survived the shuffle. */
    const Result pinned = Drive(bob, MakeCmd(kOpLookup, {5, 6, 7}));
    EXPECT_EQ(pinned.hits[0] + pinned.hits[1] + pinned.hits[2], 3);

    EXPECT_EQ(Drive(bob, MakeCmd(kOpUnpin, {5, 6})).code, 0);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(bob));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== E: a replica survives its primary's death ==*/

TEST_F(MultiprocessTest, ReplicaSurvivesPrimaryDeath)
{
    const std::string id = MakeUniqueId("dead");
    Child primary;
    Child alice;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));

    /* Sanity: the domain works before it dies. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {1})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 1);

    /* The primary dies abruptly; the replica's next call returns an error
     * Status -- it neither hangs nor self-terminates. */
    ASSERT_NO_FATAL_FAILURE(KillChild(primary));

    const Result first = Drive(alice, MakeCmd(kOpLookup, {1}));
    EXPECT_NE(first.code, 0);

    /* Still responsive: the very next call answers as well. */
    const Result second = Drive(alice, MakeCmd(kOpLookup, {1}));
    EXPECT_NE(second.code, 0);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
}

/*== F: shared reservations across replicas ==*/

TEST_F(MultiprocessTest, SharedReservationsAcrossReplicas)
{
    const std::string id = MakeUniqueId("shared");
    Child primary;
    Child alice;
    Child bob;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(bob, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(bob));

    /* Concurrent admissions of one block share a reservation: both tokens
     * address the same slot, bit for bit. */
    const Result aliceOne = Drive(alice, MakeCmd(kOpAlloc, {1}));
    const Result bobOne = Drive(bob, MakeCmd(kOpAlloc, {1}));
    EXPECT_EQ(aliceOne.code, 0);
    EXPECT_EQ(bobOne.code, 0);
    EXPECT_EQ(bobOne.slots[0], aliceOne.slots[0]);

    /* The reservation stays invisible until its conclusion, and one
     * failed sharer vetoes it: the block never becomes visible. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1}, 0, 0)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {1}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {1})).hits[0], 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {1})).hits[0], 0);

    /* Two failed conclusions release the reservation and its slot. */
    const Result aliceTwo = Drive(alice, MakeCmd(kOpAlloc, {2}));
    const Result bobTwo = Drive(bob, MakeCmd(kOpAlloc, {2}));
    EXPECT_EQ(aliceTwo.code, 0);
    EXPECT_EQ(bobTwo.code, 0);
    EXPECT_EQ(bobTwo.slots[0], aliceTwo.slots[0]);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {2}, 0, 0)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {2}, 0, 0)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {2})).hits[0], 0);

    /* The released ground is reusable by the next admission. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {3})).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {3}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {3})).hits[0], 1);

    /* The reservation's TTL fact follows the last writer: here Bob's 100ms
     * overwrite Alice's 2000ms. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {4}, 2000)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {4}, 100)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {4}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {4}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {4})).hits[0], 1);
    Drive(bob, MakeSleep(300));
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {4})).hits[0], 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {4})).hits[0], 0);

    /* And the other way around: Bob's 2000ms overwrites Alice's 100ms,
     * every sharer succeeding so the reservation commits. */
    EXPECT_EQ(Drive(alice, MakeCmd(kOpAlloc, {5}, 100)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpAlloc, {5}, 2000)).code, 0);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {5}, 0, 1)).code, 0);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpCommit, {5}, 0, 1)).code, 0);
    Drive(bob, MakeSleep(300));
    EXPECT_EQ(Drive(alice, MakeCmd(kOpLookup, {5})).hits[0], 1);
    EXPECT_EQ(Drive(bob, MakeCmd(kOpLookup, {5})).hits[0], 1);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(bob));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== The worker/scheduler domain ==*/

TEST_F(MultiprocessTest, SchedulerLateJoinDrivesDataThroughWorkers)
{
    const std::string id = MakeUniqueId("ones");
    Child leader;
    Child leaf;
    Child scheduler;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(leader, MakeDict(id, "worker", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(leader));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(leaf, MakeDict(id, "worker", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(leaf));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(scheduler, MakeDict(id, "scheduler", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(scheduler));

    /* Wrong-direction discipline through the live protocol: the worker
     * refuses metadata it does not own. */
    EXPECT_EQ(Drive(leader, MakeCmd(kOpLookup, {1})).code, Status::Unsupported().Underlying());
    EXPECT_EQ(Drive(leader, MakeCmd(kOpAlloc, {1})).code, Status::Unsupported().Underlying());

    /* The scheduler allocates three slots; the token reaches the workers
     * over the parent's pipes, standing in for the deployment's broadcast. */
    const Result alloc = Drive(scheduler, MakeCmd(kOpAlloc, {1, 2, 3}));
    ASSERT_EQ(alloc.code, 0);

    /* The leader fills the slots with the blocks' patterns (job 1). The
     * engine executes asynchronously, so the completion log is polled until
     * the job reports -- the window between submit and report is real. */
    const Result dump = Drive(
        leader,
        MakeTransferCmd(kOpDump, 1, {1, 2, 3}, {alloc.slots[0], alloc.slots[1], alloc.slots[2]}));
    ASSERT_EQ(dump.code, 0);
    const auto settled = PollFinished(leader, 1);
    ASSERT_TRUE(settled.has_value());
    EXPECT_TRUE(settled->second);
    /* A drained job is gone: Wait answers invalid, and the log stays empty. */
    EXPECT_EQ(Drive(leader, MakeJobCmd(kOpWait, 1)).code, Status::InvalidParam().Underlying());
    EXPECT_EQ(Drive(leader, MakeJobCmd(kOpGetFinished, 0)).n, 0);

    /* The scheduler concludes; the blocks are visible and re-pinnable, the
     * pin re-addressing the very same slots. */
    EXPECT_EQ(Drive(scheduler, MakeCmd(kOpCommit, {1, 2, 3}, 0, 1)).code, 0);
    const Result seen = Drive(scheduler, MakeCmd(kOpLookup, {1, 2, 3}));
    ASSERT_EQ(seen.code, 0);
    EXPECT_EQ(seen.hits[0] + seen.hits[1] + seen.hits[2], 3);
    const Result pinned = Drive(scheduler, MakeCmd(kOpPin, {1, 2, 3}));
    ASSERT_EQ(pinned.code, 0);
    EXPECT_EQ(pinned.slots[0], alloc.slots[0]);
    EXPECT_EQ(pinned.slots[1], alloc.slots[1]);
    EXPECT_EQ(pinned.slots[2], alloc.slots[2]);

    /* The leaf loads the committed bytes out of the shared pool (job 2):
     * the submit returns with the job in flight, Wait consumes the
     * completion, and only then does the buffer carry the patterns. */
    const Result load = Drive(
        leaf,
        MakeTransferCmd(kOpLoad, 2, {1, 2, 3}, {alloc.slots[0], alloc.slots[1], alloc.slots[2]}));
    ASSERT_EQ(load.code, 0);
    EXPECT_EQ(load.n, 3);
    EXPECT_EQ(Drive(leaf, MakeJobCmd(kOpWait, 2)).code, 0);
    EXPECT_EQ(Drive(leaf, MakeJobCmd(kOpWait, 2)).code, Status::InvalidParam().Underlying());
    EXPECT_EQ(Drive(leaf, MakeJobCmd(kOpGetFinished, 0)).n, 0);
    const Result verified = Drive(leaf, MakeCmd(kOpVerify, {1, 2, 3}));
    ASSERT_EQ(verified.code, 0);
    EXPECT_EQ(verified.hits[0] + verified.hits[1] + verified.hits[2], 3);

    /* Teardown order: joiners first, the leader last. */
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(scheduler));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(leaf));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(leader));
}

/*== G: the both-domain data plane, replica and primary over one pool ==*/

TEST_F(MultiprocessTest, PrimaryAndReplicaTransferOverTheSharedPool)
{
    const std::string id = MakeUniqueId("nwdata");
    Child primary;
    Child alice;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));

    /* Alice reserves three slots through the forwarded path and dumps the
     * blocks' patterns into the pool (job 1); the completion is polled out
     * of the asynchronous engine. */
    const Result alloc = Drive(alice, MakeCmd(kOpAlloc, {1, 2, 3}));
    ASSERT_EQ(alloc.code, 0);
    const Result dump = Drive(
        alice,
        MakeTransferCmd(kOpDump, 1, {1, 2, 3}, {alloc.slots[0], alloc.slots[1], alloc.slots[2]}));
    ASSERT_EQ(dump.code, 0);
    const auto settled = PollFinished(alice, 1);
    ASSERT_TRUE(settled.has_value());
    EXPECT_TRUE(settled->second);
    EXPECT_EQ(Drive(alice, MakeCmd(kOpCommit, {1, 2, 3}, 0, 1)).code, 0);

    /* The primary pins the committed blocks and lands on the very same
     * slots -- one token addresses one pool for the whole domain. */
    const Result pinned = Drive(primary, MakeCmd(kOpPin, {1, 2, 3}));
    ASSERT_EQ(pinned.code, 0);
    EXPECT_EQ(pinned.slots[0], alloc.slots[0]);
    EXPECT_EQ(pinned.slots[1], alloc.slots[1]);
    EXPECT_EQ(pinned.slots[2], alloc.slots[2]);

    /* The primary loads the committed bytes through its own engine (job 2):
     * Wait consumes the completion, then the patterns are in the buffer. */
    const Result load = Drive(
        primary,
        MakeTransferCmd(kOpLoad, 2, {1, 2, 3}, {alloc.slots[0], alloc.slots[1], alloc.slots[2]}));
    ASSERT_EQ(load.code, 0);
    EXPECT_EQ(Drive(primary, MakeJobCmd(kOpWait, 2)).code, 0);
    const Result verified = Drive(primary, MakeCmd(kOpVerify, {1, 2, 3}));
    ASSERT_EQ(verified.code, 0);
    EXPECT_EQ(verified.hits[0] + verified.hits[1] + verified.hits[2], 3);

    /* The completion was consumed by Wait: the log stays empty. */
    EXPECT_EQ(Drive(primary, MakeJobCmd(kOpGetFinished, 0)).n, 0);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== H: data-plane job discipline: submission errors and id lifecycle ==*/

TEST_F(MultiprocessTest, DataPlaneJobDiscipline)
{
    const std::string id = MakeUniqueId("discipline");
    Child primary;
    Child alice;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSmallBlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));

    /* An empty job is malformed: rejected with nothing started. */
    const Result empty = Drive(alice, MakeTransferCmd(kOpDump, 1, {}, {}));
    EXPECT_EQ(empty.code, Status::InvalidParam().Underlying());

    /* A submitted job holds its id until the completion is consumed: a
     * resubmit under the same id answers DuplicateKey. */
    const Result first = Drive(alice, MakeTransferCmd(kOpDump, 5, {1}, {0}));
    ASSERT_EQ(first.code, 0);
    const Result again = Drive(alice, MakeTransferCmd(kOpDump, 5, {1}, {0}));
    EXPECT_EQ(again.code, Status::DuplicateKey().Underlying());

    /* Consuming the completion (Wait) frees the id for reuse. */
    EXPECT_EQ(Drive(alice, MakeJobCmd(kOpWait, 5)).code, 0);
    const Result reused = Drive(alice, MakeTransferCmd(kOpLoad, 5, {1}, {0}));
    ASSERT_EQ(reused.code, 0);
    EXPECT_EQ(Drive(alice, MakeJobCmd(kOpWait, 5)).code, 0);

    /* Unknown ids are invalid on the data plane too. */
    EXPECT_EQ(Drive(alice, MakeJobCmd(kOpWait, 99)).code, Status::InvalidParam().Underlying());

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

/*== I: engine failures propagate through the data plane ==*/

TEST_F(MultiprocessTest, EngineFailuresPropagateAsCompletions)
{
    const std::string id = MakeUniqueId("fault");
    Child primary;
    Child alice;
    /* The fault engine rides the cache_dma_engine key: every admitted job
     * of this domain settles as a failure, no bytes move. */
    ASSERT_NO_FATAL_FAILURE(
        ForkStoreChild(primary, MakeDict(id, "both", kSmallBlockSize, 2, "gdr")));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(alice, MakeDict(id, "both", kSmallBlockSize, 2, "gdr")));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(alice));

    /* A well-formed dump is admitted (job 1) -- the submission succeeds --
     * and its completion carries the failure. */
    const Result dump = Drive(alice, MakeTransferCmd(kOpDump, 1, {1}, {0}));
    ASSERT_EQ(dump.code, 0);
    const auto settled = PollFinished(alice, 1);
    ASSERT_TRUE(settled.has_value());
    EXPECT_FALSE(settled->second);

    /* The Wait path surfaces the same failure as an error Status. */
    const Result load = Drive(alice, MakeTransferCmd(kOpLoad, 2, {1}, {0}));
    ASSERT_EQ(load.code, 0);
    EXPECT_NE(Drive(alice, MakeJobCmd(kOpWait, 2)).code, 0);

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(alice));
    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

TEST_F(MultiprocessTest, SchedulerDrivenRunsAgreeAcrossFreshDomains)
{
    const std::vector<Cmd> script = BuildDeterminismScript();

    std::vector<std::vector<Result>> runs;
    runs.reserve(2);
    for (const char* label : {"schdom0", "schdom1"}) {
        const std::string id = MakeUniqueId(label);
        Child leader;
        Child scheduler;
        ASSERT_NO_FATAL_FAILURE(
            ForkStoreChild(leader, MakeDict(id, "worker", kSlots6BlockSize, 1)));
        ASSERT_NO_FATAL_FAILURE(ExpectReady(leader));
        ASSERT_NO_FATAL_FAILURE(
            ForkStoreChild(scheduler, MakeDict(id, "scheduler", kSlots6BlockSize, 1)));
        ASSERT_NO_FATAL_FAILURE(ExpectReady(scheduler));

        std::vector<Result> run;
        run.reserve(script.size());
        for (const Cmd& cmd : script) { run.push_back(Drive(scheduler, cmd)); }
        runs.push_back(std::move(run));

        ASSERT_NO_FATAL_FAILURE(ReleaseChild(scheduler));
        ASSERT_NO_FATAL_FAILURE(ReleaseChild(leader));
    }

    /* Two fresh worker/scheduler domains, the same script: the decision
     * maker's answers agree bit for bit. */
    ASSERT_EQ(runs[0].size(), runs[1].size());
    for (std::size_t i = 0; i < script.size(); i++) {
        EXPECT_EQ(std::memcmp(&runs[0][i], &runs[1][i], sizeof(Result)), 0)
            << "step " << i << " diverged between the two scheduler-driven domains";
    }

    /* Spot checks so the equality cannot be two equally wrong runs: the
     * fifo-traced script, with the same expectations as the replica test. */
    EXPECT_EQ(runs[0][0].code, 0);
    EXPECT_EQ(runs[0][0].hits[0], 0);
    EXPECT_EQ(runs[0][3].hits[0] + runs[0][3].hits[1] + runs[0][3].hits[2], 3);
    EXPECT_EQ(runs[0][6].hits[0] + runs[0][6].hits[1] + runs[0][6].hits[2], 0);
    EXPECT_EQ(runs[0][9].n, 6);
    EXPECT_EQ(runs[0][13].hits[0], 1);
    EXPECT_EQ(runs[0][13].hits[1], 1);
    EXPECT_EQ(runs[0][13].hits[2], 0); /* evicted */
    EXPECT_EQ(runs[0][13].hits[3], 1);
    EXPECT_EQ(runs[0][22].code, Status::NoSpace().Underlying()); /* oversize batch */
    EXPECT_EQ(runs[0][23].hits[0], 0); /* block 4 fell to the step-19 re-admission */
    EXPECT_EQ(runs[0][24].hits[0], 1); /* rollback held */
    EXPECT_EQ(runs[0][25].code, Status::NotFound().Underlying()); /* pin of a miss */
}

TEST_F(MultiprocessTest, WrongRoleJoinIsRejectedEndToEnd)
{
    const std::string id = MakeUniqueId("cross");
    Child primary;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(primary, MakeDict(id, "both", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectReady(primary));

    /* Both late roles are refused entry into the both-led domain: the
     * facade surfaces the leader's rejection as the Setup error. */
    Child scheduler;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(scheduler, MakeDict(id, "scheduler", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectSetupFailure(scheduler));
    ASSERT_NO_FATAL_FAILURE(ReleaseFailedChild(scheduler));

    Child worker;
    ASSERT_NO_FATAL_FAILURE(ForkStoreChild(worker, MakeDict(id, "worker", kSlots4BlockSize)));
    ASSERT_NO_FATAL_FAILURE(ExpectSetupFailure(worker));
    ASSERT_NO_FATAL_FAILURE(ReleaseFailedChild(worker));

    ASSERT_NO_FATAL_FAILURE(ReleaseChild(primary));
}

}  // namespace
}  // namespace UC::Store
